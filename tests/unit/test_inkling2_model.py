import json
from itertools import accumulate, pairwise
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import pytest
import torch
from jaxtyping import Array, Float, Int

# Inkling2 landed in transformers after its latest release.
pytest.importorskip("transformers.models.inkling2_loop")

from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig
from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopForCausalLM

from lalamo.initializer import EmptyInitializer
from lalamo.model_import.loaders.inkling2 import load_inkling2_decoder
from lalamo.model_import.model_configs import HFInkling2LoopConfig
from lalamo.module import Keychain
from lalamo.modules import Decoder, DecoderForwardPassConfig
from lalamo.utils.sharding import ShardingConfig
from lalamo.utils.torch_interop import torch_to_jax
from tests.common import assert_close

VOCAB_SIZE = 96
UNPADDED_VOCAB_SIZE = 90
EOS_TOKEN_ID = 3
LOCAL_LAYER_IDS = (0, 2, 4)
CHUNK_LENGTHS = (9, 1, 6, 1, 7)

TEXT_CONFIG: dict[str, Any] = {
    "hidden_size": 32,
    "num_hidden_layers": 6,
    "vocab_size": VOCAB_SIZE,
    "unpadded_vocab_size": UNPADDED_VOCAB_SIZE,
    "rms_norm_eps": 1e-6,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "swa_num_attention_heads": 4,
    "swa_num_key_value_heads": 2,
    "swa_head_dim": 8,
    "sliding_window_size": 8,
    "local_layer_ids": list(LOCAL_LAYER_IDS),
    "d_rel": 4,
    "rel_extent": 6,
    "log_scaling_n_floor": 4,
    "log_scaling_alpha": 0.5,
    "qk_l2_clip_threshold": 1.5,
    "rel_l2_clip_threshold": 0.5,
    "sconv_kernel_size": 4,
    "q_bias": False,
    "o_bias": False,
    "use_value_norm_no_gamma": True,
    "n_layers_per_attn_res_block": 2,
    "attn_res_normalize_values": True,
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    "n_shared_experts": 0,
    "intermediate_size": 16,
    "mlp_gate_softcap": 0.5,
    "mlp_up_softcap": 0.8,
    "route_scale": 2.0,
    "use_gate_bias": True,
    "final_logit_softcapping": 3.0,
    "model_max_length": 1024,
    "loop_config": {"core_start": 2, "core_end": 4, "num_passes": 3, "gate_rank": 4, "is_varloop": False},
}


def make_hf_config() -> Inkling2LoopTextConfig:
    text = TEXT_CONFIG
    loop = cast("dict[str, int]", text["loop_config"])
    config = Inkling2LoopTextConfig(
        vocab_size=VOCAB_SIZE,
        unpadded_vocab_size=UNPADDED_VOCAB_SIZE,
        hidden_size=text["hidden_size"],
        num_hidden_layers=text["num_hidden_layers"],
        num_attention_heads=text["num_attention_heads"],
        num_key_value_heads=text["num_key_value_heads"],
        head_dim=text["head_dim"],
        swa_num_attention_heads=text["swa_num_attention_heads"],
        swa_num_key_value_heads=text["swa_num_key_value_heads"],
        swa_head_dim=text["swa_head_dim"],
        sliding_window_size=text["sliding_window_size"],
        d_rel=text["d_rel"],
        rel_extent=text["rel_extent"],
        log_scaling_n_floor=text["log_scaling_n_floor"],
        log_scaling_alpha=text["log_scaling_alpha"],
        layer_types=[
            "hybrid_sliding" if layer_index in LOCAL_LAYER_IDS else "hybrid"
            for layer_index in range(text["num_hidden_layers"])
        ],
        rms_norm_eps=text["rms_norm_eps"],
        conv_kernel_size=text["sconv_kernel_size"],
        moe_intermediate_size=text["intermediate_size"],
        n_routed_experts=text["n_routed_experts"],
        num_experts_per_tok=text["num_experts_per_tok"],
        route_scale=text["route_scale"],
        eos_token_id=EOS_TOKEN_ID,
        qk_l2_clip_threshold=text["qk_l2_clip_threshold"],
        rel_l2_clip_threshold=text["rel_l2_clip_threshold"],
        n_layers_per_attn_res_block=text["n_layers_per_attn_res_block"],
        mlp_gate_softcap=text["mlp_gate_softcap"],
        mlp_up_softcap=text["mlp_up_softcap"],
        final_logit_softcapping=text["final_logit_softcapping"],
        loop_layer_index_start=loop["core_start"],
        loop_layer_index_end=loop["core_end"],
        loop_num_passes=loop["num_passes"],
        loop_gate_rank=loop["gate_rank"],
    )
    config._attn_implementation = "eager"  # noqa: SLF001
    return config


@pytest.fixture(scope="module")
def hf_model() -> Inkling2LoopForCausalLM:
    torch.manual_seed(0)
    model = Inkling2LoopForCausalLM(make_hf_config()).eval()
    with torch.no_grad():
        for name, tensor in (*model.named_parameters(), *model.named_buffers()):
            if not tensor.is_floating_point():
                continue
            # Scales and norms stay near one so that clipping and normalization see realistic inputs.
            offset = 1.0 if name.endswith(("norm.weight", "layernorm.weight", ".scale")) else 0.0
            tensor.copy_(torch.randn_like(tensor) * 0.3 + offset)
    return model


def _to_mlx_lm_layout(name: str, tensor: torch.Tensor) -> tuple[str, torch.Tensor]:
    if name == "lm_head.weight":
        return name, tensor
    renames = (
        ("model.", "model.language_model."),
        ("embed_tokens.embed_norm.", "embed_norm."),
        (".q_proj.linear.", ".q_proj."),
        (".q_proj.norm.scale", ".q_l2_clip.scale"),
        (".k_proj.linear.", ".k_proj."),
        (".v_proj.linear.", ".v_proj."),
        (".k_proj.sconv.", ".k_sconv."),
        (".v_proj.sconv.", ".v_sconv."),
        (".o_proj.gate_proj.", ".gate_proj."),
        (".o_proj.linear.", ".o_proj."),
        (".relative_position_bias.linear.", ".r_proj."),
        (".relative_position_bias.norm.scale", ".rel_l2_clip.scale"),
        (".relative_position_bias.proj", ".rel_logits_proj.proj"),
    )
    for old, new in renames:
        name = name.replace(old, new, 1)
    if name.endswith("conv1d.weight"):
        # PyTorch stores depthwise conv weights as [channels, 1, taps]; MLX as [channels, taps, 1].
        return name, tensor.transpose(1, 2)
    if name.endswith("mlp.experts.gate_up_proj"):
        # Transformers splits gate and up halves; the checkpoints interleave their rows.
        gate, up = tensor.chunk(2, dim=1)
        return f"{name}.weight", torch.stack([gate, up], dim=2).flatten(1, 2)
    if name.endswith("mlp.experts.down_proj"):
        return f"{name}.weight", tensor
    return name, tensor


@pytest.fixture(scope="module")
def mlx_lm_weights(hf_model: Inkling2LoopForCausalLM) -> dict[str, Array]:
    converted = (_to_mlx_lm_layout(name, tensor) for name, tensor in hf_model.state_dict().items())
    return {name: torch_to_jax(tensor.detach().contiguous()) for name, tensor in converted}


@pytest.fixture(scope="module")
def config_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("inkling2") / "config.json"
    path.write_text(
        json.dumps({"model_type": "inkling2_loop", "eos_token_id": EOS_TOKEN_ID, "text_config": TEXT_CONFIG})
    )
    return path


def load_decoder(config_path: Path, weights: dict[str, Array], device: jax.Device) -> Decoder:
    decoder_config = HFInkling2LoopConfig.from_json(config_path).to_decoder_config({})
    decoder = decoder_config.init(
        EmptyInitializer(default_dtype=jnp.float32, sharding_config=ShardingConfig.replicated([device])),
    )
    return load_inkling2_decoder(decoder, weights)


def make_token_ids() -> Int[Array, "batch tokens"]:
    return jax.random.randint(jax.random.key(0), (2, sum(CHUNK_LENGTHS)), 4, UNPADDED_VOCAB_SIZE, dtype=jnp.int32)


def token_positions_for(token_ids: Int[Array, "batch tokens"]) -> Int[Array, "batch tokens"]:
    batch_size, num_tokens = token_ids.shape
    return jnp.broadcast_to(jnp.arange(num_tokens, dtype=jnp.int32), (batch_size, num_tokens))


def hf_logits(
    hf_model: Inkling2LoopForCausalLM,
    token_ids: Int[Array, "batch tokens"],
) -> Float[Array, "batch tokens vocabulary"]:
    with torch.no_grad():
        logits = hf_model(input_ids=torch.tensor(jax.device_get(token_ids).tolist())).logits
    return torch_to_jax(logits)


def test_decoder_matches_hf(
    hf_model: Inkling2LoopForCausalLM,
    mlx_lm_weights: dict[str, Array],
    config_path: Path,
) -> None:
    device, *_ = jax.devices("cpu")
    with jax.default_device(device):
        decoder = load_decoder(config_path, mlx_lm_weights, device)
        token_ids = make_token_ids()
        with jax.set_mesh(decoder.sharding_config.mesh):
            result = decoder(
                token_ids,
                token_positions_for(token_ids),
                forward_pass_config=DecoderForwardPassConfig.for_tracer_tests(),
                keychain=Keychain.init(0, sharding_config=decoder.sharding_config),
            )

    assert_close(result=result.logits, reference=hf_logits(hf_model, token_ids))


def test_chunked_decoding_matches_hf(
    hf_model: Inkling2LoopForCausalLM,
    mlx_lm_weights: dict[str, Array],
    config_path: Path,
) -> None:
    device, *_ = jax.devices("cpu")
    with jax.default_device(device):
        decoder = load_decoder(config_path, mlx_lm_weights, device)
        token_ids = make_token_ids()
        token_positions = token_positions_for(token_ids)
        batch_size, num_tokens = token_ids.shape
        state = decoder.init_static_state(batch_size, num_tokens, jnp.float32)
        chunk_logits = []
        with jax.set_mesh(decoder.sharding_config.mesh):
            for start, end in pairwise(accumulate(CHUNK_LENGTHS, initial=0)):
                result = decoder(
                    token_ids[:, start:end],
                    token_positions[:, start:end],
                    state=state,
                    return_updated_state=True,
                    forward_pass_config=DecoderForwardPassConfig.for_tracer_tests(),
                    keychain=Keychain.init(0, sharding_config=decoder.sharding_config),
                )
                state = result.updated_state
                chunk_logits.append(result.logits)

    assert_close(result=jnp.concatenate(chunk_logits, axis=1), reference=hf_logits(hf_model, token_ids))
