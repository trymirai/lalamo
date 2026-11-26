import json
from itertools import accumulate, chain, pairwise
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
import torch
from jaxtyping import Array, Float, Int
from transformers.core_model_loading import Inkling2LoopDequantizeExperts
from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig
from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopForCausalLM

from lalamo.initializer import EmptyInitializer
from lalamo.model_import.loaders.inkling import load_inkling2_loop_decoder
from lalamo.model_import.loaders.utils import dequantize_experts
from lalamo.model_import.model_configs import HFInkling2LoopConfig
from lalamo.model_import.origins import load_safetensors_weights
from lalamo.module import ForwardPassMode, Keychain
from lalamo.modules import Decoder, DecoderForwardPassConfig
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.sharding import ShardingConfig
from lalamo.utils.torch_interop import torch_to_jax
from tests.common import assert_close

UNPADDED_VOCAB_SIZE = 120
NUM_LAYERS = 6
CHUNK_LENGTHS = (8, 1, 1, 1, 1, 1, 1, 1, 1, 5, 3)
LOCAL_LAYER_IDS = (0, 2, 4)
TEXT_CONFIG = {
    "model_type": "inkling2_loop_text",
    "hidden_size": 32,
    "num_hidden_layers": NUM_LAYERS,
    "vocab_size": 128,
    "unpadded_vocab_size": UNPADDED_VOCAB_SIZE,
    "rms_norm_eps": 1e-6,
    "final_logit_softcapping": 30.0,
    "eos_token_id": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "swa_num_attention_heads": 4,
    "swa_num_key_value_heads": 2,
    "swa_head_dim": 8,
    "sliding_window_size": 6,
    "local_layer_ids": list(LOCAL_LAYER_IDS),
    "d_rel": 4,
    "rel_extent": 8,
    "log_scaling_n_floor": 4,
    "log_scaling_alpha": 0.1,
    "qk_l2_clip_threshold": 2.0,
    "rel_l2_clip_threshold": 1.0,
    "sconv_kernel_size": 4,
    "q_bias": False,
    "o_bias": False,
    "use_value_norm_no_gamma": True,
    "n_layers_per_attn_res_block": 2,
    "attn_res_normalize_values": True,
    "loop_config": {"core_start": 2, "core_end": 4, "num_passes": 3, "gate_rank": 8, "is_varloop": False},
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "n_shared_experts": 0,
    "intermediate_size": 16,
    "mlp_gate_softcap": 4.0,
    "mlp_up_softcap": 25.0,
    "route_scale": 8.0,
    "use_gate_bias": True,
}


@pytest.fixture(scope="module")
def hf_model() -> Inkling2LoopForCausalLM:
    config = Inkling2LoopTextConfig.from_dict(
        {
            **TEXT_CONFIG,
            "layer_types": [
                "hybrid_sliding" if layer_index in LOCAL_LAYER_IDS else "hybrid" for layer_index in range(NUM_LAYERS)
            ],
        },
    )
    model = Inkling2LoopForCausalLM(config).eval()
    torch.manual_seed(0)
    with torch.no_grad():
        for tensor in chain(model.parameters(), model.buffers()):
            if tensor.is_floating_point():
                tensor.copy_(torch.randn_like(tensor) * 0.2)
    return model


@pytest.fixture(scope="module")
def checkpoint_path(hf_model: Inkling2LoopForCausalLM, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("inkling2_loop")
    hf_model.save_pretrained(path)
    (path / "config.json").write_text(
        json.dumps({"model_type": "inkling2_loop", "eos_token_id": 3, "text_config": TEXT_CONFIG}),
    )
    return path


def load_decoder(checkpoint_path: Path, device: jax.Device, dtype: jnp.dtype = jnp.dtype(jnp.float32)) -> Decoder:
    hf_config = HFInkling2LoopConfig.from_json(checkpoint_path / "config.json")
    decoder = hf_config.to_decoder_config({}).init(
        EmptyInitializer(default_dtype=dtype, sharding_config=ShardingConfig.replicated([device])),
    )
    with load_safetensors_weights(checkpoint_path / "model.safetensors") as (weights, _):
        return load_inkling2_loop_decoder(
            decoder,
            weights,
            loop_start=hf_config.loop_config.core_start,
        )


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


BACKEND_TOLERANCES = [
    pytest.param("cpu", 1e-5, 1e-4, id="cpu"),
    pytest.param("gpu", None, None, id="gpu", marks=pytest.mark.gpu),
]


@pytest.mark.parametrize(("backend", "atol", "rtol"), BACKEND_TOLERANCES)
def test_decoder_matches_hf(
    backend: str,
    atol: float | None,
    rtol: float | None,
    hf_model: Inkling2LoopForCausalLM,
    checkpoint_path: Path,
) -> None:
    device, *_ = jax.devices(backend)
    with jax.default_device(device):
        decoder = load_decoder(checkpoint_path, device)
        token_ids = make_token_ids()
        with jax.set_mesh(decoder.sharding_config.mesh):
            result = decoder(
                token_ids,
                token_positions_for(token_ids),
                forward_pass_config=DecoderForwardPassConfig.for_tracer_tests(),
                keychain=Keychain.init(0, sharding_config=decoder.sharding_config),
            )

    assert_close(result=result.logits, reference=hf_logits(hf_model, token_ids), atol=atol, rtol=rtol)


def chunked_logits(
    decoder: Decoder,
    token_ids: Int[Array, "batch tokens"],
    state_dtype: jnp.dtype,
    prefill_config: DecoderForwardPassConfig,
    decode_config: DecoderForwardPassConfig,
) -> Float[Array, "batch tokens vocabulary"]:
    token_positions = token_positions_for(token_ids)
    batch_size, num_tokens = token_ids.shape
    state = decoder.init_static_state(batch_size, num_tokens, state_dtype)
    chunk_logits = []
    with jax.set_mesh(decoder.sharding_config.mesh):
        for start, end in pairwise(accumulate(CHUNK_LENGTHS, initial=0)):
            result = decoder(
                token_ids[:, start:end],
                token_positions[:, start:end],
                state=state,
                return_updated_state=True,
                forward_pass_config=decode_config if end - start == 1 else prefill_config,
                keychain=Keychain.init(0, sharding_config=decoder.sharding_config),
            )
            state = result.updated_state
            chunk_logits.append(result.logits)
    return jnp.concatenate(chunk_logits, axis=1).astype(jnp.float32)


def test_bfloat16_chunked_inference_tracks_hf(hf_model: Inkling2LoopForCausalLM, checkpoint_path: Path) -> None:
    device, *_ = jax.devices("cpu")
    decoder = load_decoder(checkpoint_path, device, jnp.dtype(jnp.bfloat16))
    token_ids = make_token_ids()
    logits = chunked_logits(
        decoder,
        token_ids,
        jnp.dtype(jnp.bfloat16),
        DecoderForwardPassConfig.for_inference(ForwardPassMode.MULTI_TOKEN),
        DecoderForwardPassConfig.for_inference(ForwardPassMode.SINGLE_TOKEN),
    )

    assert_close(
        result=logits,
        reference=hf_logits(hf_model, token_ids),
        atol=0.05,
        rtol=0.05,
        fraction_of_allowed_violations=0.01,
    )


@pytest.mark.parametrize(("backend", "atol", "rtol"), BACKEND_TOLERANCES)
def test_chunked_decoding_matches_hf(
    backend: str,
    atol: float | None,
    rtol: float | None,
    hf_model: Inkling2LoopForCausalLM,
    checkpoint_path: Path,
) -> None:
    device, *_ = jax.devices(backend)
    with jax.default_device(device):
        decoder = load_decoder(checkpoint_path, device)
        token_ids = make_token_ids()
        logits = chunked_logits(
            decoder,
            token_ids,
            jnp.dtype(jnp.float32),
            DecoderForwardPassConfig.for_tracer_tests(),
            DecoderForwardPassConfig.for_tracer_tests(),
        )

    assert_close(result=logits, reference=hf_logits(hf_model, token_ids), atol=atol, rtol=rtol)


def test_suffix_prefill_matches_last_token_logits(checkpoint_path: Path) -> None:
    device, *_ = jax.devices("cpu")
    decoder = load_decoder(checkpoint_path, device)
    token_ids = make_token_ids()
    batch_size, num_tokens = token_ids.shape
    lengths = jnp.array([num_tokens, num_tokens - 3], dtype=jnp.int32)
    forward_pass_config = DecoderForwardPassConfig.for_tracer_tests()
    keychain = Keychain.init(0, sharding_config=decoder.sharding_config)
    with jax.set_mesh(decoder.sharding_config.mesh):
        full_logits = decoder(
            token_ids,
            token_positions_for(token_ids),
            forward_pass_config=forward_pass_config,
            keychain=keychain,
        ).logits
        suffix_logits = decoder(
            token_ids,
            token_positions_for(token_ids),
            state=decoder.init_static_state(batch_size, num_tokens, jnp.float32),
            return_updated_state=True,
            lengths_without_padding=lengths,
            return_suffix_tokens=1,
            forward_pass_config=forward_pass_config,
            keychain=keychain,
        ).logits

    assert_close(result=suffix_logits[:, 0], reference=full_logits[jnp.arange(batch_size), lengths - 1])


def float8_to_jax(tensor: torch.Tensor) -> Array:
    return jnp.asarray(tensor.view(torch.uint8).numpy()).view(jnp.float8_e4m3fn)


def quantized_experts(quantization: str) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    num_experts, rows, columns = 9, 4, 64
    if quantization == "nvfp4":
        return {
            "weight": torch.randint(0, 256, (num_experts, rows, columns // 2), dtype=torch.uint8, generator=generator),
            "weight_scale": (torch.rand(num_experts, rows, columns // 16, generator=generator) + 0.1).to(
                torch.float8_e4m3fn,
            ),
            "weight_amax": torch.rand(num_experts, generator=generator) + 0.5,
            "weight_outer_scale_denominator": torch.tensor([3.0]),
        }
    return {
        "weight": torch.randn(num_experts, rows, columns, generator=generator).to(torch.float8_e4m3fn),
        "weight_scale": torch.randint(
            120, 134, (num_experts, rows, columns // 32), dtype=torch.uint8, generator=generator
        ),
    }


@pytest.mark.parametrize("quantization", ["nvfp4", "mxfp8"])
def test_expert_dequantization_matches_hf(quantization: str) -> None:
    tensors = quantized_experts(quantization)
    source_patterns = ["weight", "weight_scale", "weight_amax", "weight_outer_scale_denominator"]
    reference = Inkling2LoopDequantizeExperts().convert(tensors, source_patterns, ["experts"])["experts"]
    weights_dict: dict[str, Array] = {
        ParameterPath(name): float8_to_jax(tensor) if tensor.dtype == torch.float8_e4m3fn else torch_to_jax(tensor)
        for name, tensor in tensors.items()
    }

    result = dequantize_experts(weights_dict, ParameterPath("weight"), experts_per_chunk=4)

    assert jnp.array_equal(result.astype(jnp.bfloat16), torch_to_jax(reference))
