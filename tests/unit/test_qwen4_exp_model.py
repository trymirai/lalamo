from itertools import accumulate, pairwise
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
import torch
from jaxtyping import Array, Float, Int
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig, Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpForConditionalGeneration,
    Qwen4ExpTextNGramEmbedding,
)

from lalamo.initializer import EmptyInitializer
from lalamo.model_import.loaders.huggingface import load_huggingface_decoder
from lalamo.model_import.model_configs import HFQwen4ExpConfig
from lalamo.model_import.model_configs.huggingface.qwen4_exp import ngram_head_vocab_sizes, ngram_multipliers
from lalamo.model_import.origins import load_safetensors_weights
from lalamo.module import Keychain
from lalamo.modules import Decoder, DecoderForwardPassConfig
from lalamo.utils.sharding import ShardingConfig
from lalamo.utils.torch_interop import torch_to_jax
from tests.common import assert_close
from tests.unit.modules.qwen4_exp import randomized

EOS_TOKEN_ID = 3
CHUNK_LENGTHS = (9, 1, 6, 1, 7)


def make_text_config() -> Qwen4ExpTextConfig:
    return Qwen4ExpTextConfig(
        hidden_size=32,
        num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "full_attention", "linear_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5},
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        output_gate_type="sigmoid",
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        hc_count=4,
        hc_lowrank=16,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=8,
        indexer_compress_ratio=4,
        ple_layer_ids=[2],
        ple_embed_dim=16,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=1000,
        split_ngram_parts=4,
        vocab_size=128,
        eos_token_id=EOS_TOKEN_ID,
        rms_norm_eps=1e-6,
        max_position_embeddings=256,
        tie_word_embeddings=False,
    )


@pytest.fixture(scope="module")
def hf_model() -> Qwen4ExpForConditionalGeneration:
    config = Qwen4ExpConfig(
        text_config=make_text_config(),
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "num_heads": 2,
            "intermediate_size": 32,
            "out_hidden_size": 32,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "num_position_embeddings": 16,
        },
        tie_word_embeddings=False,
    )
    return randomized(Qwen4ExpForConditionalGeneration(config)).eval()


@pytest.fixture(scope="module")
def checkpoint_path(hf_model: Qwen4ExpForConditionalGeneration, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("qwen4_exp")
    hf_model.save_pretrained(path)
    return path


def load_decoder(checkpoint_path: Path, device: jax.Device) -> Decoder:
    decoder_config = HFQwen4ExpConfig.from_json(checkpoint_path / "config.json").to_decoder_config({})
    decoder = decoder_config.init(
        EmptyInitializer(default_dtype=jnp.float32, sharding_config=ShardingConfig.replicated([device])),
    )
    with load_safetensors_weights(checkpoint_path / "model.safetensors") as (weights, _):
        return load_huggingface_decoder(decoder, weights)


def make_token_ids() -> Int[Array, "batch tokens"]:
    token_ids = jax.random.randint(jax.random.key(0), (2, sum(CHUNK_LENGTHS)), 4, 128, dtype=jnp.int32)
    return token_ids.at[0, jnp.array([5, 6, 17])].set(EOS_TOKEN_ID).at[1, jnp.array([11])].set(EOS_TOKEN_ID)


def hf_logits(
    hf_model: Qwen4ExpForConditionalGeneration,
    token_ids: Int[Array, "batch tokens"],
) -> Float[Array, "batch tokens vocabulary"]:
    with torch.no_grad():
        logits = hf_model(input_ids=torch.tensor(jax.device_get(token_ids).tolist())).logits
    return torch_to_jax(logits)


def token_positions_for(token_ids: Int[Array, "batch tokens"]) -> Int[Array, "batch tokens"]:
    batch_size, num_tokens = token_ids.shape
    return jnp.broadcast_to(jnp.arange(num_tokens, dtype=jnp.int32), (batch_size, num_tokens))


@pytest.mark.parametrize("ple_layer_index", [0, 1])
def test_ngram_hash_constants_match_hf(ple_layer_index: int) -> None:
    hf_config = make_text_config()
    hf_embedding = Qwen4ExpTextNGramEmbedding(hf_config, 16, layer_idx=1, ple_layer_index=ple_layer_index)

    assert ngram_multipliers(128, 3, ple_layer_index, hf_config.seed) == tuple(hf_embedding.layer_multipliers.tolist())
    assert ngram_head_vocab_sizes(1000, 4, ple_layer_index) == tuple(hf_embedding.head_vocab_sizes)


@pytest.mark.parametrize("backend", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
def test_decoder_matches_hf(
    backend: str,
    hf_model: Qwen4ExpForConditionalGeneration,
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

    assert_close(result=result.logits, reference=hf_logits(hf_model, token_ids))


@pytest.mark.parametrize("backend", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
def test_chunked_decoding_matches_hf(
    backend: str,
    hf_model: Qwen4ExpForConditionalGeneration,
    checkpoint_path: Path,
) -> None:
    device, *_ = jax.devices(backend)
    with jax.default_device(device):
        decoder = load_decoder(checkpoint_path, device)
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
