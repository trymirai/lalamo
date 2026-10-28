import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jaxtyping import Array, Float
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextAttention, Qwen4ExpTextRotaryEmbedding

from lalamo.initializer import EmptyInitializer
from lalamo.model_import.loaders.huggingface import load_sparse_attention
from lalamo.module import Keychain
from lalamo.modules.linear import LinearConfig
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.rope import PositionalEmbeddings, UnscaledRoPEConfig
from lalamo.modules.token_mixer import AttentionImplementation, MixerForwardPassConfig
from lalamo.modules.token_mixers.attention import Attention, AttentionConfig
from lalamo.modules.token_mixers.sparse_attention import (
    SparseAttention,
    SparseAttentionConfig,
    SparseAttentionIndexerConfig,
    SparseAttentionStateLayer,
)
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.torch_interop import torch_to_jax
from tests.common import assert_close
from tests.helpers import make_test_sharding_config

HIDDEN_SIZE = 64
NUM_HEADS = 4
NUM_GROUPS = 2
HEAD_DIM = 32
ROPE_DIM = 8
INDEXER_HEADS = 4
INDEXER_HEAD_DIM = 16
COMPRESSION_RATIO = 4
TOKEN_BUDGET = 16
SEQUENCE_LENGTH = 64
ROPE_BASE = 10000.0

pytestmark = pytest.mark.usefixtures("fake_mesh")


def make_hf_attention() -> tuple[Qwen4ExpTextAttention, Qwen4ExpTextRotaryEmbedding]:
    config = Qwen4ExpTextConfig(
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_GROUPS,
        head_dim=HEAD_DIM,
        num_hidden_layers=4,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": ROPE_BASE,
            "partial_rotary_factor": ROPE_DIM / HEAD_DIM,
        },
        indexer_n_heads=INDEXER_HEADS,
        indexer_kv_heads=1,
        indexer_head_dim=INDEXER_HEAD_DIM,
        indexer_budget=TOKEN_BUDGET,
        indexer_compress_ratio=COMPRESSION_RATIO,
        rms_norm_eps=1e-6,
        vocab_size=128,
    )
    torch.manual_seed(0)
    attention = Qwen4ExpTextAttention(config, layer_idx=3)
    with torch.no_grad():
        for name, parameter in attention.named_parameters():
            if name.startswith("indexer.index_qk_proj"):
                parameter.copy_(torch.rand_like(parameter) * 0.2)
            else:
                parameter.copy_(torch.randn_like(parameter) * 0.2)
    return attention, Qwen4ExpTextRotaryEmbedding(config)


def make_lalamo_attention(hf_attention: Qwen4ExpTextAttention) -> SparseAttention:
    norm_config = NormalizationConfig(
        epsilon=1e-6,
        scale_offset=1.0,
        upcast_mode=UpcastMode.FULL_LAYER,
        subtract_mean=False,
    )
    config = SparseAttentionConfig(
        qkvg_projection_config=LinearConfig(),
        out_projection_config=LinearConfig(),
        query_norm_config=norm_config,
        key_norm_config=norm_config,
        indexer_config=SparseAttentionIndexerConfig(
            projection_config=LinearConfig(),
            query_norm_config=norm_config,
            key_norm_config=norm_config,
            num_heads=INDEXER_HEADS,
            head_dim=INDEXER_HEAD_DIM,
            compression_ratio=COMPRESSION_RATIO,
            token_budget=TOKEN_BUDGET,
        ),
        num_heads=NUM_HEADS,
        num_groups=NUM_GROUPS,
        head_dim=HEAD_DIM,
    )
    module = config.init(
        EmptyInitializer(default_dtype=jnp.float32, sharding_config=make_test_sharding_config()),
        model_dim=HIDDEN_SIZE,
    )
    weights = {name: torch_to_jax(tensor.detach()) for name, tensor in hf_attention.state_dict().items()}
    return load_sparse_attention(module, weights, ParameterPath())


def dense_counterpart(module: SparseAttention) -> Attention:
    config = module.config
    dense_config = AttentionConfig(
        qkvg_projection_config=config.qkvg_projection_config,
        out_projection_config=config.out_projection_config,
        query_norm_config=config.query_norm_config,
        key_norm_config=config.key_norm_config,
        num_heads=config.num_heads,
        num_groups=config.num_groups,
        head_dim=config.head_dim,
        is_causal=True,
        scale=None,
        sliding_window_size=None,
        logit_soft_cap=None,
        has_sinks=False,
        has_qkvg_biases=False,
        has_out_biases=False,
        has_gate=True,
    )
    return Attention(
        config=dense_config,
        sharding_config=module.sharding_config,
        qkvg_projection=module.qkvg_projection,
        out_projection=module.out_projection,
        query_norm=module.query_norm,
        key_norm=module.key_norm,
        sinks=None,
    )


def make_hidden_states(num_tokens: int) -> Float[Array, "tokens channels"]:
    return jax.random.uniform(jax.random.key(1), (num_tokens, HIDDEN_SIZE), dtype=jnp.float32)


def positional_embeddings(start: int, num_tokens: int) -> PositionalEmbeddings:
    rope_config = UnscaledRoPEConfig(base=ROPE_BASE, max_sequence_length=2 * SEQUENCE_LENGTH, head_dim=ROPE_DIM)
    return rope_config.compute_positional_embeddings(jnp.arange(start, start + num_tokens))


def hf_reference(
    attention: Qwen4ExpTextAttention,
    rotary: Qwen4ExpTextRotaryEmbedding,
    hidden_states: Float[Array, "tokens channels"],
) -> Float[Array, "tokens channels"]:
    num_tokens, _ = hidden_states.shape
    torch_hidden_states = torch.from_numpy(np.asarray(jax.device_get(hidden_states)))[None]
    position_embeddings = rotary(torch_hidden_states, torch.arange(num_tokens)[None])
    is_visible = torch.ones(1, 1, num_tokens, num_tokens, dtype=torch.bool).tril()
    causal_mask = torch.zeros(is_visible.shape).masked_fill(~is_visible, torch.finfo(torch.float32).min)
    with torch.no_grad():
        outputs, _ = attention(torch_hidden_states, position_embeddings, causal_mask)
    return torch_to_jax(outputs[0])


def run_in_chunks(
    module: SparseAttention,
    hidden_states: Float[Array, "tokens channels"],
    chunk_lengths: tuple[int, ...],
    state: SparseAttentionStateLayer | None,
    num_padding_tokens: int = 0,
    forward_pass_config: MixerForwardPassConfig = MixerForwardPassConfig(),
) -> Float[Array, "tokens channels"]:
    outputs = []
    start = 0
    for chunk_index, length in enumerate(chunk_lengths):
        padding = jnp.zeros((num_padding_tokens, HIDDEN_SIZE), dtype=hidden_states.dtype)
        result = module(
            jnp.concatenate([hidden_states[start : start + length], padding]),
            positional_embeddings(start, length + num_padding_tokens),
            state=state,
            return_updated_state=True,
            length_without_padding=length,
            forward_pass_config=forward_pass_config,
            keychain=Keychain.init(chunk_index, sharding_config=make_test_sharding_config()),
        )
        outputs.append(result.outputs[:length])
        state = result.state
        start += length
    return jnp.concatenate(outputs)


@pytest.mark.parametrize(
    "implementation",
    [AttentionImplementation.XLA, AttentionImplementation.STABLE_REDUCTION],
)
def test_full_prefill_matches_hf(implementation: AttentionImplementation) -> None:
    hf_attention, rotary = make_hf_attention()
    module = make_lalamo_attention(hf_attention)
    hidden_states = make_hidden_states(SEQUENCE_LENGTH)

    result = run_in_chunks(
        module,
        hidden_states,
        (SEQUENCE_LENGTH,),
        state=None,
        forward_pass_config=MixerForwardPassConfig(attention_implementation=implementation),
    )

    assert_close(result=result, reference=hf_reference(hf_attention, rotary, hidden_states))


@pytest.mark.parametrize("num_padding_tokens", [0, 5])
def test_chunked_prefill_with_static_state_matches_hf(num_padding_tokens: int) -> None:
    hf_attention, rotary = make_hf_attention()
    module = make_lalamo_attention(hf_attention)
    hidden_states = make_hidden_states(SEQUENCE_LENGTH)
    state = module.init_static_state(capacity=SEQUENCE_LENGTH + 8, dtype=jnp.float32)

    result = eqx.filter_jit(run_in_chunks)(
        module,
        hidden_states,
        (13, 1, 19, 31),
        state,
        num_padding_tokens,
    )

    assert_close(result=result, reference=hf_reference(hf_attention, rotary, hidden_states))


def test_token_by_token_decode_with_dynamic_state_matches_hf() -> None:
    hf_attention, rotary = make_hf_attention()
    module = make_lalamo_attention(hf_attention)
    hidden_states = make_hidden_states(SEQUENCE_LENGTH)
    prefill_length = 21

    result = run_in_chunks(
        module,
        hidden_states,
        (prefill_length,) + (1,) * (SEQUENCE_LENGTH - prefill_length),
        state=None,
    )

    assert_close(result=result, reference=hf_reference(hf_attention, rotary, hidden_states))


def test_matches_dense_attention_only_within_token_budget() -> None:
    hf_attention, _ = make_hf_attention()
    module = make_lalamo_attention(hf_attention)
    hidden_states = make_hidden_states(SEQUENCE_LENGTH)
    keychain = Keychain.init(0, sharding_config=make_test_sharding_config())

    sparse_outputs = module(hidden_states, positional_embeddings(0, SEQUENCE_LENGTH), keychain=keychain).outputs
    dense_outputs = dense_counterpart(module)(
        hidden_states,
        positional_embeddings(0, SEQUENCE_LENGTH),
        keychain=keychain,
    ).outputs

    num_dense_tokens = TOKEN_BUDGET + COMPRESSION_RATIO - 1
    assert_close(result=sparse_outputs[:num_dense_tokens], reference=dense_outputs[:num_dense_tokens])
    assert not jnp.allclose(sparse_outputs[num_dense_tokens:], dense_outputs[num_dense_tokens:], atol=1e-3)
