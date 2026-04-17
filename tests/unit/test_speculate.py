from typing import ClassVar, Self

import jax
import jax.numpy as jnp
import pytest

from lalamo.modules import (
    Decoder,
    DecoderConfig,
    DenseMLPConfig,
    FullPrecisionLinearConfig,
    Identity,
    NormalizationConfig,
    TiedEmbeddingConfig,
    TransformerConfig,
    TransformerLayerConfig,
    UnscaledRoPEConfig,
    UpcastMode,
)
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.speculator.common import LMState, SamplerConfig
from lalamo.speculator.sampler import GumbelSeed
from lalamo.speculator.speculate import SpeculationRun
from lalamo.speculator.trie import TreeSpeculator, TrieNode
from tests.common import assert_close


class EmptyTreeSpeculator(TreeSpeculator):
    name: ClassVar[str] = "empty"

    def draft(self, lm: LMState) -> TrieNode:
        return TrieNode(token=lm.bonus, seed=self.seed.value)

    def serialize(self) -> bytes:
        return b""

    @classmethod
    def deserialize_impl(cls, data: bytes, **kwargs: object) -> Self:
        raise NotImplementedError


@pytest.fixture
def decoder() -> Decoder:
    precision = jnp.float32
    model_dim = 8
    hidden_dim = 16
    vocab_size = 32
    context_length = 128

    norm_config = NormalizationConfig(
        scale_precision=precision,
        accumulation_precision=precision,
        epsilon=1e-5,
        scale_offset=None,
        upcast_mode=UpcastMode.ONLY_NORMALIZATION,
        subtract_mean=False,
    )
    attention_config = AttentionConfig(
        qkv_projection_config=FullPrecisionLinearConfig(precision=precision),
        out_projection_config=FullPrecisionLinearConfig(precision=precision),
        query_norm_config=None,
        key_norm_config=None,
        num_heads=2,
        num_groups=2,
        head_dim=4,
        is_causal=True,
        scale=None,
        sliding_window_size=None,
        logit_soft_cap=None,
        has_sinks=False,
        has_qkv_biases=False,
        has_out_biases=False,
        use_rope=True,
    )
    mlp_config = DenseMLPConfig(
        linear_config=FullPrecisionLinearConfig(precision=precision),
        activation=Identity(),
        has_up_biases=False,
        has_down_biases=False,
        gate_clipping=None,
        up_clipping=None,
    )
    layer_config = TransformerLayerConfig(
        pre_mixer_norm_config=norm_config,
        mixer_config=attention_config,
        post_mixer_norm_config=None,
        pre_mlp_norm_config=norm_config,
        mlp_config=mlp_config,
        post_mlp_norm_config=None,
    )
    transformer_config = TransformerConfig(
        global_rope_config=UnscaledRoPEConfig(
            precision=precision,
            base=10_000.0,
            max_sequence_length=context_length,
        ),
        local_rope_config=None,
        layer_configs=(layer_config,),
        output_norm_config=norm_config,
        model_dim=model_dim,
        hidden_dim=hidden_dim,
        context_length=context_length,
    )
    decoder_config = DecoderConfig(
        embedding_config=TiedEmbeddingConfig(
            input_scale=None,
            logit_soft_cap=None,
            precision=precision,
        ),
        transformer_config=transformer_config,
        vocab_size=vocab_size,
    )
    return decoder_config.random_init(key=jax.random.key(7))


def test_lm_state_matches_reference_forward_after_bonus_only_step(decoder: Decoder) -> None:
    prompt_ids = [1, 2, 3]
    config = SamplerConfig(width=2, K=2, max_tokens=16)
    speculator = EmptyTreeSpeculator(
        decoder=decoder,
        config=config,
        eos_set=frozenset(),
        seed=GumbelSeed(config.seed),
    )
    run = SpeculationRun(speculator, prompt_ids)
    _ = next(iter(run))

    emitted = list(run.result.generated)
    # Empty speculator has no drafts; only the bonus is emitted.
    assert len(emitted) == 1

    full_seq = jnp.array([prompt_ids + emitted], dtype=jnp.int32)
    positions = jnp.arange(full_seq.shape[1], dtype=jnp.int32)[None, :]
    ref = decoder(full_seq, positions, state=None, return_updated_state=True)

    assert_close(
        result=run.lm_state.logits,
        reference=ref.logits[0, -1],
        operation_name="post-step lm_state.logits vs reference forward",
    )
    assert run.lm_state.position == len(prompt_ids) + len(emitted)
