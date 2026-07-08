import jax
import jax.numpy as jnp
import pytest

from lalamo.initializer import RandomInitializer
from lalamo.modules import (
    Decoder,
    DecoderConfig,
    DecoderForwardPassConfig,
    DenseMLPConfig,
    EmbeddingForwardPassConfig,
    ForwardPassMode,
    Identity,
    Keychain,
    LinearConfig,
    NormalizationConfig,
    SiLU,
    TiedEmbeddingConfig,
    TokenMixerConfig,
    TransformerConfig,
    TransformerForwardPassConfig,
    TransformerLayerConfig,
    UnscaledRoPEConfig,
    UpcastMode,
)
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.modules.token_mixers.convolutions import SeparableCausalConvConfig
from lalamo.modules.token_mixers.deltanet import DeltaNetConfig
from lalamo.modules.token_mixers.mamba import Mamba2Config
from lalamo.utils.sharding import ShardingConfig
from tests.common import assert_close
from tests.helpers import make_test_sharding_config

CONTEXT_LENGTH = 16


def replicated_sharding_config() -> ShardingConfig:
    return ShardingConfig.replicated(jax.devices("cpu")[:8])


NORM_CONFIG = NormalizationConfig(
    epsilon=1e-5,
    scale_offset=None,
    upcast_mode=UpcastMode.ONLY_NORMALIZATION,
    subtract_mean=False,
)


def make_ssm_mixer_config(kind: str) -> TokenMixerConfig:
    if kind == "deltanet":
        return DeltaNetConfig(
            in_proj_config=LinearConfig(),
            conv_config=SeparableCausalConvConfig(has_biases=False),
            out_proj_config=LinearConfig(),
            norm_config=NORM_CONFIG,
            num_heads=2,
            num_groups=2,
            head_dim=3,
            value_head_dim=2,
            kernel_size=3,
        )
    return Mamba2Config(
        in_projection_config=LinearConfig(),
        out_projection_config=LinearConfig(),
        conv_config=SeparableCausalConvConfig(has_biases=False),
        activation=SiLU(),
        kernel_size=3,
        num_heads=2,
        num_groups=1,
        head_dim=2,
        state_dim=3,
        has_in_biases=False,
        has_out_biases=False,
    )


@pytest.fixture(params=["deltanet", "mamba2"])
def decoder(request: pytest.FixtureRequest) -> Decoder:
    mlp_config = DenseMLPConfig(
        linear_config=LinearConfig(),
        activation=Identity(),
        has_up_biases=False,
        has_down_biases=False,
        gate_clipping=None,
        up_clipping=None,
    )
    rope_config = UnscaledRoPEConfig(
        base=10_000.0,
        max_sequence_length=CONTEXT_LENGTH,
        head_dim=4,
    )
    attention_config = AttentionConfig(
        qkv_projection_config=LinearConfig(),
        out_projection_config=LinearConfig(),
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
        is_kv_sharing=False,
    )

    def layer_config(mixer_config: TokenMixerConfig, *, with_rope: bool) -> TransformerLayerConfig:
        return TransformerLayerConfig(
            pre_mixer_norm_config=NORM_CONFIG,
            mixer_config=mixer_config,
            post_mixer_norm_config=None,
            pre_mlp_norm_config=NORM_CONFIG,
            mlp_config=mlp_config,
            post_mlp_norm_config=None,
            rope_config=rope_config if with_rope else None,
            kv_source_layer_index=None,
        )

    decoder_config = DecoderConfig(
        embedding_config=TiedEmbeddingConfig(
            input_scale=None,
            logit_soft_cap=None,
        ),
        transformer_config=TransformerConfig(
            layer_configs=(
                layer_config(attention_config, with_rope=True),
                layer_config(make_ssm_mixer_config(request.param), with_rope=False),
            ),
            output_norm_config=NORM_CONFIG,
            model_dim=8,
            hidden_dim=16,
        ),
        vocab_size=32,
    )
    return decoder_config.init(
        RandomInitializer(
            default_dtype=jnp.float32,
            sharding_config=replicated_sharding_config(),
            key=jax.random.key(4),
        ),
    )


SINGLE_TOKEN_CONFIG = DecoderForwardPassConfig(
    embedding_forward_pass_config=EmbeddingForwardPassConfig(activation_dtype=jnp.float32),
    transformer_forward_pass_config=TransformerForwardPassConfig.for_inference(ForwardPassMode.SINGLE_TOKEN),
)
PREFILL_CONFIG = DecoderForwardPassConfig(
    embedding_forward_pass_config=EmbeddingForwardPassConfig(activation_dtype=jnp.float32),
)


def test_ssm_verify_matches_sequential_chain(decoder: Decoder) -> None:
    prefix_token_ids = jnp.array([[1, 2, 3]], dtype=jnp.int32)
    prefix_positions = jnp.array([[0, 1, 2]], dtype=jnp.int32)
    chain_token_ids = jnp.array([4, 5, 6], dtype=jnp.int32)

    prefix_result = decoder(
        prefix_token_ids,
        prefix_positions,
        state=None,
        return_updated_state=True,
        forward_pass_config=PREFILL_CONFIG,
        keychain=Keychain.init(10, sharding_config=make_test_sharding_config()),
    )
    assert prefix_result.updated_state is not None

    sequential_logits = []
    state = prefix_result.updated_state
    for step, token_id in enumerate(chain_token_ids):
        step_result = decoder(
            token_id[None, None],
            jnp.array([[3 + step]], dtype=jnp.int32),
            state=state,
            return_updated_state=True,
            forward_pass_config=SINGLE_TOKEN_CONFIG,
            keychain=Keychain.init(20 + step, sharding_config=make_test_sharding_config()),
        )
        assert step_result.updated_state is not None
        state = step_result.updated_state
        sequential_logits.append(step_result.logits[0, -1])

    verify_result = decoder(
        chain_token_ids[None, :],
        jnp.array([[3, 4, 5]], dtype=jnp.int32),
        state=prefix_result.updated_state,
        attention_parent_indices=jnp.array([[-1, 0, 1]], dtype=jnp.int32),
        forward_pass_config=SINGLE_TOKEN_CONFIG,
        keychain=Keychain.init(30, sharding_config=make_test_sharding_config()),
    )
    assert_close(
        result=verify_result.logits[0],
        reference=jnp.stack(sequential_logits),
        operation_name="ssm verify chain vs sequential logits",
    )


def test_ssm_verify_sibling_isolation(decoder: Decoder) -> None:
    prefix_result = decoder(
        jnp.array([[1, 2]], dtype=jnp.int32),
        jnp.array([[0, 1]], dtype=jnp.int32),
        state=None,
        return_updated_state=True,
        forward_pass_config=PREFILL_CONFIG,
        keychain=Keychain.init(40, sharding_config=make_test_sharding_config()),
    )
    assert prefix_result.updated_state is not None

    isolated_logits = []
    for seed, token_id in ((50, 10), (60, 20)):
        result = decoder(
            jnp.array([[token_id]], dtype=jnp.int32),
            jnp.array([[2]], dtype=jnp.int32),
            state=prefix_result.updated_state,
            forward_pass_config=SINGLE_TOKEN_CONFIG,
            keychain=Keychain.init(seed, sharding_config=make_test_sharding_config()),
        )
        isolated_logits.append(result.logits[0, 0])

    fork_result = decoder(
        jnp.array([[10, 20]], dtype=jnp.int32),
        jnp.array([[2, 2]], dtype=jnp.int32),
        state=prefix_result.updated_state,
        attention_parent_indices=jnp.array([[-1, -1]], dtype=jnp.int32),
        forward_pass_config=SINGLE_TOKEN_CONFIG,
        keychain=Keychain.init(70, sharding_config=make_test_sharding_config()),
    )

    assert_close(
        result=fork_result.logits[0, 0],
        reference=isolated_logits[0],
        operation_name="ssm sibling A matches isolated A",
    )
    assert_close(
        result=fork_result.logits[0, 1],
        reference=isolated_logits[1],
        operation_name="ssm sibling B matches isolated B",
    )


def test_ssm_verify_fold_matches_sequential_state(decoder: Decoder) -> None:
    num_accepted = 2
    chain_token_ids = jnp.array([4, 5, 6], dtype=jnp.int32)
    next_token_id = jnp.array([[7]], dtype=jnp.int32)

    initial_state = decoder.init_static_state(batch_size=1, capacity=CONTEXT_LENGTH, dtype=jnp.float32)
    prefix_result = decoder(
        jnp.array([[1, 2, 3]], dtype=jnp.int32),
        jnp.array([[0, 1, 2]], dtype=jnp.int32),
        state=initial_state,
        return_updated_state=True,
        forward_pass_config=PREFILL_CONFIG,
        keychain=Keychain.init(80, sharding_config=make_test_sharding_config()),
    )
    prefix_state = prefix_result.updated_state
    assert prefix_state is not None

    sequential_state = prefix_state
    for step in range(num_accepted):
        step_result = decoder(
            chain_token_ids[step][None, None],
            jnp.array([[3 + step]], dtype=jnp.int32),
            state=sequential_state,
            return_updated_state=True,
            forward_pass_config=SINGLE_TOKEN_CONFIG,
            keychain=Keychain.init(90 + step, sharding_config=make_test_sharding_config()),
        )
        assert step_result.updated_state is not None
        sequential_state = step_result.updated_state
    sequential_result = decoder(
        next_token_id,
        jnp.array([[3 + num_accepted]], dtype=jnp.int32),
        state=sequential_state,
        return_updated_state=True,
        forward_pass_config=SINGLE_TOKEN_CONFIG,
        keychain=Keychain.init(100, sharding_config=make_test_sharding_config()),
    )

    verify_result = decoder(
        chain_token_ids[None, :],
        jnp.array([[3, 4, 5]], dtype=jnp.int32),
        state=prefix_state.begin_verification(chain_token_ids.size),
        return_updated_state=True,
        attention_parent_indices=jnp.array([[-1, 0, 1]], dtype=jnp.int32),
        forward_pass_config=SINGLE_TOKEN_CONFIG,
        keychain=Keychain.init(110, sharding_config=make_test_sharding_config()),
    )
    assert verify_result.updated_state is not None

    accepted_node_indices = jnp.array([[0, 1, -1]], dtype=jnp.int32)
    num_accepted_nodes = jnp.array([num_accepted], dtype=jnp.int32)
    with jax.set_mesh(replicated_sharding_config().mesh):
        committed_state = verify_result.updated_state.commit_accepted(
            accepted_node_indices,
            num_accepted_nodes,
        )

    folded_result = decoder(
        next_token_id,
        jnp.array([[3 + num_accepted]], dtype=jnp.int32),
        state=committed_state,
        return_updated_state=True,
        forward_pass_config=SINGLE_TOKEN_CONFIG,
        keychain=Keychain.init(120, sharding_config=make_test_sharding_config()),
    )

    assert_close(
        result=folded_result.logits[0, -1],
        reference=sequential_result.logits[0, -1],
        operation_name="post-fold next-token logits vs sequential path",
    )
