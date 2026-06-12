import jax
import jax.numpy as jnp
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from lalamo.initializer import RandomInitializer
from lalamo.models.chat_codec import ChatCodecConfig
from lalamo.models.language_model import GenerationConfig, LanguageModel, LanguageModelConfig
from lalamo.models.raw_text_codec import RawTextCodecConfig
from lalamo.modules import (
    DecoderConfig,
    DenseMLPConfig,
    DFlashAttentionConfig,
    DFlashDraftConfig,
    DFlashDraftLayerConfig,
    Identity,
    LinearConfig,
    NormalizationConfig,
    TiedEmbeddingConfig,
    TransformerConfig,
    TransformerLayerConfig,
    UnscaledRoPEConfig,
    UpcastMode,
)
from lalamo.modules.activations import SiLU
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.speculator import DFlashSpeculator, DFlashSpeculatorConfig
from lalamo.utils.sharding import ShardingConfig

MODEL_DIM = 8
HIDDEN_DIM = 16
VOCAB_SIZE = 32
CONTEXT_LENGTH = 128
NUM_LAYERS = 2
HEAD_DIM = 4
BLOCK_SIZE = 4

NORM_CONFIG = NormalizationConfig(
    epsilon=1e-5,
    scale_offset=None,
    upcast_mode=UpcastMode.ONLY_NORMALIZATION,
    subtract_mean=False,
)
ROPE_CONFIG = UnscaledRoPEConfig(
    base=10_000.0,
    max_sequence_length=CONTEXT_LENGTH,
    head_dim=HEAD_DIM,
)


def make_language_model_config() -> LanguageModelConfig:
    mlp_config = DenseMLPConfig(
        linear_config=LinearConfig(),
        activation=Identity(),
        has_up_biases=False,
        has_down_biases=False,
        gate_clipping=None,
        up_clipping=None,
    )
    attention_config = AttentionConfig(
        qkv_projection_config=LinearConfig(),
        out_projection_config=LinearConfig(),
        query_norm_config=None,
        key_norm_config=None,
        num_heads=2,
        num_groups=2,
        head_dim=HEAD_DIM,
        is_causal=True,
        scale=None,
        sliding_window_size=None,
        logit_soft_cap=None,
        has_sinks=False,
        has_qkv_biases=False,
        has_out_biases=False,
        is_kv_sharing=False,
    )
    layer_config = TransformerLayerConfig(
        pre_mixer_norm_config=NORM_CONFIG,
        mixer_config=attention_config,
        post_mixer_norm_config=None,
        pre_mlp_norm_config=NORM_CONFIG,
        mlp_config=mlp_config,
        post_mlp_norm_config=None,
        rope_config=ROPE_CONFIG,
        kv_source_layer_index=None,
    )
    decoder_config = DecoderConfig(
        embedding_config=TiedEmbeddingConfig(
            input_scale=None,
            logit_soft_cap=None,
        ),
        transformer_config=TransformerConfig(
            layer_configs=(layer_config,) * NUM_LAYERS,
            output_norm_config=NORM_CONFIG,
            model_dim=MODEL_DIM,
            hidden_dim=HIDDEN_DIM,
        ),
        vocab_size=VOCAB_SIZE,
    )
    codec_config = ChatCodecConfig(
        prompt_template="{% for message in messages %}{{ message.content }}{% endfor %}",
        output_parser_regex=None,
        system_role_name="system",
        user_role_name="user",
        assistant_role_name="assistant",
        eos_token=None,
        bos_token=None,
        end_of_thinking_tag=None,
    )
    return LanguageModelConfig(
        token_codec_config=codec_config,
        decoder_config=decoder_config,
        generation_config=GenerationConfig(),
    )


def make_draft_config() -> DFlashDraftConfig:
    draft_mlp_config = DenseMLPConfig(
        linear_config=LinearConfig(),
        activation=SiLU(),
        has_up_biases=False,
        has_down_biases=False,
        gate_clipping=None,
        up_clipping=None,
    )
    return DFlashDraftConfig(
        model_dim=MODEL_DIM,
        hidden_dim=HIDDEN_DIM,
        block_size=BLOCK_SIZE,
        mask_token_id=VOCAB_SIZE - 1,
        target_layer_ids=(0, 1),
        num_target_layers=NUM_LAYERS,
        vocab_size=VOCAB_SIZE,
        context_projection_config=LinearConfig(),
        context_norm_config=NORM_CONFIG,
        layer_configs=(
            DFlashDraftLayerConfig(
                attention_config=DFlashAttentionConfig(
                    linear_config=LinearConfig(),
                    query_norm_config=NORM_CONFIG,
                    key_norm_config=NORM_CONFIG,
                    rope_config=ROPE_CONFIG,
                    num_heads=2,
                    num_key_value_heads=2,
                    head_dim=HEAD_DIM,
                    has_attention_biases=False,
                    has_output_biases=False,
                    sliding_window_size=None,
                    scale=HEAD_DIM**-0.5,
                ),
                input_norm_config=NORM_CONFIG,
                post_attention_norm_config=NORM_CONFIG,
                mlp_config=draft_mlp_config,
            ),
        ),
        output_norm_config=NORM_CONFIG,
    )


@pytest.fixture(scope="package")
def sharding_config() -> ShardingConfig:
    return ShardingConfig.replicated()


@pytest.fixture(scope="package")
def language_model(sharding_config: ShardingConfig) -> LanguageModel:
    tokenizer = Tokenizer(WordLevel({f"tok{i}": i for i in range(VOCAB_SIZE)}, unk_token="tok0"))
    return make_language_model_config().init(
        tokenizer,
        RandomInitializer(
            default_dtype=jnp.float32,
            sharding_config=sharding_config,
            key=jax.random.key(0),
        ),
    )


@pytest.fixture(scope="package")
def dflash_speculator(sharding_config: ShardingConfig) -> DFlashSpeculator:
    draft_config = make_draft_config()
    speculator_config = DFlashSpeculatorConfig(
        token_codec_config=RawTextCodecConfig(),
        draft_config=draft_config,
    )
    tokenizer = Tokenizer(WordLevel({f"tok{i}": i for i in range(VOCAB_SIZE)}, unk_token="tok0"))
    return DFlashSpeculator(
        config=speculator_config,
        sharding_config=sharding_config,
        token_codec=speculator_config.token_codec_config.init(tokenizer),
        draft_model=draft_config.init(
            RandomInitializer(
                default_dtype=jnp.float32,
                sharding_config=sharding_config,
                key=jax.random.key(1),
            ),
        ),
    )
