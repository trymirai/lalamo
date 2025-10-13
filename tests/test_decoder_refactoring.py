"""
Tests to verify that NewDecoder (using Transformer) produces identical outputs to original Decoder.
"""
import jax.numpy as jnp
from jax import random

from lalamo.modules.activations import SiLU
from lalamo.modules.attention import AttentionConfig
from lalamo.modules.decoder import Decoder, DecoderConfig
from lalamo.modules.decoder_layer import DecoderLayerConfig
from lalamo.modules.embedding import TiedEmbeddingConfig
from lalamo.modules.linear import FullPrecisionLinearConfig
from lalamo.modules.mlp import DenseMLPConfig
from lalamo.modules.new_decoder import NewDecoder, NewDecoderConfig
from lalamo.modules.normalization import RMSNormConfig, UpcastMode
from lalamo.modules.rope import UnscaledRoPEConfig
from lalamo.modules.transformer import TransformerConfig


def create_test_decoder_config() -> DecoderConfig:
    """Create a small decoder config for testing."""
    vocab_size = 1000
    model_dim = 256
    hidden_dim = 512
    num_heads = 8
    num_groups = 4
    head_dim = 32
    num_layers = 4
    context_length = 128

    embedding_config = TiedEmbeddingConfig(
        precision=jnp.float32,
        input_scale=None,
        logit_soft_cap=None,
    )

    rope_config = UnscaledRoPEConfig(
        precision=jnp.float32,
        base=10000.0,
        max_sequence_length=context_length,
    )

    norm_config = RMSNormConfig(
        scale_precision=jnp.float32,
        accumulation_precision=jnp.float32,
        epsilon=1e-5,
        scale_offset=None,
        upcast_mode=UpcastMode.ONLY_NORMALIZATION,
    )

    linear_config = FullPrecisionLinearConfig(
        precision=jnp.float32,
    )

    attention_config = AttentionConfig(
        qkv_projection_config=linear_config,
        out_projection_config=linear_config,
        query_norm_config=None,
        key_norm_config=None,
        logit_soft_cap=None,
        has_sinks=False,
        has_qkv_biases=False,
        has_out_biases=False,
    )

    mlp_config = DenseMLPConfig(
        linear_config=linear_config,
        activation=SiLU(),
        has_up_biases=False,
        has_down_biases=False,
        gate_clipping=None,
        up_clipping=None,
    )

    layer_config = DecoderLayerConfig(
        pre_attention_norm_config=norm_config,
        attention_config=attention_config,
        post_attention_norm_config=None,
        pre_mlp_norm_config=norm_config,
        mlp_config=mlp_config,
        post_mlp_norm_config=None,
    )

    return DecoderConfig(
        embedding_config=embedding_config,
        global_rope_config=rope_config,
        local_rope_config=None,
        layer_config=layer_config,
        output_norm_config=norm_config,
        vocab_size=vocab_size,
        model_dim=model_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_groups=num_groups,
        head_dim=head_dim,
        attention_scale=None,
        num_layers=num_layers,
        sliding_window_sizes=None,
        context_length=context_length,
    )


def create_test_new_decoder_config() -> NewDecoderConfig:
    """Create equivalent NewDecoder config from Decoder config."""
    decoder_config = create_test_decoder_config()

    transformer_config = TransformerConfig(
        global_rope_config=decoder_config.global_rope_config,
        local_rope_config=decoder_config.local_rope_config,
        layer_config=decoder_config.layer_config,
        output_norm_config=decoder_config.output_norm_config,
        model_dim=decoder_config.model_dim,
        hidden_dim=decoder_config.hidden_dim,
        num_heads=decoder_config.num_heads,
        num_groups=decoder_config.num_groups,
        head_dim=decoder_config.head_dim,
        attention_scale=decoder_config.attention_scale,
        num_layers=decoder_config.num_layers,
        sliding_window_sizes=decoder_config.sliding_window_sizes,
        context_length=decoder_config.context_length,
    )

    return NewDecoderConfig(
        embedding_config=decoder_config.embedding_config,
        transformer_config=transformer_config,
        vocab_size=decoder_config.vocab_size,
    )


def test_decoder_shapes_match():
    """Test that both decoders produce outputs of the same shape."""
    rng_key = random.PRNGKey(42)
    decoder_key, new_decoder_key, input_key = random.split(rng_key, 3)

    # Create configs
    decoder_config = create_test_decoder_config()
    new_decoder_config = create_test_new_decoder_config()

    # Initialize models
    decoder = decoder_config.random_init(key=decoder_key)
    new_decoder = new_decoder_config.random_init(key=new_decoder_key)

    # Create test input
    batch_size = 2
    seq_length = 16
    token_ids = random.randint(input_key, (batch_size, seq_length), 0, decoder_config.vocab_size)
    token_positions = jnp.arange(seq_length)[None, :].repeat(batch_size, axis=0)

    # Run inference
    decoder_result = decoder(token_ids, token_positions)
    new_decoder_result = new_decoder(token_ids, token_positions)

    # Check shapes match
    assert decoder_result.logits.shape == new_decoder_result.logits.shape
    assert decoder_result.logits.shape == (batch_size, seq_length, decoder_config.vocab_size)


def test_decoder_outputs_match_with_same_weights():
    """Test that outputs match exactly when using the same weights."""
    rng_key = random.PRNGKey(123)
    init_key, input_key = random.split(rng_key)

    # Create configs
    decoder_config = create_test_decoder_config()
    new_decoder_config = create_test_new_decoder_config()

    # Initialize original decoder
    decoder = decoder_config.random_init(key=init_key)

    # Export weights from original decoder
    decoder_weights = decoder.export_weights()

    # Restructure weights for NewDecoder
    # NewDecoder expects: {"embedding": {...}, "transformer": {"global_rope": ..., "layers": ..., "output_norm": ..., "local_rope": ...}}
    new_decoder_weights = {
        "embedding": decoder_weights["embedding"],
        "transformer": {
            "global_rope": decoder_weights["global_rope"],
            "layers": decoder_weights["layers"],
            "output_norm": decoder_weights["output_norm"],
        }
    }
    if "local_rope" in decoder_weights:
        new_decoder_weights["transformer"]["local_rope"] = decoder_weights["local_rope"]

    # Create new decoder and import weights
    new_decoder = new_decoder_config.empty()
    new_decoder = new_decoder.import_weights(new_decoder_weights)

    # Create test input
    batch_size = 2
    seq_length = 16
    token_ids = random.randint(input_key, (batch_size, seq_length), 0, decoder_config.vocab_size)
    token_positions = jnp.arange(seq_length)[None, :].repeat(batch_size, axis=0)

    # Run inference
    decoder_result = decoder(token_ids, token_positions)
    new_decoder_result = new_decoder(token_ids, token_positions)

    # Outputs should match exactly
    assert jnp.allclose(decoder_result.logits, new_decoder_result.logits, rtol=1e-5, atol=1e-5)


def test_decoder_with_kv_cache():
    """Test that KV cache behavior matches between both decoders."""
    rng_key = random.PRNGKey(456)
    init_key, input_key = random.split(rng_key)

    # Create configs
    decoder_config = create_test_decoder_config()
    new_decoder_config = create_test_new_decoder_config()

    # Initialize original decoder
    decoder = decoder_config.random_init(key=init_key)

    # Export and restructure weights
    decoder_weights = decoder.export_weights()
    new_decoder_weights = {
        "embedding": decoder_weights["embedding"],
        "transformer": {
            "global_rope": decoder_weights["global_rope"],
            "layers": decoder_weights["layers"],
            "output_norm": decoder_weights["output_norm"],
        }
    }

    # Create new decoder with same weights
    new_decoder = new_decoder_config.empty()
    new_decoder = new_decoder.import_weights(new_decoder_weights)

    # Create test input
    batch_size = 1
    seq_length = 8
    token_ids = random.randint(input_key, (batch_size, seq_length), 0, decoder_config.vocab_size)
    token_positions = jnp.arange(seq_length)[None, :].repeat(batch_size, axis=0)

    # Initialize KV cache
    cache_capacity = 32
    kv_cache_decoder = decoder.init_static_kv_cache(batch_size, cache_capacity)
    kv_cache_new_decoder = new_decoder.init_static_kv_cache(batch_size, cache_capacity)

    # Run with KV cache
    decoder_result = decoder(
        token_ids,
        token_positions,
        kv_cache=kv_cache_decoder,
        return_updated_kv_cache=True,
    )
    new_decoder_result = new_decoder(
        token_ids,
        token_positions,
        kv_cache=kv_cache_new_decoder,
        return_updated_kv_cache=True,
    )

    # Check logits match
    assert jnp.allclose(decoder_result.logits, new_decoder_result.logits, rtol=1e-5, atol=1e-5)

    # Check updated KV caches exist
    assert decoder_result.updated_kv_cache is not None
    assert new_decoder_result.updated_kv_cache is not None


def test_decoder_activation_trace():
    """Test that activation trace works for both decoders."""
    rng_key = random.PRNGKey(789)
    init_key, input_key = random.split(rng_key)

    # Create configs
    decoder_config = create_test_decoder_config()
    new_decoder_config = create_test_new_decoder_config()

    # Initialize original decoder
    decoder = decoder_config.random_init(key=init_key)

    # Export and restructure weights
    decoder_weights = decoder.export_weights()
    new_decoder_weights = {
        "embedding": decoder_weights["embedding"],
        "transformer": {
            "global_rope": decoder_weights["global_rope"],
            "layers": decoder_weights["layers"],
            "output_norm": decoder_weights["output_norm"],
        }
    }

    # Create new decoder with same weights
    new_decoder = new_decoder_config.empty()
    new_decoder = new_decoder.import_weights(new_decoder_weights)

    # Create test input
    batch_size = 1
    seq_length = 4
    token_ids = random.randint(input_key, (batch_size, seq_length), 0, decoder_config.vocab_size)
    token_positions = jnp.arange(seq_length)[None, :].repeat(batch_size, axis=0)

    # Run with activation trace
    decoder_result = decoder(
        token_ids,
        token_positions,
        return_activation_trace=True,
    )
    new_decoder_result = new_decoder(
        token_ids,
        token_positions,
        return_activation_trace=True,
    )

    # Check activation traces exist
    assert decoder_result.activation_trace is not None
    assert new_decoder_result.activation_trace is not None

    # Check layer results match in count
    assert len(decoder_result.activation_trace.layer_results) == len(new_decoder_result.activation_trace.layer_results)

    # Check output norms match
    assert jnp.allclose(
        decoder_result.activation_trace.output_norm,
        new_decoder_result.activation_trace.output_norm,
        rtol=1e-5,
        atol=1e-5,
    )


def test_decoder_weight_export_import_roundtrip():
    """Test that weight export/import works correctly for NewDecoder."""
    rng_key = random.PRNGKey(999)

    # Create config
    new_decoder_config = create_test_new_decoder_config()

    # Initialize model
    new_decoder = new_decoder_config.random_init(key=rng_key)

    # Export weights
    weights = new_decoder.export_weights()

    # Create new instance and import
    new_decoder_2 = new_decoder_config.empty()
    new_decoder_2 = new_decoder_2.import_weights(weights)

    # Create test input
    token_ids = jnp.array([[1, 2, 3, 4]])
    token_positions = jnp.array([[0, 1, 2, 3]])

    # Run inference on both
    result_1 = new_decoder(token_ids, token_positions)
    result_2 = new_decoder_2(token_ids, token_positions)

    # Outputs should match exactly
    assert jnp.allclose(result_1.logits, result_2.logits, rtol=1e-7, atol=1e-7)
