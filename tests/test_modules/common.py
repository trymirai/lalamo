import numpy as np
import torch
import jax
from jax import numpy as jnp
from jax.experimental.checkify import checkify, div_checks, index_checks, nan_checks, user_checks
from jaxtyping import PRNGKeyArray

from fartsovka.modules import (
    Activation,
    AttentionConfig,
    Decoder,
    DecoderConfig,
    DecoderLayerConfig,
    FullPrecisionLinearConfig,
    MLPConfig,
    RMSNormConfig,
    TiedEmbeddingConfig as EmbeddingConfig,
    UnscaledRoPEConfig,
)

__all__ = [
    "assert_close", 
    "from_torch", 
    "to_torch", 
    "create_dummy_decoder",
]

ATOL = 1e-3
RTOL = 0.01
QUANTIZED_ATOL = 0.03
QUANTIZED_RTOL = 0.1


LAYERS_TO_TEST = list(range(16))


def assert_close(
    *,
    result: jnp.ndarray,
    reference: jnp.ndarray,
    atol: float = ATOL,
    rtol: float = RTOL,
    operation_name: str | None = None,
) -> None:
    absdiff = jnp.abs(result - reference)

    allowed_diff = atol + rtol * jnp.abs(reference)
    err = jnp.maximum(absdiff - allowed_diff, 0)
    err_rel = err / (jnp.abs(reference) + 1e-10)
    max_err = jnp.max(err)
    max_err_idx = tuple(i.item() for i in jnp.unravel_index(jnp.argmax(err), err.shape))
    max_err_rel = err_rel[max_err_idx]
    max_err_reference_value = reference[max_err_idx]

    num_violations = jnp.sum(err > 0)

    rms_diff = jnp.sqrt(jnp.mean(jnp.square(absdiff)))
    rms_result = jnp.sqrt(jnp.mean(jnp.square(result)))
    rms_reference = jnp.sqrt(jnp.mean(jnp.square(reference)))
    rel_rms_reference = rms_diff / (rms_reference + 1e-10)

    if operation_name is not None:
        operation_description = f" during {operation_name}"
    else:
        operation_description = ""

    message = (
        f"{num_violations} violations > {atol:.1e} + {rtol:.2%}{operation_description}."
        f" Max error: {max_err:.3g} ({max_err_rel:.2%}) at index {max_err_idx}"
        f" (reference value: {max_err_reference_value:.3g})."
        f" Error RMS: {rms_diff:.3g}."
        f" RMS of result: {rms_result:.3g}, RMS of reference: {rms_reference:.3g}."
        f" Relative error RMS: {rel_rms_reference:.2%} of RMS of reference."
        f" Shape: {result.shape}"
    )
    assert jnp.allclose(result, reference, atol=atol, rtol=rtol), message


@torch.no_grad()
def from_torch(tensor: torch.Tensor) -> jnp.ndarray:
    return jnp.array(tensor.cpu().numpy())


@torch.no_grad()
def to_torch(array: jnp.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.array(array))


def checkify_forward(module):  # noqa: ANN001, ANN202
    return checkify(
        module.__call__,
        errors=index_checks | nan_checks | div_checks | user_checks,
    )


def create_dummy_decoder(rng_key: PRNGKeyArray, vocab_size: int = 32000, num_layers: int = 2) -> Decoder:
    """Create a small dummy decoder for testing."""
    # Define a minimal configuration
    model_dim = 128
    hidden_dim = 256
    num_heads = 4
    num_groups = 2
    head_dim = 32
    context_length = 128
    
    # Create embedding config
    embedding_config = EmbeddingConfig(
        input_scale=None,
        logits_soft_cap=None,
        precision=jnp.float32,
    )
    
    # Create RoPE config
    rope_config = UnscaledRoPEConfig(
        precision=jnp.float32,
        base=10000.0,
        max_sequence_length=context_length,
    )
    
    # Create norm config
    norm_config = RMSNormConfig(
        scale_precision=jnp.float32,
        accumulation_precision=jnp.float32,
        epsilon=1e-6,
    )
    
    # Create linear config
    linear_config = FullPrecisionLinearConfig(
        precision=jnp.float32,
    )
    
    # Create attention and MLP configs
    attention_config = AttentionConfig(
        qkv_projection_config=linear_config,
        out_projection_config=linear_config,
        logit_soft_cap=None,
        has_qkv_biases=False,
        has_out_biases=False,
    )
    
    mlp_config = MLPConfig(
        linear_config=linear_config,
        activation=Activation.SILU,
    )
    
    # Create decoder layer config
    layer_config = DecoderLayerConfig(
        pre_attention_norm_config=norm_config,
        attention_config=attention_config,
        post_attention_norm_config=None,
        pre_mlp_norm_config=norm_config,
        mlp_config=mlp_config,
        post_mlp_norm_config=None,
    )
    
    # Create decoder config
    decoder_config = DecoderConfig(
        embedding_config=embedding_config,
        rope_config=rope_config,
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
    
    # Initialize the decoder
    return decoder_config.random_init(key=rng_key)
