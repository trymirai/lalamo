import jax

# import transformers # No longer needed for test_rms_norm signature
from jaxtyping import PRNGKeyArray

# Decoder and ETTransformer are still needed for test_out_norm_executorch
from fartsovka.modules import Decoder
from tests.executorch_llama.transformer import Transformer as ETTransformer

from .common import QUANTIZED_RTOL, assert_close, checkify_forward, from_torch, to_torch

# Import LayerPair for type hinting the new fixture
from .conftest import LayerPair


def test_rms_norm(
    layers: LayerPair,  # Changed from request, current_hf_model, current_fartsovka_model
    rng_key: PRNGKeyArray,
) -> None:
    # layers.reference_layer is the HF/reference model's full DecoderLayer object
    # layers.fartsovka_layer is the Fartsovka DecoderLayer object

    # Access the specific norm components from the respective layer objects
    # Assuming the reference_layer (e.g., LlamaDecoderLayer) has an 'input_layernorm' attribute
    hf_norm_layer = layers.reference_layer.input_layernorm  # type: ignore
    # Assuming the fartsovka_layer (DecoderLayer) has a 'pre_attention_norm' attribute
    fs_norm_layer = layers.fartsovka_layer.pre_attention_norm

    fs_norm_layer_forward = checkify_forward(fs_norm_layer)

    # Assuming the norm layer components (e.g., fs_norm_layer) have an 'input_dim' attribute
    # or a way to access the dimension they operate on.
    # If 'input_dim' is not directly on the norm layer, this might need adjustment,
    # e.g., layers.fartsovka_layer.attention.model_dim or similar if norm is tied to attention dim.
    # For simplicity, assuming direct 'input_dim' attribute as per previous structure.
    input_dim = fs_norm_layer.input_dim

    sample_input = jax.random.normal(rng_key, (input_dim,))
    sample_input_torch = to_torch(sample_input).unsqueeze(0)
    hf_output = from_torch(hf_norm_layer(sample_input_torch).squeeze(0))
    err, fs_output = fs_norm_layer_forward(sample_input)
    err.throw()
    assert_close(
        result=fs_output,
        reference=hf_output,
        operation_name="rms_norm",
    )


# test_gemma2_rms_norm is removed as its functionality is covered by
# the refactored test_rms_norm when Gemma is part of MODEL_PAIRS_TO_TEST.


def test_out_norm_executorch(  # This test remains unchanged
    executorch_llama: ETTransformer,
    fartsovka_qlora_llama: Decoder,
    rng_key: PRNGKeyArray,
) -> None:
    hf_layer = executorch_llama.norm
    fs_layer = fartsovka_qlora_llama.output_norm
    fs_layer_forward = checkify_forward(fs_layer)

    input_dim = fs_layer.input_dim

    sample_input = jax.random.normal(rng_key, (input_dim,))
    sample_input_torch = to_torch(sample_input).unsqueeze(0)
    hf_output = from_torch(hf_layer(sample_input_torch).squeeze(0))
    err, fs_output = fs_layer_forward(sample_input)
    err.throw()
    assert_close(
        result=fs_output,
        reference=hf_output,
        rtol=QUANTIZED_RTOL,
        operation_name="out_norm",
    )
