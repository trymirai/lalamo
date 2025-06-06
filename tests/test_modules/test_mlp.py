import jax

# import transformers # No longer needed as fixture in test_mlp
from jaxtyping import PRNGKeyArray

# from fartsovka.modules import Decoder # No longer needed as fixture
# from tests.executorch_llama.transformer import Transformer as ETTransformer # No longer needed as fixture
from .common import (
    QUANTIZED_ATOL,
    QUANTIZED_RTOL,
    assert_close,
    checkify_forward,
    from_torch,
    to_torch,
)

# Import LayerPair for type hinting the new fixture
from .conftest import LayerPair


def test_mlp(
    layers: LayerPair,
    rng_key: PRNGKeyArray,
) -> None:
    # layers.reference_layer is the HF/reference model's full DecoderLayer
    # layers.fartsovka_layer is the Fartsovka DecoderLayer
    hf_mlp_layer = layers.reference_layer.mlp  # type: ignore
    fs_mlp_layer = layers.fartsovka_layer.mlp
    fs_mlp_layer_forward = checkify_forward(fs_mlp_layer)

    # Assuming up_projection is a direct attribute of the MLP component
    input_dim = fs_mlp_layer.up_projection.input_dim

    sample_input = jax.random.normal(rng_key, (input_dim,))
    sample_input_torch = to_torch(sample_input).unsqueeze(0)
    hf_output = from_torch(hf_mlp_layer(sample_input_torch).squeeze(0))
    err, fs_output = fs_mlp_layer_forward(sample_input)
    err.throw()
    assert_close(
        result=fs_output,
        reference=hf_output,
    )


# test_gemma2_mlp is removed as its functionality should be covered by
# the refactored test_mlp when Gemma is part of MODEL_PAIRS_TO_TEST.


def test_qlora_mlp(
    qlora_layers: LayerPair,
    rng_key: PRNGKeyArray,
) -> None:
    # qlora_layers.fartsovka_layer is a DecoderLayer from fartsovka_qlora_llama
    # qlora_layers.reference_layer is an ETDecoderLayer from executorch_llama
    fs_mlp_layer = qlora_layers.fartsovka_layer.mlp
    fs_mlp_layer_forward = checkify_forward(fs_mlp_layer)
    # ETDecoderLayer (TransformerBlock) has 'feed_forward' not 'mlp'
    et_feed_forward_layer = qlora_layers.reference_layer.feed_forward  # type: ignore

    input_dim = fs_mlp_layer.up_projection.input_dim

    sample_input = jax.random.normal(rng_key, (input_dim,))
    sample_input_torch = to_torch(sample_input).unsqueeze(0)

    et_output = from_torch(et_feed_forward_layer(sample_input_torch).squeeze(0))
    err, fs_output = fs_mlp_layer_forward(sample_input)
    err.throw()
    assert_close(
        result=fs_output,
        reference=et_output,
        atol=QUANTIZED_ATOL,
        rtol=QUANTIZED_RTOL,
    )
