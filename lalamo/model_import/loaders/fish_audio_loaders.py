from collections.abc import Mapping

from jax import numpy as jnp
from jaxtyping import Array

from lalamo.common import ParameterPath
from lalamo.modules import (
    Attention,
    DenseMLP,
    FullPrecisionLinear,
    LayerNorm,
    LinearBase,
    MLPBase,
    Normalization,
    Transformer,
    TransformerLayer,
)

from .common import load_parameters


def _fuse_full_precision_weights(
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    sublayers_to_fuse: list[str] | None,
) -> Array:
    if sublayers_to_fuse is None:
        return weights_dict[path / "weight"]

    weights = [weights_dict[path / layer_name / "weight"] for layer_name in sublayers_to_fuse]
    return jnp.concatenate(weights, axis=0)


def load_linear(
    module: LinearBase,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    sublayers_to_fuse: list[str] | None = None,
) -> LinearBase:
    """Loads a linear layer, optionally fusing weights from sublayers."""
    if not module.has_biases:
        if sublayers_to_fuse:
            paths_to_check = [path / proj / "bias" for proj in sublayers_to_fuse]
        else:
            paths_to_check = path / "bias"
        for p in paths_to_check:
            if p in weights_dict:
                raise ValueError(f"Bias tensor found at {p} but module does not support it.")
        bias = None
    elif sublayers_to_fuse is None:
        bias = weights_dict[path / "bias"]
    else:
        bias = jnp.concatenate(
            [weights_dict[path / proj_name / "bias"] for proj_name in sublayers_to_fuse],
            axis=0,
        )

    if isinstance(module, FullPrecisionLinear):
        weights = _fuse_full_precision_weights(weights_dict, path, sublayers_to_fuse)
        return load_parameters(lambda m: (m.weights, m.biases), module, (weights, bias))

    raise TypeError(f"Unsupported module type for loading: {type(module)}")


def load_rmsnorm(
    module: Normalization,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> Normalization:
    scales = weights_dict[path / "weight"]
    return load_parameters(lambda m: (m.scales,), module, (scales,))


def load_layer_norm(
    module: LayerNorm,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> LayerNorm:
    scales = weights_dict[path / "gamma"]
    return load_parameters(lambda m: (m.scales,), module, (scales,))


def load_transformer_block(module: Transformer, weights_dict: Mapping[str, Array], fast: bool = False) -> Transformer:
    def load_attention_local(
        module: Attention,
        weights_dict: Mapping[str, Array],
        path: ParameterPath,
    ) -> Attention:
        qkv_projection = load_linear(
            module.qkv_projection,
            weights_dict,
            path / "wqkv",
            sublayers_to_fuse=None,
        )
        out_projection = load_linear(module.out_projection, weights_dict, path / "wo")

        if module.query_norm is not None:
            query_norm = load_rmsnorm(module.query_norm, weights_dict, path / "q_norm")
        else:
            query_norm = None

        if module.key_norm is not None:
            key_norm = load_rmsnorm(module.key_norm, weights_dict, path / "k_norm")
        else:
            key_norm = None

        return load_parameters(
            lambda m: (m.qkv_projection, m.out_projection, m.query_norm, m.key_norm),
            module,
            (qkv_projection, out_projection, query_norm, key_norm),
        )

    def load_mlp(
        module: MLPBase,
        weights_dict: Mapping[str, Array],
        path: ParameterPath,
        up_proj_key: str,
        gate_proj_key: str,
        down_proj_key: str,
    ) -> MLPBase:
        assert isinstance(module, DenseMLP)
        # Standard dense MLP with separate sublayers.
        up_projection = load_linear(
            module.up_projection,
            weights_dict,
            path,
            sublayers_to_fuse=[up_proj_key, gate_proj_key],
        )
        down_projection = load_linear(module.down_projection, weights_dict, path / down_proj_key)
        return load_parameters(
            lambda m: (m.up_projection, m.down_projection),
            module,
            (up_projection, down_projection),
        )

    def load_transformer_layer_local(
        module: TransformerLayer,
        weights_dict: Mapping[str, Array],
        path: ParameterPath,
    ) -> TransformerLayer:
        if module.pre_mixer_norm is not None:
            pre_mixer_norm = load_rmsnorm(
                module.pre_mixer_norm,
                weights_dict,
                path / "attention_norm",
            )
        else:
            pre_mixer_norm = None

        if module.post_mixer_norm is not None:
            post_mixer_norm = load_layer_norm(module.post_mixer_norm, weights_dict, path / "attention_layer_scale")
        else:
            post_mixer_norm = None

        assert isinstance(module.mixer, Attention)
        attention = load_attention_local(module.mixer, weights_dict, path / "attention")

        pre_mlp_norm = load_rmsnorm(
            module.pre_mlp_norm,
            weights_dict,
            path / "ffn_norm",
        )
        if module.post_mlp_norm is not None:
            post_mlp_norm = load_layer_norm(module.post_mlp_norm, weights_dict, path / "ffn_layer_scale")
        else:
            post_mlp_norm = None

        mlp = load_mlp(module.mlp, weights_dict, path / "feed_forward", "w3", "w1", "w2")

        return load_parameters(
            lambda m: (
                m.pre_mixer_norm,
                m.mixer,
                m.post_mixer_norm,
                m.pre_mlp_norm,
                m.mlp,
                m.post_mlp_norm,
            ),
            module,
            (
                pre_mixer_norm,
                attention,
                post_mixer_norm,
                pre_mlp_norm,
                mlp,
                post_mlp_norm,
            ),
        )

    base_path = ParameterPath()

    layers_name = "layers" if not fast else "fast_layers"
    norm_name = "norm" if not fast else "fast_norm"

    transformer_layers = tuple(
        load_transformer_layer_local(layer, weights_dict, base_path / layers_name / i)
        for i, layer in enumerate(module.layers)
    )
    output_norm = load_rmsnorm(module.output_norm, weights_dict, base_path / norm_name)

    module = load_parameters(
        lambda m: (
            m.layers,
            m.output_norm,
        ),
        module,
        (
            transformer_layers,
            output_norm,
        ),
    )

    return module


def load_fish_audio_text_decoding_modules(
    transformer: Transformer, output: FullPrecisionLinear, weights_dict: Mapping[str, Array], fast: bool = False
) -> tuple[Transformer, FullPrecisionLinear]:
    transformer = load_transformer_block(transformer, weights_dict=weights_dict, fast=fast)

    base_path = ParameterPath()
    output_linear_name = "output" if not fast else "fast_output"
    output_linear = load_linear(output, weights_dict, base_path / output_linear_name)
    output = load_parameters(
        lambda m: (m,),
        output,
        (output_linear,),
    )

    return (transformer, output)
