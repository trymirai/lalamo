import json
from dataclasses import dataclass, replace
from pathlib import Path

import jax.numpy as jnp
from cattrs import Converter
from jaxtyping import Array

from lalamo.modules.decoder import Decoder
from lalamo.modules.linear import FullPrecisionLinear
from lalamo.modules.mlp import DenseMLP

__all__ = [
    "MlpZeroChannelLayer",
    "MlpZeroChannelSpec",
    "load_mlp_zero_channel_spec",
    "zero_decoder_mlp_channels",
    "zero_mlp_channels",
]


_CONVERTER = Converter()


@dataclass(frozen=True)
class MlpZeroChannelLayer:
    layer_index: int
    channels: tuple[int, ...]


@dataclass(frozen=True)
class MlpZeroChannelSpec:
    layers: tuple[MlpZeroChannelLayer, ...]


def load_mlp_zero_channel_spec(path: Path | str) -> MlpZeroChannelSpec:
    with Path(path).open() as spec_file:
        return _validate_spec(_CONVERTER.structure(json.load(spec_file), MlpZeroChannelSpec))


def zero_decoder_mlp_channels(decoder: Decoder, spec: MlpZeroChannelSpec) -> Decoder:
    layers = list(decoder.transformer.layers)
    for layer_spec in spec.layers:
        if layer_spec.layer_index >= len(layers):
            raise ValueError(f"Layer index {layer_spec.layer_index} exceeds decoder depth {len(layers)}")
        mlp = layers[layer_spec.layer_index].mlp
        if not isinstance(mlp, DenseMLP):
            raise TypeError(f"Layer {layer_spec.layer_index} MLP is {type(mlp).__name__}, expected DenseMLP")
        layers[layer_spec.layer_index] = replace(
            layers[layer_spec.layer_index],
            mlp=zero_mlp_channels(mlp, layer_spec.channels),
        )
    return replace(decoder, transformer=replace(decoder.transformer, layers=tuple(layers)))


def zero_mlp_channels(mlp: DenseMLP, channels: tuple[int, ...]) -> DenseMLP:
    up_projection = _require_full_precision(mlp.up_projection, "up_projection")
    down_projection = _require_full_precision(mlp.down_projection, "down_projection")
    if mlp.mixture_size is not None:
        raise ValueError("MLP zero-channel induction requires a non-mixture DenseMLP")

    channels = _normalize_channels(channels, mlp.hidden_dim)
    hidden_dim = mlp.hidden_dim
    up_rows = jnp.asarray(channels + tuple(hidden_dim + channel for channel in channels), dtype=jnp.int32)
    down_columns = jnp.asarray(channels, dtype=jnp.int32)

    zeroed_up = replace(
        up_projection,
        weights=up_projection.weights.at[up_rows, :].set(0),
        biases=_zero_vector_indices(up_projection.biases, up_rows),
    )
    zeroed_down = replace(
        down_projection,
        weights=down_projection.weights.at[:, down_columns].set(0),
    )
    return replace(mlp, up_projection=zeroed_up, down_projection=zeroed_down)


def _validate_spec(spec: MlpZeroChannelSpec) -> MlpZeroChannelSpec:
    if not spec.layers:
        raise ValueError("MLP zero-channel spec must contain at least one layer")
    for layer in spec.layers:
        if layer.layer_index < 0:
            raise ValueError("MLP zero-channel layer indices must be non-negative")
        if not layer.channels:
            raise ValueError("MLP zero-channel layers must contain at least one channel")
    return spec


def _normalize_channels(channels: tuple[int, ...], hidden_dim: int) -> tuple[int, ...]:
    if not channels:
        raise ValueError("MLP zero-channel list must contain at least one channel")
    result = tuple(sorted(channels))
    if len(set(result)) != len(result):
        raise ValueError(f"Duplicate MLP zero-channel index in {channels}")
    if result[0] < 0 or result[-1] >= hidden_dim:
        raise ValueError(f"MLP zero-channel indices {channels} exceed hidden dimension {hidden_dim}")
    return result


def _require_full_precision(linear: object, name: str) -> FullPrecisionLinear:
    if not isinstance(linear, FullPrecisionLinear):
        raise TypeError(f"MLP zero-channel induction requires full precision {name}, got {type(linear).__name__}")
    return linear


def _zero_vector_indices(values: Array | None, indices: Array) -> Array | None:
    if values is None:
        return None
    return values.at[indices].set(0)
