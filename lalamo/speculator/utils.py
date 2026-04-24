import numpy as np
from jaxtyping import Array, Int, Shaped

from lalamo.modules.decoder import DecoderActivationTrace


def extract_activations(
    trace: DecoderActivationTrace,
    sample_index: int,
    positions: int | slice | Int[Array, " positions"],
    trace_layer_outputs: tuple[int, ...] | None,
    trace_output_norm: bool,
) -> tuple[tuple[Array, ...], Array | None]:
    layer_outputs: tuple[Array, ...] = (
        ()
        if trace_layer_outputs is None
        else tuple(trace.layer_results[layer].outputs[sample_index, positions] for layer in trace_layer_outputs)
    )
    output_norm = trace.output_norm[sample_index, positions] if trace_output_norm else None
    return layer_outputs, output_norm


def pad_or_trim(arr: Shaped[np.ndarray, " in_length"], length: int, fill: int = 0) -> Shaped[np.ndarray, " length"]:
    pad_len = length - len(arr)
    if pad_len <= 0:
        return arr[:length]
    return np.concatenate([arr, np.full(pad_len, fill, dtype=arr.dtype)])
