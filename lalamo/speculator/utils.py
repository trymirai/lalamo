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
    """Extract the subset of activations a drafter asked to retain.

    ``sample_index`` picks one sample out of the leading batch dimension.

    Returns ``(layer_outputs, output_norm)`` where:
    - ``layer_outputs`` has one ``(N, d)`` array per layer in
      ``trace_layer_outputs`` (empty tuple if ``None``).
    - ``output_norm`` is the ``(N, d)`` slice if ``trace_output_norm`` is True,
      else ``None``.
    """
    layer_outputs: tuple[Array, ...] = (
        ()
        if trace_layer_outputs is None
        else tuple(trace.layer_results[layer].outputs[sample_index, positions] for layer in trace_layer_outputs)
    )
    output_norm = trace.output_norm[sample_index, positions] if trace_output_norm else None
    return layer_outputs, output_norm


def pad_or_trim(arr: Shaped[np.ndarray, " in_length"], length: int, fill: int = 0) -> Shaped[np.ndarray, " length"]:
    """Pad or truncate a 1-D array to ``length``.

    Used to give XLA a fixed-shape input, which avoids re-compilation when the
    logical payload length varies across speculation steps. If ``arr`` is
    longer than ``length`` it is truncated; if shorter it is padded with ``fill``.
    """
    pad_len = length - len(arr)
    if pad_len <= 0:
        return arr[:length]
    return np.concatenate([arr, np.full(pad_len, fill, dtype=arr.dtype)])
