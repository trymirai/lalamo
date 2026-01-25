import gc
import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import equinox as eqx
import jax

jax.config.update("jax_compiler_enable_remat_pass", False)
import jax.numpy as jnp

from lalamo.message_processor import UserMessage
from lalamo.model_import import REPO_TO_MODEL, import_model


def _read_peak_bytes(stats: dict[str, int] | None) -> int | None:
    if stats is None:
        return None
    if "peak_bytes_in_use" in stats:
        return stats["peak_bytes_in_use"]
    if "bytes_in_use" in stats:
        return stats["bytes_in_use"]
    return None


def _snapshot_bytes(stats: dict[str, int] | None) -> int | None:
    if stats is None:
        return None
    if "bytes_in_use" in stats:
        return stats["bytes_in_use"]
    return None


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "n/a"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} PiB"


def _run_and_get_peak(
    model,
    *,
    device: jax.Device,
    batch_size: int,
    max_input_length: int,
    max_output_length: int,
    num_logits_per_token: int,
    prompt_token_ids: jax.Array,
) -> dict[str, int | None]:
    before_stats = device.memory_stats()
    before_peak = _read_peak_bytes(before_stats)
    before_bytes = _snapshot_bytes(before_stats)

    prompt_len = prompt_token_ids.size
    if prompt_len > max_input_length:
        raise ValueError("prompt exceeds max_input_length")

    prompt_token_ids = jax.device_put(prompt_token_ids, device)
    padded = jnp.pad(
        prompt_token_ids,
        (0, max_input_length - prompt_len),
        constant_values=0,
    )
    batch_prompt_ids = jnp.repeat(padded[None, :], batch_size, axis=0)
    batch_prompt_lengths = jnp.array([prompt_len] * batch_size, dtype=jnp.int32)

    with jax.default_device(device):
        results = eqx.filter_jit(model.generate_tokens)(
            batch_prompt_ids,
            prompt_lengths_without_padding=batch_prompt_lengths,
            max_output_length=max_output_length,
            num_top_logits_to_return=num_logits_per_token,
        )
    jax.block_until_ready(results.token_ids)

    after_stats = device.memory_stats()
    after_peak = _read_peak_bytes(after_stats)
    after_bytes = _snapshot_bytes(after_stats)

    # Best-effort cleanup so later runs don't see cached allocations.
    del results
    del batch_prompt_ids
    del batch_prompt_lengths
    jax.clear_caches()
    gc.collect()

    if after_peak is None and before_peak is None and after_bytes is None and before_bytes is None:
        raise RuntimeError("device.memory_stats missing peak_bytes_in_use/bytes_in_use")

    return {
        "before_peak": before_peak,
        "before_current": before_bytes,
        "after_peak": after_peak,
        "after_current": after_bytes,
    }


def main() -> None:
    model_repo = sys.argv[1]
    batch_size = int(sys.argv[2])

    gpu_devices = jax.devices("gpu")
    if not gpu_devices:
        raise RuntimeError("No GPU devices detected; set JAX to use a GPU for this script.")
    device = gpu_devices[0]

    start_stats = device.memory_stats()
    start_current = _snapshot_bytes(start_stats)

    model = import_model(REPO_TO_MODEL[model_repo]).model
    prompt = [UserMessage("Count from 1 to 20 separated by spaces.")]
    prompt_token_ids = jnp.array(model.message_processor.tokenize_request(prompt), dtype=jnp.int32)

    max_input_length = max(128, prompt_token_ids.size)
    max_output_length = 32
    num_logits_per_token = 8

    stats = _run_and_get_peak(
        model,
        device=device,
        batch_size=batch_size,
        max_input_length=max_input_length,
        max_output_length=max_output_length,
        num_logits_per_token=num_logits_per_token,
        prompt_token_ids=prompt_token_ids,
    )

    after_peak = stats["after_peak"]

    assert after_peak is not None
    peak_delta_from_start_current = max(0, after_peak - start_current) if start_current is not None else None

    payload = {
        "model": model_repo,
        "batch_size": batch_size,
        "after_peak_bytes": after_peak,
        "after_peak_human": _format_bytes(after_peak),
        "peak_delta_from_start_current_bytes": peak_delta_from_start_current,
        "peak_delta_from_start_current_human": _format_bytes(peak_delta_from_start_current),
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
