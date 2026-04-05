"""Benchmark time-to-first-token for DeltaNet chunked vs sequential scan.

Loads Qwen3.5-0.8B and measures forward pass latency with ~2048 token input.
Run with: uv run pytest tests/speed/test_delta_net_ttft.py -v -s
"""

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from lalamo.model_import.common import import_model
from lalamo.modules.token_mixers.delta_net_attention import DeltaNetAttention

MODEL_REPO = "Qwen/Qwen3.5-0.8B"
NUM_TOKENS = 2048
NUM_WARMUP = 2
NUM_RUNS = 5


def _print_times(label: str, times: list[float], num_tokens: int) -> None:
    median = sorted(times)[len(times) // 2]
    print(f"\n[{label}] {num_tokens} tokens, {NUM_RUNS} runs:")
    print(f"  Median: {median:.3f}s")
    print(f"  Min:    {min(times):.3f}s")
    print(f"  Max:    {max(times):.3f}s")
    print(f"  All:    {[f'{t:.3f}' for t in times]}")


def _measure(fn, num_runs: int = NUM_RUNS, num_warmup: int = NUM_WARMUP) -> list[float]:
    for _ in range(num_warmup):
        fn()
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return times


@pytest.fixture(scope="module")
def qwen35_model():
    model, _ = import_model(
        MODEL_REPO,
        context_length=NUM_TOKENS,
        precision=jnp.float32,
    )
    return model.model


def _swap_chunk_size(decoder, chunk_size: int, min_chunk_len: int | None = None):
    """Replace all DeltaNet layers' chunk_size (and optionally min_chunk_len)."""
    from dataclasses import replace as dc_replace

    leaves, treedef = jax.tree.flatten(decoder, is_leaf=lambda x: isinstance(x, DeltaNetAttention))
    new_leaves = []
    for leaf in leaves:
        if isinstance(leaf, DeltaNetAttention):
            overrides = {"chunk_size": chunk_size}
            if min_chunk_len is not None:
                overrides["min_chunk_len"] = min_chunk_len
            new_config = dc_replace(leaf.config, **overrides)
            leaf = DeltaNetAttention(
                new_config,
                in_proj=leaf.in_proj,
                conv=leaf.conv,
                out_proj=leaf.out_proj,
                norm=leaf.norm,
                dt_bias=leaf.dt_bias,
                a_log=leaf.a_log,
                num_heads=leaf.num_heads,
                num_groups=leaf.num_groups,
                head_dim=leaf.head_dim,
                value_head_dim=leaf.value_head_dim,
                kernel_size=leaf.kernel_size,
            )
        new_leaves.append(leaf)
    return treedef.unflatten(new_leaves)


def test_full_model_ttft(qwen35_model) -> None:
    """Measure full model TTFT: chunked scan vs effectively-sequential (chunk_size > seq_len)."""
    token_ids = jnp.arange(0, NUM_TOKENS, dtype=jnp.int32)[None, :]
    token_positions = jnp.arange(0, NUM_TOKENS, dtype=jnp.int32)[None, :]

    def make_runner(model):
        def run():
            result = model(
                token_ids=token_ids,
                token_positions=token_positions,
                return_updated_state=False,
                return_activation_trace=False,
            )
            jax.block_until_ready(result.logits)

        return run

    configs = [
        ("chunk_size=16", 16),
        ("chunk_size=32", 32),
        ("chunk_size=64", 64),
        ("chunk_size=128", 128),
        ("chunk_size=256", 256),
        ("chunk_size=1024", 1024),
        ("sequential", NUM_TOKENS + 1),
    ]

    # Pre-compile all configs to avoid JIT overhead in measurements
    runners = {}
    for label, cs in configs:
        if label == "sequential":
            model = _swap_chunk_size(qwen35_model, cs, min_chunk_len=NUM_TOKENS + 1)
        else:
            model = _swap_chunk_size(qwen35_model, cs)
        runner = make_runner(model)
        runner()  # trigger JIT compilation
        jax.clear_caches()  # free compilation memory, keep compiled code
        runners[label] = runner

    results = {}
    for label, _ in configs:
        times = _measure(runners[label])
        _print_times(label, times, NUM_TOKENS)
        results[label] = sorted(times)[len(times) // 2]

    baseline = results["sequential"]
    print("\n  Speedups vs sequential:")
    for label, median in results.items():
        print(f"    {label}: {baseline / median:.2f}x")
