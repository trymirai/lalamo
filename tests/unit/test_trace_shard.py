from pathlib import Path

import jax.numpy as jnp

from lalamo.data.trace_shard import TraceCompletionRecord, TraceShard, TraceShardWriter
from lalamo.models.language_model import GenerationTraceConfig


def _build_record(prefix: list[int], completion: list[int], *, offset: int) -> TraceCompletionRecord:
    completion_length = len(completion)
    raw_topk_token_ids = jnp.asarray(
        [[offset + 10, offset + 11], [offset + 12, offset + 13]][:completion_length],
        dtype=jnp.int32,
    )
    raw_topk_token_logits = jnp.asarray(
        [[0.1, -0.2], [0.3, -0.4]][:completion_length],
        dtype=jnp.bfloat16,
    )
    output_norm_hidden = jnp.asarray(
        [[7.0, 8.0], [9.0, 10.0]][:completion_length],
        dtype=jnp.bfloat16,
    )
    layer_hidden_states = jnp.asarray(
        [
            [[11.0, 12.0], [13.0, 14.0]][:completion_length],
            [[15.0, 16.0], [17.0, 18.0]][:completion_length],
        ],
        dtype=jnp.bfloat16,
    )
    return TraceCompletionRecord(
        prefix_token_ids=prefix,
        completion_token_ids=completion,
        raw_topk_token_ids=raw_topk_token_ids,
        raw_topk_token_logits=raw_topk_token_logits,
        output_norm_hidden=output_norm_hidden,
        layer_indices=(1, 3),
        layer_hidden_states=layer_hidden_states,
    )


def test_trace_shard_roundtrip(tmp_path: Path) -> None:
    output_dir = tmp_path / "traces"
    writer = TraceShardWriter(output_dir=output_dir, target_shard_size_bytes=10_000_000)
    writer.add(_build_record([1, 2], [3, 4], offset=0))
    writer.add(_build_record([5], [6, 7], offset=100))
    writer.finish()

    shard = TraceShard.from_path(output_dir / "part-00000.safetensors")
    records = list(shard.iter_completions())

    assert len(records) == 2
    assert records[0].prefix_token_ids == [1, 2]
    assert records[0].completion_token_ids == [3, 4]
    assert records[0].raw_topk_token_logits.dtype == jnp.bfloat16
    assert records[0].output_norm_hidden.dtype == jnp.bfloat16
    assert records[0].layer_hidden_states is not None
    assert records[0].layer_hidden_states.shape == (2, 2, 2)


def test_iter_trace_records_reads_trace_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "traces"
    writer = TraceShardWriter(output_dir=output_dir, target_shard_size_bytes=1)
    writer.add(_build_record([1], [2, 3], offset=0))
    writer.add(_build_record([4], [5, 6], offset=100))
    writer.finish()

    completions = list(TraceShard.iter_records(output_dir))

    assert [completion.prefix_token_ids for completion in completions] == [[1], [4]]
    assert [completion.completion_token_ids for completion in completions] == [[2, 3], [5, 6]]


def test_generation_trace_config_resolves_negative_indices() -> None:
    config = GenerationTraceConfig(trace_layers=(-1, 2, -3))

    assert config.resolve_trace_layers(8) == (2, 5, 7)
