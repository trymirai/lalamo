import json

from pydantic import TypeAdapter

from lalamo.inference.continuous_batching import ContinuousDecodeCompletedEvent
from lalamo.models import GenerationConfig
from lalamo.server import Batch, BenchmarkMetricsCollector, LogitsResponseConfig, RequestBody


def test_batch_ids_keep_full_uuid_entropy() -> None:
    first = Batch.init(1)
    second = Batch.init(1)

    assert len(first.id) == len("batch_") + 32
    assert first.id != second.id


def test_continuous_metrics_store_one_weighted_latency_sample_per_decode_step() -> None:
    collector = BenchmarkMetricsCollector.init(
        "batch",
        (("first", "first"), ("second", "second")),
        started_at_seconds=10.0,
    )
    collector.record(
        ContinuousDecodeCompletedEvent(
            request_ids=("first", "second"),
            completed=(False, False),
            duration_seconds=0.25,
        ),
        completed_at_seconds=11.0,
    )
    collector.add_requests((("third", "third"),), admitted_at_seconds=11.5)
    collector.record(
        ContinuousDecodeCompletedEvent(
            request_ids=("third",),
            completed=(False,),
            duration_seconds=0.1,
        ),
        completed_at_seconds=12.5,
    )
    collector.record(
        ContinuousDecodeCompletedEvent(
            request_ids=("first", "second"),
            completed=(False, False),
            duration_seconds=0.5,
        ),
        completed_at_seconds=12.0,
    )

    snapshot = collector.snapshot(completed_at_seconds=13.0)

    assert len(snapshot.inter_token_latency_samples) == 1
    assert snapshot.inter_token_latency_samples[0].sample_count == 2
    assert snapshot.inter_token_latency_samples[0].latency_seconds == 0.5
    assert snapshot.requests[2].first_token_at_seconds == 1.0


def test_request_logits_empty_object_uses_default_top_k() -> None:
    request = TypeAdapter(RequestBody).validate_python(
        json.loads(
            """{
                "sequence_id": "request",
                "messages": [{"role": "user", "content": "hello"}],
                "model": "model",
                "logits": {}
            }"""
        )
    )

    assert request.logits == LogitsResponseConfig(top_k=256)


def test_request_accepts_sampling_config_and_seed() -> None:
    request = TypeAdapter(RequestBody).validate_python(
        json.loads(
            """{
                "sequence_id": "request",
                "messages": [{"role": "user", "content": "hello"}],
                "model": "model",
                "generation_config": {"temperature": 0.7, "top_p": 0.9},
                "seed": 42
            }"""
        )
    )

    assert request.generation_config == GenerationConfig(temperature=0.7, top_p=0.9)
    assert request.seed == 42
