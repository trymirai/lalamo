import os

import numpy as np
import pytest
import torch
import transformers
from transformers.tokenization_utils_base import BatchEncoding

from lalamo.qwen_moe_routing import (
    AgreementAccumulator,
    ChatMessage,
    ConversationSample,
    MessageRole,
    WindowAccumulator,
    auto_device_map_kwargs,
    conversation_samples,
    default_cache_sizes,
    distinct_experts_in_windows,
    ewma_topk_active_experts,
    load_model,
    lru_resident_budget_counts,
    oracle_cache_hit_rates_in_windows,
    parse_ewma_alphas,
    parse_window_sizes,
    prompt_and_continuation_ids,
    random_distinct_expert_baseline,
    row_conversations,
    topk_active_experts,
    topk_overlap_counts,
)

pytestmark = pytest.mark.fast


def test_parse_window_sizes_rejects_unsorted_input() -> None:
    try:
        parse_window_sizes("4,2,8")
    except ValueError as error:
        assert "sorted ascending" in str(error)
    else:
        raise AssertionError("parse_window_sizes should reject unsorted input.")


def test_parse_ewma_alphas_accepts_sorted_fractional_values() -> None:
    assert parse_ewma_alphas("0.2,0.5,0.8") == (0.2, 0.5, 0.8)
    assert parse_ewma_alphas("") == ()


def test_auto_device_map_kwargs_leaves_gpu_headroom(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeProps:
        total_memory = 80 * 1024**3

    def fake_is_dir(path: object) -> bool:
        return str(path) == "/dev/shm"

    def fake_mkdir(_path: object, parents: bool = False, exist_ok: bool = False) -> None:
        assert parents
        assert exist_ok

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _index: FakeProps())
    monkeypatch.setattr("lalamo.qwen_moe_routing.Path.is_dir", fake_is_dir)
    monkeypatch.setattr("lalamo.qwen_moe_routing.Path.mkdir", fake_mkdir)

    kwargs = auto_device_map_kwargs()

    assert kwargs["torch_dtype"] is torch.bfloat16
    assert kwargs["device_map"] == "auto"
    assert kwargs["max_memory"] == {0: "60GiB", 1: "60GiB", "cpu": "512GiB"}
    assert kwargs["offload_folder"] == "/dev/shm/qwen_moe_offload"
    assert kwargs["offload_state_dict"] is True


def test_lru_resident_budget_counts_warm_starts_continuation_cache() -> None:
    active_experts = np.asarray(
        [
            [
                [0, 1],
                [0, 2],
                [0, 2],
                [2, 3],
            ]
        ],
        dtype=np.int64,
    )

    prompt_hits, prompt_misses, continuation_hits, continuation_misses = lru_resident_budget_counts(
        active_experts,
        prompt_length=2,
        cache_sizes=(2,),
    )

    assert prompt_hits.tolist() == [1]
    assert prompt_misses.tolist() == [3]
    assert continuation_hits.tolist() == [3]
    assert continuation_misses.tolist() == [1]


def test_lru_resident_budget_counts_rejects_cache_smaller_than_topk() -> None:
    active_experts = np.asarray([[[0, 1]]], dtype=np.int64)

    with pytest.raises(ValueError, match="cache_sizes must be at least top_k"):
        lru_resident_budget_counts(active_experts, prompt_length=1, cache_sizes=(1,))


def test_auto_device_map_kwargs_accepts_custom_headroom(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeProps:
        total_memory = 80 * 1024**3

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _index: FakeProps())
    monkeypatch.setattr("lalamo.qwen_moe_routing.Path.is_dir", lambda path: str(path) == "/dev/shm")
    monkeypatch.setattr("lalamo.qwen_moe_routing.Path.mkdir", lambda *_args, **_kwargs: None)

    kwargs = auto_device_map_kwargs(headroom_gib=32)

    assert kwargs["max_memory"] == {0: "48GiB", "cpu": "512GiB"}


def test_auto_device_map_kwargs_accepts_balanced_low_0(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeProps:
        total_memory = 80 * 1024**3

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _index: FakeProps())
    monkeypatch.setattr("lalamo.qwen_moe_routing.Path.is_dir", lambda path: str(path) == "/dev/shm")
    monkeypatch.setattr("lalamo.qwen_moe_routing.Path.mkdir", lambda *_args, **_kwargs: None)

    kwargs = auto_device_map_kwargs(device_map="balanced_low_0")

    assert kwargs["device_map"] == "balanced_low_0"


def test_load_model_auto_disables_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    original_warmup = transformers.modeling_utils.caching_allocator_warmup
    captured: dict[str, object] = {}

    def fake_from_pretrained(model_repo: str, **kwargs: object) -> str:
        captured["repo"] = model_repo
        captured["kwargs"] = kwargs
        captured["parallel_loading"] = os.environ["HF_ENABLE_PARALLEL_LOADING"]
        captured["warmup"] = transformers.modeling_utils.caching_allocator_warmup
        return "loaded-model"

    monkeypatch.setenv("HF_ENABLE_PARALLEL_LOADING", "true")
    monkeypatch.setattr(
        "lalamo.qwen_moe_routing.auto_device_map_kwargs",
        lambda headroom_gib, *, device_map="auto": {"headroom": headroom_gib, "device_map": device_map},
    )
    monkeypatch.setattr("lalamo.qwen_moe_routing.Qwen3_5MoeForCausalLM.from_pretrained", fake_from_pretrained)

    loaded = load_model("repo", "auto", auto_headroom_gib=33)

    assert loaded == "loaded-model"
    assert captured["repo"] == "repo"
    assert captured["kwargs"] == {"headroom": 33, "device_map": "auto"}
    assert captured["parallel_loading"] == "false"
    assert captured["warmup"] is not original_warmup
    assert os.environ["HF_ENABLE_PARALLEL_LOADING"] == "true"
    assert transformers.modeling_utils.caching_allocator_warmup is original_warmup


def test_load_model_balanced_low_0_uses_balanced_map(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_from_pretrained(_model_repo: str, **kwargs: object) -> str:
        captured["kwargs"] = kwargs
        return "loaded-model"

    monkeypatch.setattr(
        "lalamo.qwen_moe_routing.auto_device_map_kwargs",
        lambda headroom_gib, *, device_map="auto": {"headroom": headroom_gib, "device_map": device_map},
    )
    monkeypatch.setattr("lalamo.qwen_moe_routing.Qwen3_5MoeForCausalLM.from_pretrained", fake_from_pretrained)

    loaded = load_model("repo", "balanced-low-0", auto_headroom_gib=33)

    assert loaded == "loaded-model"
    assert captured["kwargs"] == {"headroom": 33, "device_map": "balanced_low_0"}


def test_conversation_samples_returns_user_to_assistant_turns() -> None:
    messages = (
        ChatMessage(role=MessageRole.SYSTEM, content="sys"),
        ChatMessage(role=MessageRole.USER, content="u1"),
        ChatMessage(role=MessageRole.ASSISTANT, content="a1"),
        ChatMessage(role=MessageRole.USER, content="u2"),
        ChatMessage(role=MessageRole.ASSISTANT, content="a2"),
    )

    samples = conversation_samples(messages)

    assert samples == (
        ConversationSample(
            prompt=(
                ChatMessage(role=MessageRole.SYSTEM, content="sys"),
                ChatMessage(role=MessageRole.USER, content="u1"),
            ),
            continuation=ChatMessage(role=MessageRole.ASSISTANT, content="a1"),
        ),
        ConversationSample(
            prompt=(
                ChatMessage(role=MessageRole.SYSTEM, content="sys"),
                ChatMessage(role=MessageRole.USER, content="u1"),
                ChatMessage(role=MessageRole.ASSISTANT, content="a1"),
                ChatMessage(role=MessageRole.USER, content="u2"),
            ),
            continuation=ChatMessage(role=MessageRole.ASSISTANT, content="a2"),
        ),
    )


def test_row_conversations_supports_arena_schema() -> None:
    row = {
        "conversation_a": [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ],
        "conversation_b": [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "b"},
        ],
    }

    conversations = row_conversations(row)

    assert len(conversations) == 2
    assert conversations[0][1].content == "a"
    assert conversations[1][1].content == "b"


def test_row_conversations_supports_hermes_schema() -> None:
    row = {
        "conversations": [
            {"from": "system", "value": "sys"},
            {"from": "human", "value": "u"},
            {"from": "gpt", "value": "a"},
        ]
    }

    (conversation,) = row_conversations(row)

    assert conversation == (
        ChatMessage(role=MessageRole.SYSTEM, content="sys"),
        ChatMessage(role=MessageRole.USER, content="u"),
        ChatMessage(role=MessageRole.ASSISTANT, content="a"),
    )


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_tensors: str | None = None,
    ) -> str:
        assert not tokenize
        assert return_tensors is None
        text = "".join(f"<{message['role']}>{message['content']}" for message in messages)
        if add_generation_prompt:
            return f"{text}<assistant>"
        return text

    def __call__(self, text: str, *, add_special_tokens: bool, return_tensors: str) -> BatchEncoding:
        assert not add_special_tokens
        assert return_tensors == "pt"
        token_ids = torch.tensor([[ord(character) for character in text]], dtype=torch.long)
        return BatchEncoding({"input_ids": token_ids})


def test_prompt_and_continuation_ids_split_full_suffix_text() -> None:
    tokenizer = FakeTokenizer()
    sample = ConversationSample(
        prompt=(
            ChatMessage(role=MessageRole.SYSTEM, content="sys"),
            ChatMessage(role=MessageRole.USER, content="u"),
        ),
        continuation=ChatMessage(role=MessageRole.ASSISTANT, content="answer"),
    )

    prompt_ids, continuation_ids, skip_reason = prompt_and_continuation_ids(
        tokenizer=tokenizer,
        sample=sample,
        max_prompt_tokens=1024,
        max_continuation_tokens=1024,
    )

    assert skip_reason is None
    assert prompt_ids is not None
    assert continuation_ids is not None
    assert prompt_ids.tolist() == [[ord(character) for character in "<system>sys<user>u<assistant>"]]
    assert continuation_ids.tolist() == [[ord(character) for character in "answer"]]


def test_distinct_experts_in_windows_counts_per_layer() -> None:
    active_experts = np.asarray(
        [
            [[0, 1], [1, 2], [2, 3], [3, 4]],
            [[0, 1], [0, 1], [0, 1], [4, 5]],
        ],
        dtype=np.int16,
    )

    distinct_counts, num_windows = distinct_experts_in_windows(active_experts, window_size=2, num_experts=8)

    assert num_windows == 3
    np.testing.assert_array_equal(distinct_counts[0], np.asarray([3, 3, 3], dtype=np.float64))
    np.testing.assert_array_equal(distinct_counts[1], np.asarray([2, 2, 4], dtype=np.float64))


def test_window_accumulator_reports_window_and_sequence_weighted_means() -> None:
    accumulator = WindowAccumulator(
        window_size=2,
        num_layers=1,
        num_experts=4,
        num_active_experts=1,
        cache_sizes=(1, 2, 4),
    )

    accumulator.update(np.asarray([[[0], [0]]], dtype=np.int16))
    accumulator.update(np.asarray([[[0], [1], [2]]], dtype=np.int16))

    statistics = accumulator.finalize()

    assert statistics.num_windows == 3
    assert statistics.sequence_count_with_windows == 2
    assert statistics.window_weighted_mean_distinct_experts_overall == pytest.approx(5.0 / 3.0)
    assert statistics.sequence_weighted_mean_distinct_experts_overall == pytest.approx(1.5)
    assert statistics.sequence_weighted_mean_distinct_experts_overall_ci95 > 0.0
    assert statistics.oracle_cache_hit_rates[0].cache_size == 1
    assert statistics.oracle_cache_hit_rates[0].window_weighted_hit_rate == pytest.approx(2.0 / 3.0)
    assert statistics.oracle_cache_hit_rates[0].sequence_weighted_hit_rate == pytest.approx(0.75)
    assert statistics.oracle_cache_hit_rates[1].window_weighted_hit_rate == pytest.approx(1.0)


def test_oracle_cache_hit_rates_in_windows_matches_top_frequency_oracle() -> None:
    active_experts = np.asarray([[[0], [1], [0], [2]]], dtype=np.int16)

    hit_rates = oracle_cache_hit_rates_in_windows(
        active_experts,
        window_size=2,
        num_experts=4,
        num_active_experts=1,
        cache_sizes=(1, 2),
    )

    np.testing.assert_allclose(hit_rates[0], np.asarray([0.5, 0.5, 0.5], dtype=np.float64))
    np.testing.assert_allclose(hit_rates[1], np.asarray([1.0, 1.0, 1.0], dtype=np.float64))


def test_default_cache_sizes_scales_with_active_experts() -> None:
    assert default_cache_sizes(num_experts=256, num_active_experts=8) == (8, 16, 32)
    assert default_cache_sizes(num_experts=12, num_active_experts=8) == (8, 12)


def test_random_distinct_expert_baseline_matches_closed_form() -> None:
    observed = random_distinct_expert_baseline(window_size=4, num_experts=16, num_active_experts=2)
    expected = 16.0 * (1.0 - (1.0 - 2.0 / 16.0) ** 4)
    assert observed == expected


def test_topk_active_experts_preserves_token_axis() -> None:
    router_logits = (
        np.asarray([[0.1, 0.9, 0.2], [0.7, 0.2, 0.1]], dtype=np.float32),
        np.asarray([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]], dtype=np.float32),
    )
    tensors = tuple(torch.tensor(layer) for layer in router_logits)

    active = topk_active_experts(tensors, num_active_experts=2)

    assert active.shape == (2, 2, 2)
    np.testing.assert_array_equal(active[0], np.asarray([[1, 2], [0, 1]], dtype=np.int16))
    np.testing.assert_array_equal(active[1], np.asarray([[0, 1], [2, 1]], dtype=np.int16))


def test_topk_overlap_counts_treats_topk_as_sets() -> None:
    reference = np.asarray([[[0, 1], [0, 2], [3, 4]]], dtype=np.int16)
    comparison = np.asarray([[[1, 0], [0, 3], [5, 6]]], dtype=np.int16)

    overlap = topk_overlap_counts(reference, comparison)

    np.testing.assert_array_equal(overlap, np.asarray([[2, 1, 0]], dtype=np.int16))


def test_ewma_topk_active_experts_prefers_recent_history() -> None:
    router_logits = (
        torch.tensor(
            [
                [10.0, 0.0, 0.0],
                [0.0, 9.0, 0.0],
                [0.0, 0.0, 8.0],
            ]
        ),
    )

    raw_active = topk_active_experts(router_logits, num_active_experts=1)
    ewma_active = ewma_topk_active_experts(router_logits, num_active_experts=1, alpha=0.25)

    np.testing.assert_array_equal(raw_active[0], np.asarray([[0], [1], [2]], dtype=np.int16))
    np.testing.assert_array_equal(ewma_active[0], np.asarray([[0], [0], [0]], dtype=np.int16))


def test_agreement_accumulator_reports_retention_and_exact_match() -> None:
    accumulator = AgreementAccumulator(num_layers=1)

    accumulator.update(
        np.asarray([[[0, 1], [0, 1]]], dtype=np.int16),
        np.asarray([[[1, 0], [0, 2]]], dtype=np.int16),
    )
    accumulator.update(
        np.asarray([[[0, 1], [2, 3], [4, 5]]], dtype=np.int16),
        np.asarray([[[0, 1], [2, 4], [6, 7]]], dtype=np.int16),
    )

    statistics = accumulator.finalize()

    assert statistics.token_weighted_mean_retained_fraction_overall == pytest.approx(0.6)
    assert statistics.sequence_weighted_mean_retained_fraction_overall == pytest.approx(0.625)
    assert statistics.sequence_weighted_mean_retained_fraction_overall_ci95 > 0.0
    assert statistics.token_weighted_exact_match_rate == pytest.approx(0.4)
    assert statistics.sequence_weighted_exact_match_rate == pytest.approx(5.0 / 12.0)
    assert statistics.sequence_weighted_exact_match_rate_ci95 > 0.0
