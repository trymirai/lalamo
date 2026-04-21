import pytest
import torch

from lalamo.qwen_moe_eval_common import PreparedSample, PreparedSamples, SequenceEWMARouter, next_token_nll
from lalamo.qwen_moe_ewma_eval import evaluate_variant
from lalamo.qwen_moe_routing import ProcessedSample

pytestmark = pytest.mark.fast


def test_next_token_nll_matches_cross_entropy() -> None:
    logits = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float16)
    targets = torch.tensor([2], dtype=torch.long)
    losses = next_token_nll(logits, targets)
    expected = torch.nn.functional.cross_entropy(logits.float(), targets, reduction="none")
    assert torch.allclose(losses, expected, atol=1e-4, rtol=1e-4)


def test_sequence_ewma_router_carries_state_across_decode_steps() -> None:
    router = SequenceEWMARouter.build(1)
    first_probs = torch.tensor([[0.90, 0.10, 0.00]], dtype=torch.float32)
    second_probs = torch.tensor([[0.10, 0.90, 0.00]], dtype=torch.float32)

    first_smoothed, _, first_indices = router.apply(0, first_probs, top_k=1, alpha=0.25)
    second_smoothed, _, second_indices = router.apply(0, second_probs, top_k=1, alpha=0.25)

    assert torch.allclose(first_smoothed, first_probs)
    assert first_indices[:, 0].tolist() == [0]
    assert torch.allclose(second_smoothed, torch.tensor([[0.70, 0.30, 0.00]]))
    assert second_indices[:, 0].tolist() == [0]

    router.reset_sequence()
    reset_smoothed, _, reset_indices = router.apply(0, second_probs, top_k=1, alpha=0.25)
    assert torch.allclose(reset_smoothed, second_probs)
    assert reset_indices[:, 0].tolist() == [1]


def test_evaluate_variant_uses_cached_teacher_forcing() -> None:
    prompt_ids = torch.tensor([[10, 11]], dtype=torch.long)
    continuation_ids = torch.tensor([[1, 2]], dtype=torch.long)

    class FakeEmbeddings:
        def __init__(self) -> None:
            self.weight = torch.zeros((1, 1))

    class FakeModel:
        def __init__(self) -> None:
            self.embed = FakeEmbeddings()
            self.calls: list[tuple[tuple[int, ...], object]] = []

        def get_input_embeddings(self) -> FakeEmbeddings:
            return self.embed

        def __call__(
            self,
            *,
            input_ids: torch.Tensor,
            use_cache: bool,
            output_router_logits: bool,
            return_dict: bool,
            past_key_values: object | None = None,
        ) -> object:
            assert use_cache
            assert not output_router_logits
            assert return_dict
            self.calls.append((tuple(input_ids.view(-1).tolist()), past_key_values))
            logits = torch.full((1, input_ids.shape[1], 4), -10.0)
            if past_key_values is None:
                logits[0, -1, 1] = 0.0
                return type("Output", (), {"logits": logits, "past_key_values": "pkv0"})()
            assert past_key_values == "pkv0"
            logits[0, -1, 2] = 0.0
            return type("Output", (), {"logits": logits, "past_key_values": "pkv1"})()

    prepared_samples = PreparedSamples(
        samples=(
            PreparedSample(
                prompt_ids=prompt_ids,
                continuation_ids=continuation_ids,
                processed_sample=ProcessedSample(
                    row_id=7,
                    conversation_index=0,
                    assistant_turn_index=0,
                    prompt_tokens=2,
                    continuation_tokens=2,
                ),
            ),
        ),
        skipped_prompt_too_long=0,
        skipped_continuation_too_long=0,
        skipped_empty_continuation=0,
    )
    model = FakeModel()
    accumulator = evaluate_variant(model=model, prepared_samples=prepared_samples)

    assert model.calls == [((10, 11), None), ((1,), "pkv0")]
    assert accumulator.token_count == 2
    assert accumulator.total_nll < 1e-3
