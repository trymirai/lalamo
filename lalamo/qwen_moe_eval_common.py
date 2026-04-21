from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from types import MethodType
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from lalamo.qwen_moe_routing import IndexedConversationSample, ProcessedSample, prompt_and_continuation_ids

if TYPE_CHECKING:
    from collections.abc import Iterator
    from contextlib import AbstractContextManager

    from transformers.models.qwen3_5_moe import Qwen3_5MoeForCausalLM
    from transformers.tokenization_utils import PreTrainedTokenizer


@dataclass(frozen=True)
class LossStatistics:
    token_weighted_mean_continuation_nll: float
    token_weighted_continuation_perplexity: float
    sequence_weighted_mean_continuation_nll: float
    sequence_weighted_mean_continuation_nll_std: float
    sequence_weighted_mean_continuation_nll_sem: float
    sequence_weighted_mean_continuation_nll_ci95: float
    sequence_weighted_continuation_perplexity: float


@dataclass
class LossAccumulator:
    total_nll: float = 0.0
    token_count: int = 0
    sample_mean_nll: list[float] = field(default_factory=list)

    def update(self, token_nll: torch.Tensor) -> None:
        if token_nll.numel() == 0:
            return
        self.total_nll += float(token_nll.sum())
        self.token_count += int(token_nll.numel())
        self.sample_mean_nll.append(float(token_nll.mean()))

    def finalize(self) -> LossStatistics:
        if self.token_count == 0:
            raise ValueError("Evaluation produced no continuation tokens.")
        sequence_values = np.asarray(self.sample_mean_nll, dtype=np.float64)
        sequence_mean = float(sequence_values.mean())
        sequence_std = float(sequence_values.std(ddof=0))
        sequence_sem = float(sequence_std / np.sqrt(len(sequence_values)))
        token_mean = self.total_nll / self.token_count
        return LossStatistics(
            token_weighted_mean_continuation_nll=token_mean,
            token_weighted_continuation_perplexity=float(np.exp(token_mean)),
            sequence_weighted_mean_continuation_nll=sequence_mean,
            sequence_weighted_mean_continuation_nll_std=sequence_std,
            sequence_weighted_mean_continuation_nll_sem=sequence_sem,
            sequence_weighted_mean_continuation_nll_ci95=1.96 * sequence_sem,
            sequence_weighted_continuation_perplexity=float(np.exp(sequence_mean)),
        )


@dataclass(frozen=True)
class PreparedSample:
    prompt_ids: torch.Tensor
    continuation_ids: torch.Tensor
    processed_sample: ProcessedSample


@dataclass(frozen=True)
class PreparedSamples:
    samples: tuple[PreparedSample, ...]
    skipped_prompt_too_long: int
    skipped_continuation_too_long: int
    skipped_empty_continuation: int

    @property
    def prompts_processed(self) -> int:
        return len(self.samples)

    @property
    def processed_samples(self) -> tuple[ProcessedSample, ...]:
        return tuple(sample.processed_sample for sample in self.samples)


@dataclass
class SequenceEWMARouter:
    previous_router_probs: list[torch.Tensor | None]

    @staticmethod
    def build(num_layers: int) -> SequenceEWMARouter:
        return SequenceEWMARouter(previous_router_probs=[None] * num_layers)

    def reset_sequence(self) -> None:
        self.previous_router_probs = [None] * len(self.previous_router_probs)

    def apply(
        self,
        layer_index: int,
        router_probs: torch.Tensor,
        top_k: int,
        alpha: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        smoothed_probs = router_probs.clone()
        previous_router_probs = self.previous_router_probs[layer_index]
        if previous_router_probs is not None:
            smoothed_probs[0] = alpha * smoothed_probs[0] + (1.0 - alpha) * previous_router_probs.to(smoothed_probs)
        for token_index in range(1, smoothed_probs.shape[0]):
            smoothed_probs[token_index] = (
                alpha * smoothed_probs[token_index] + (1.0 - alpha) * smoothed_probs[token_index - 1]
            )
        self.previous_router_probs[layer_index] = smoothed_probs[-1].detach()
        router_top_value, router_indices = torch.topk(smoothed_probs, top_k, dim=-1)
        router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(smoothed_probs.dtype)
        return smoothed_probs, router_top_value, router_indices


def prepare_samples(
    samples: tuple[IndexedConversationSample, ...],
    tokenizer: PreTrainedTokenizer,
    max_prompt_tokens: int,
    max_continuation_tokens: int,
) -> PreparedSamples:
    skipped_prompt_too_long = 0
    skipped_continuation_too_long = 0
    skipped_empty_continuation = 0
    prepared_samples: list[PreparedSample] = []
    for indexed_sample in samples:
        prompt_ids, continuation_ids, skip_reason = prompt_and_continuation_ids(
            tokenizer=tokenizer,
            sample=indexed_sample.sample,
            max_prompt_tokens=max_prompt_tokens,
            max_continuation_tokens=max_continuation_tokens,
        )
        if skip_reason == "prompt_too_long":
            skipped_prompt_too_long += 1
            continue
        if skip_reason == "continuation_too_long":
            skipped_continuation_too_long += 1
            continue
        if skip_reason == "empty_continuation":
            skipped_empty_continuation += 1
            continue
        assert prompt_ids is not None
        assert continuation_ids is not None
        prepared_samples.append(
            PreparedSample(
                prompt_ids=prompt_ids,
                continuation_ids=continuation_ids,
                processed_sample=ProcessedSample(
                    row_id=indexed_sample.row_id,
                    conversation_index=indexed_sample.conversation_index,
                    assistant_turn_index=indexed_sample.assistant_turn_index,
                    prompt_tokens=prompt_ids.shape[1],
                    continuation_tokens=continuation_ids.shape[1],
                ),
            )
        )
    return PreparedSamples(
        samples=tuple(prepared_samples),
        skipped_prompt_too_long=skipped_prompt_too_long,
        skipped_continuation_too_long=skipped_continuation_too_long,
        skipped_empty_continuation=skipped_empty_continuation,
    )


def next_token_nll(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or target_ids.ndim != 1:
        raise ValueError(
            f"Expected logits [batch,vocab] and targets [batch], got {logits.shape} and {target_ids.shape}."
        )
    if logits.shape[0] != target_ids.shape[0]:
        raise ValueError(f"Batch mismatch between logits {logits.shape} and targets {target_ids.shape}.")
    return F.cross_entropy(logits.to(device="cpu", dtype=torch.float32), target_ids.to(device="cpu"), reduction="none")


def teacher_forced_token_nll(
    model: Qwen3_5MoeForCausalLM,
    prompt_ids: torch.Tensor,
    continuation_ids: torch.Tensor,
    *,
    prompt_context: AbstractContextManager[None] | None = None,
    continuation_context: AbstractContextManager[None] | None = None,
) -> torch.Tensor:
    input_device = model.get_input_embeddings().weight.device
    prompt_context = nullcontext() if prompt_context is None else prompt_context
    continuation_context = nullcontext() if continuation_context is None else continuation_context
    token_losses: list[torch.Tensor] = []
    with torch.inference_mode():
        with prompt_context:
            outputs = model(
                input_ids=prompt_ids.to(input_device),
                use_cache=True,
                output_router_logits=False,
                return_dict=True,
            )
        token_losses.append(next_token_nll(outputs.logits[:, -1, :], continuation_ids[:, 0].to(input_device)))
        past_key_values = outputs.past_key_values
        with continuation_context:
            for token_index in range(continuation_ids.shape[1] - 1):
                outputs = model(
                    input_ids=continuation_ids[:, token_index : token_index + 1].to(input_device),
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_router_logits=False,
                    return_dict=True,
                )
                token_losses.append(
                    next_token_nll(outputs.logits[:, -1, :], continuation_ids[:, token_index + 1].to(input_device))
                )
                past_key_values = outputs.past_key_values
    del outputs, past_key_values
    return torch.cat(token_losses)


def moe_gates(model: Qwen3_5MoeForCausalLM) -> tuple[torch.nn.Module, ...]:
    return tuple(layer.mlp.gate for layer in model.model.layers)


@contextmanager
def patched_ewma_routing(model: Qwen3_5MoeForCausalLM, alpha: float) -> Iterator[SequenceEWMARouter]:
    gates = moe_gates(model)
    ewma_router = SequenceEWMARouter.build(len(gates))
    originals = [gate.forward for gate in gates]
    try:
        for layer_index, gate in enumerate(gates):

            def forward(
                self: torch.nn.Module,
                hidden_states: torch.Tensor,
                *,
                _alpha: float = alpha,
                _layer_index: int = layer_index,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                hidden_states = hidden_states.reshape(-1, self.hidden_dim)
                router_probs = torch.nn.functional.softmax(
                    F.linear(hidden_states, self.weight), dtype=torch.float, dim=-1
                )
                return ewma_router.apply(_layer_index, router_probs, self.top_k, _alpha)

            gate.forward = MethodType(forward, gate)
        yield ewma_router
    finally:
        for gate, original in zip(gates, originals, strict=True):
            gate.forward = original
