from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from lalamo.qwen_moe_routing import (
    AUTO_DEVICE_MAP_HEADROOM_GIB,
    IndexedConversationSample,
    ModelRoutingConfig,
    ProcessedSample,
    RuntimeInfo,
    dataset_fingerprint,
    indexed_conversation_samples,
    load_model,
    load_rows,
    model_routing_config,
    parse_ewma_alphas,
    prompt_and_continuation_ids,
    runtime_info,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from transformers.models.qwen3_5_moe import Qwen3_5MoeForCausalLM
    from transformers.tokenization_utils import PreTrainedTokenizer


NLL_AUTO_DEVICE_MAP_HEADROOM_GIB = AUTO_DEVICE_MAP_HEADROOM_GIB + 12


@dataclass(frozen=True)
class EvalConfig:
    model_repo: str
    dataset: str
    dataset_split: str
    output_path: Path
    seed: int
    max_rows: int | None
    max_prompts: int | None
    max_prompt_tokens: int
    max_continuation_tokens: int
    ewma_alphas: tuple[float, ...]
    device_map_mode: str


@dataclass(frozen=True)
class LossStatistics:
    token_weighted_mean_continuation_nll: float
    token_weighted_continuation_perplexity: float
    sequence_weighted_mean_continuation_nll: float
    sequence_weighted_mean_continuation_nll_std: float
    sequence_weighted_mean_continuation_nll_sem: float
    sequence_weighted_mean_continuation_nll_ci95: float
    sequence_weighted_continuation_perplexity: float


@dataclass(frozen=True)
class VariantResult:
    name: str
    alpha: float
    statistics: LossStatistics
    delta_token_weighted_mean_continuation_nll_vs_baseline: float
    delta_sequence_weighted_mean_continuation_nll_vs_baseline: float


@dataclass
class LossAccumulator:
    total_nll: float = 0.0
    token_count: int = 0
    sample_mean_nll: list[float] | None = None

    def __post_init__(self) -> None:
        self.sample_mean_nll = []

    def update(self, token_nll: torch.Tensor) -> None:
        if token_nll.numel() == 0:
            return
        self.total_nll += float(token_nll.sum())
        self.token_count += int(token_nll.numel())
        assert self.sample_mean_nll is not None
        self.sample_mean_nll.append(float(token_nll.mean()))

    def finalize(self) -> LossStatistics:
        assert self.sample_mean_nll is not None
        if self.token_count == 0:
            return LossStatistics(
                token_weighted_mean_continuation_nll=0.0,
                token_weighted_continuation_perplexity=1.0,
                sequence_weighted_mean_continuation_nll=0.0,
                sequence_weighted_mean_continuation_nll_std=0.0,
                sequence_weighted_mean_continuation_nll_sem=0.0,
                sequence_weighted_mean_continuation_nll_ci95=0.0,
                sequence_weighted_continuation_perplexity=1.0,
            )
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
class EwmaEvalResult:
    config: EvalConfig
    model: ModelRoutingConfig
    runtime: RuntimeInfo
    dataset_fingerprint: str
    dataset_rows_total: int
    assistant_turns_total: int
    baseline: VariantResult
    ewma_variants: tuple[VariantResult, ...]
    dataset_rows_processed: int
    prompts_processed: int
    skipped_prompt_too_long: int
    skipped_continuation_too_long: int
    skipped_empty_continuation: int
    processed_samples: tuple[ProcessedSample, ...]


def continuation_token_nll(
    logits: torch.Tensor,
    full_ids: torch.Tensor,
    prompt_length: int,
    continuation_length: int,
) -> torch.Tensor:
    if logits.ndim != 3 or full_ids.ndim != 2:
        raise ValueError(
            f"Expected logits [batch,tokens,vocab] and ids [batch,tokens], got {logits.shape} and {full_ids.shape}."
        )
    if logits.shape[0] != 1 or full_ids.shape[0] != 1:
        raise ValueError("This evaluation expects batch size 1.")
    if continuation_length <= 0:
        raise ValueError("continuation_length must be positive.")
    start = prompt_length - 1
    stop = start + continuation_length
    if start < 0 or stop > full_ids.shape[1] - 1:
        raise ValueError(
            "Invalid prompt/continuation split: "
            f"prompt={prompt_length}, continuation={continuation_length}, total={full_ids.shape[1]}."
        )
    selected_logits = logits[:, start:stop, :].to(device="cpu", dtype=torch.float32)
    selected_targets = full_ids[:, start + 1 : stop + 1].to(device="cpu")
    losses = F.cross_entropy(
        selected_logits.view(-1, selected_logits.shape[-1]),
        selected_targets.view(-1),
        reduction="none",
    )
    return losses.view(continuation_length)


def next_token_nll(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or target_ids.ndim != 1:
        raise ValueError(
            f"Expected logits [batch,vocab] and targets [batch], got {logits.shape} and {target_ids.shape}."
        )
    if logits.shape[0] != target_ids.shape[0]:
        raise ValueError(f"Batch mismatch between logits {logits.shape} and targets {target_ids.shape}.")
    return F.cross_entropy(logits.to(device="cpu", dtype=torch.float32), target_ids.to(device="cpu"), reduction="none")


def ewma_router_outputs(
    router_probs: torch.Tensor, top_k: int, alpha: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    smoothed_probs = router_probs.clone()
    for token_index in range(1, smoothed_probs.shape[0]):
        smoothed_probs[token_index] = (
            alpha * smoothed_probs[token_index] + (1.0 - alpha) * smoothed_probs[token_index - 1]
        )
    router_top_value, router_indices = torch.topk(smoothed_probs, top_k, dim=-1)
    router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
    router_top_value = router_top_value.to(smoothed_probs.dtype)
    return smoothed_probs, router_top_value, router_indices


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


def evaluate_variant(
    model: Qwen3_5MoeForCausalLM,
    samples: tuple[IndexedConversationSample, ...],
    tokenizer: PreTrainedTokenizer,
    max_prompt_tokens: int,
    max_continuation_tokens: int,
    routing_state: SequenceEWMARouter | None = None,
) -> tuple[LossAccumulator, int, int, int, int, tuple[ProcessedSample, ...]]:
    accumulator = LossAccumulator()
    prompts_processed = 0
    skipped_prompt_too_long = 0
    skipped_continuation_too_long = 0
    skipped_empty_continuation = 0
    processed_samples: list[ProcessedSample] = []
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
        prompt_length = prompt_ids.shape[1]
        continuation_length = continuation_ids.shape[1]
        input_device = model.get_input_embeddings().weight.device
        if routing_state is not None:
            routing_state.reset_sequence()
        token_losses: list[torch.Tensor] = []
        with torch.inference_mode():
            outputs = model(
                input_ids=prompt_ids.to(input_device),
                use_cache=True,
                output_router_logits=False,
                return_dict=True,
            )
            token_losses.append(next_token_nll(outputs.logits[:, -1, :], continuation_ids[:, 0].to(input_device)))
            past_key_values = outputs.past_key_values
            for token_index in range(continuation_length - 1):
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
        token_nll = torch.cat(token_losses)
        accumulator.update(token_nll)
        prompts_processed += 1
        processed_samples.append(
            ProcessedSample(
                row_id=indexed_sample.row_id,
                conversation_index=indexed_sample.conversation_index,
                assistant_turn_index=indexed_sample.assistant_turn_index,
                prompt_tokens=prompt_length,
                continuation_tokens=continuation_length,
            )
        )
    return (
        accumulator,
        prompts_processed,
        skipped_prompt_too_long,
        skipped_continuation_too_long,
        skipped_empty_continuation,
        tuple(processed_samples),
    )


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Local dataset path or Hugging Face dataset repo id.")
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--model-repo", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-continuation-tokens", type=int, default=512)
    parser.add_argument("--ewma-alphas", default="0.8")
    parser.add_argument("--device-map-mode", choices=("single-gpu", "auto", "balanced-low-0"), default=None)
    args = parser.parse_args()
    device_map_mode = args.device_map_mode
    if device_map_mode is None:
        device_map_mode = "single-gpu" if torch.cuda.is_available() else "auto"
    return EvalConfig(
        model_repo=args.model_repo,
        dataset=args.dataset,
        dataset_split=args.dataset_split,
        output_path=args.output_path,
        seed=args.seed,
        max_rows=args.max_rows,
        max_prompts=args.max_prompts,
        max_prompt_tokens=args.max_prompt_tokens,
        max_continuation_tokens=args.max_continuation_tokens,
        ewma_alphas=parse_ewma_alphas(args.ewma_alphas),
        device_map_mode=device_map_mode,
    )


def main() -> None:
    config = parse_args()
    routing_config = model_routing_config(config.model_repo)
    rows, dataset_rows_total = load_rows(config.dataset, config.dataset_split, config.max_rows, config.seed)
    samples = indexed_conversation_samples(rows, config.seed)
    assistant_turns_total = len(samples)
    if config.max_prompts is not None:
        samples = samples[: config.max_prompts]
    tokenizer = AutoTokenizer.from_pretrained(config.model_repo)
    model = load_model(
        config.model_repo,
        config.device_map_mode,
        auto_headroom_gib=NLL_AUTO_DEVICE_MAP_HEADROOM_GIB,
    )
    model.eval()

    (
        baseline_accumulator,
        prompts_processed,
        skipped_prompt_too_long,
        skipped_continuation_too_long,
        skipped_empty_continuation,
        processed_samples,
    ) = evaluate_variant(
        model=model,
        samples=samples,
        tokenizer=tokenizer,
        max_prompt_tokens=config.max_prompt_tokens,
        max_continuation_tokens=config.max_continuation_tokens,
    )
    baseline = VariantResult(
        name="baseline",
        alpha=0.0,
        statistics=baseline_accumulator.finalize(),
        delta_token_weighted_mean_continuation_nll_vs_baseline=0.0,
        delta_sequence_weighted_mean_continuation_nll_vs_baseline=0.0,
    )

    ewma_variants = []
    for alpha in config.ewma_alphas:
        with patched_ewma_routing(model, alpha) as routing_state:
            (
                accumulator,
                alpha_prompts_processed,
                alpha_skipped_prompt_too_long,
                alpha_skipped_continuation_too_long,
                alpha_skipped_empty_continuation,
                alpha_processed_samples,
            ) = evaluate_variant(
                model=model,
                samples=samples,
                tokenizer=tokenizer,
                max_prompt_tokens=config.max_prompt_tokens,
                max_continuation_tokens=config.max_continuation_tokens,
                routing_state=routing_state,
            )
        if (
            alpha_prompts_processed != prompts_processed
            or alpha_skipped_prompt_too_long != skipped_prompt_too_long
            or alpha_skipped_continuation_too_long != skipped_continuation_too_long
            or alpha_skipped_empty_continuation != skipped_empty_continuation
            or alpha_processed_samples != processed_samples
        ):
            raise ValueError("EWMA evaluation changed the sampled prompt set; this invalidates baseline comparison.")
        statistics = accumulator.finalize()
        ewma_variants.append(
            VariantResult(
                name=f"ewma_{alpha:.3f}",
                alpha=alpha,
                statistics=statistics,
                delta_token_weighted_mean_continuation_nll_vs_baseline=(
                    statistics.token_weighted_mean_continuation_nll
                    - baseline.statistics.token_weighted_mean_continuation_nll
                ),
                delta_sequence_weighted_mean_continuation_nll_vs_baseline=(
                    statistics.sequence_weighted_mean_continuation_nll
                    - baseline.statistics.sequence_weighted_mean_continuation_nll
                ),
            )
        )

    result = EwmaEvalResult(
        config=config,
        model=routing_config,
        runtime=runtime_info(model, config.device_map_mode),
        dataset_fingerprint=dataset_fingerprint(rows),
        dataset_rows_total=dataset_rows_total,
        assistant_turns_total=assistant_turns_total,
        baseline=baseline,
        ewma_variants=tuple(ewma_variants),
        dataset_rows_processed=len(rows),
        prompts_processed=prompts_processed,
        skipped_prompt_too_long=skipped_prompt_too_long,
        skipped_continuation_too_long=skipped_continuation_too_long,
        skipped_empty_continuation=skipped_empty_continuation,
        processed_samples=processed_samples,
    )
    payload = asdict(result)
    payload["config"]["output_path"] = str(config.output_path)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
