from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING

import torch
from transformers import AutoTokenizer

from lalamo.qwen_moe_ewma_eval import (
    LossAccumulator,
    LossStatistics,
    SequenceEWMARouter,
    next_token_nll,
    patched_ewma_routing,
)
from lalamo.qwen_moe_routing import (
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
    parse_window_sizes,
    prompt_and_continuation_ids,
    runtime_info,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from transformers.models.qwen3_5_moe import Qwen3_5MoeForCausalLM
    from transformers.tokenization_utils import PreTrainedTokenizer


OFFLOAD_AUTO_DEVICE_MAP_HEADROOM_GIB = 40


@dataclass(frozen=True)
class OffloadEvalConfig:
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
    cache_sizes: tuple[int, ...]
    device_map_mode: str


@dataclass(frozen=True)
class TransferStatistics:
    cache_size: int
    resident_gib_total: float
    prompt_transfer_bytes: int
    continuation_transfer_bytes: int
    continuation_transfer_mib_per_token: float
    continuation_expert_loads_per_token: float
    continuation_hit_rate: float
    prompt_elapsed_seconds: float
    continuation_elapsed_seconds: float
    continuation_tokens_per_second: float


@dataclass(frozen=True)
class OffloadVariantResult:
    name: str
    alpha: float
    statistics: LossStatistics
    transfer: TransferStatistics
    delta_token_weighted_mean_continuation_nll_vs_baseline: float
    delta_sequence_weighted_mean_continuation_nll_vs_baseline: float


@dataclass(frozen=True)
class CacheBudgetResult:
    cache_size: int
    resident_gib_total: float
    baseline: OffloadVariantResult
    ewma_variants: tuple[OffloadVariantResult, ...]
    dataset_rows_processed: int
    prompts_processed: int
    skipped_prompt_too_long: int
    skipped_continuation_too_long: int
    skipped_empty_continuation: int
    processed_samples: tuple[ProcessedSample, ...]


@dataclass(frozen=True)
class OffloadEvalResult:
    config: OffloadEvalConfig
    model: ModelRoutingConfig
    runtime: RuntimeInfo
    dataset_fingerprint: str
    dataset_rows_total: int
    assistant_turns_total: int
    cache_budgets: tuple[CacheBudgetResult, ...]


@dataclass
class PhaseTransferAccumulator:
    bytes_loaded: int = 0
    hits: int = 0
    misses: int = 0
    elapsed_seconds: float = 0.0


@dataclass
class OffloadRuntimeState:
    expert_bytes: int
    prompt: PhaseTransferAccumulator = field(default_factory=PhaseTransferAccumulator)
    continuation: PhaseTransferAccumulator = field(default_factory=PhaseTransferAccumulator)
    phase: str = "prompt"

    def reset_variant(self) -> None:
        self.prompt = PhaseTransferAccumulator()
        self.continuation = PhaseTransferAccumulator()
        self.phase = "prompt"

    def switch_to_prompt(self) -> None:
        self.phase = "prompt"

    def switch_to_continuation(self) -> None:
        self.phase = "continuation"

    def accumulator(self) -> PhaseTransferAccumulator:
        return self.prompt if self.phase == "prompt" else self.continuation

    def record_hit(self) -> None:
        self.accumulator().hits += 1

    def record_miss(self) -> None:
        accumulator = self.accumulator()
        accumulator.misses += 1
        accumulator.bytes_loaded += self.expert_bytes


@dataclass
class ResidentExpertCache:
    cache_size: int
    gate_up_cpu: torch.Tensor
    down_cpu: torch.Tensor
    device: torch.device
    state: OffloadRuntimeState
    gate_up_cache: torch.Tensor = field(init=False)
    down_cache: torch.Tensor = field(init=False)
    expert_by_slot: list[int | None] = field(init=False)
    slot_by_expert: dict[int, int] = field(default_factory=dict)
    lru: OrderedDict[int, None] = field(default_factory=OrderedDict)

    def __post_init__(self) -> None:
        self.gate_up_cache = torch.empty(
            (self.cache_size, *self.gate_up_cpu.shape[1:]),
            device=self.device,
            dtype=self.gate_up_cpu.dtype,
        )
        self.down_cache = torch.empty(
            (self.cache_size, *self.down_cpu.shape[1:]),
            device=self.device,
            dtype=self.down_cpu.dtype,
        )
        self.expert_by_slot = [None] * self.cache_size

    def clear(self) -> None:
        self.expert_by_slot = [None] * self.cache_size
        self.slot_by_expert.clear()
        self.lru.clear()

    def weights(self, expert_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        slot = self.slot_by_expert.get(expert_idx)
        if slot is None:
            slot = self._reserve_slot(expert_idx)
            self.gate_up_cache[slot].copy_(self.gate_up_cpu[expert_idx], non_blocking=False)
            self.down_cache[slot].copy_(self.down_cpu[expert_idx], non_blocking=False)
            self.state.record_miss()
        else:
            self.state.record_hit()
            self.lru.move_to_end(expert_idx)
        return self.gate_up_cache[slot], self.down_cache[slot]

    def _reserve_slot(self, expert_idx: int) -> int:
        for slot, current_expert in enumerate(self.expert_by_slot):
            if current_expert is None:
                self.expert_by_slot[slot] = expert_idx
                self.slot_by_expert[expert_idx] = slot
                self.lru[expert_idx] = None
                return slot
        evicted_expert, _ = self.lru.popitem(last=False)
        slot = self.slot_by_expert.pop(evicted_expert)
        self.expert_by_slot[slot] = expert_idx
        self.slot_by_expert[expert_idx] = slot
        self.lru[expert_idx] = None
        return slot


@dataclass
class OffloadRuntimeController:
    caches: tuple[ResidentExpertCache, ...]
    state: OffloadRuntimeState
    expert_bytes: int
    cache_size: int

    def reset_variant(self) -> None:
        self.state.reset_variant()
        for cache in self.caches:
            cache.clear()

    def reset_sequence(self) -> None:
        self.state.switch_to_prompt()
        for cache in self.caches:
            cache.clear()

    def switch_to_continuation(self) -> None:
        self.state.switch_to_continuation()

    def finalize(self, continuation_token_count: int) -> TransferStatistics:
        resident_bytes_total = self.cache_size * len(self.caches) * self.expert_bytes
        continuation = self.state.continuation
        continuation_total_requests = continuation.hits + continuation.misses
        continuation_transfer_mib = continuation.bytes_loaded / (1024**2)
        return TransferStatistics(
            cache_size=self.cache_size,
            resident_gib_total=resident_bytes_total / (1024**3),
            prompt_transfer_bytes=self.state.prompt.bytes_loaded,
            continuation_transfer_bytes=continuation.bytes_loaded,
            continuation_transfer_mib_per_token=(
                continuation_transfer_mib / continuation_token_count if continuation_token_count > 0 else 0.0
            ),
            continuation_expert_loads_per_token=(
                continuation.misses / continuation_token_count if continuation_token_count > 0 else 0.0
            ),
            continuation_hit_rate=(
                continuation.hits / continuation_total_requests if continuation_total_requests > 0 else 0.0
            ),
            prompt_elapsed_seconds=self.state.prompt.elapsed_seconds,
            continuation_elapsed_seconds=continuation.elapsed_seconds,
            continuation_tokens_per_second=(
                continuation_token_count / continuation.elapsed_seconds if continuation.elapsed_seconds > 0 else 0.0
            ),
        )


def synchronize_visible_devices() -> None:
    if not torch.cuda.is_available():
        return
    for device_index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device_index)


@contextmanager
def timed_phase(accumulator: PhaseTransferAccumulator) -> Iterator[None]:
    synchronize_visible_devices()
    start_time = time.perf_counter()
    try:
        yield
    finally:
        synchronize_visible_devices()
        accumulator.elapsed_seconds += time.perf_counter() - start_time


def pin_if_cuda(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.pin_memory() if torch.cuda.is_available() else tensor


def moe_expert_modules(model: Qwen3_5MoeForCausalLM) -> tuple[torch.nn.Module, ...]:
    return tuple(layer.mlp.experts for layer in model.model.layers)


def install_expert_offload(
    model: Qwen3_5MoeForCausalLM,
    cache_size: int,
    expert_bytes: int,
) -> OffloadRuntimeController:
    state = OffloadRuntimeState(expert_bytes=expert_bytes)
    caches: list[ResidentExpertCache] = []
    for experts in moe_expert_modules(model):
        gate_up_cpu = pin_if_cuda(experts.gate_up_proj.detach().to(device="cpu", copy=True).contiguous())
        down_cpu = pin_if_cuda(experts.down_proj.detach().to(device="cpu", copy=True).contiguous())
        cache = ResidentExpertCache(
            cache_size=cache_size,
            gate_up_cpu=gate_up_cpu,
            down_cpu=down_cpu,
            device=experts.gate_up_proj.device,
            state=state,
        )
        caches.append(cache)
        experts.gate_up_proj = torch.nn.Parameter(
            torch.empty(0, device=cache.device, dtype=gate_up_cpu.dtype),
            requires_grad=False,
        )
        experts.down_proj = torch.nn.Parameter(
            torch.empty(0, device=cache.device, dtype=down_cpu.dtype),
            requires_grad=False,
        )

        def forward(
            self: torch.nn.Module,
            hidden_states: torch.Tensor,
            top_k_index: torch.Tensor,
            top_k_weights: torch.Tensor,
            *,
            _cache: ResidentExpertCache = cache,
        ) -> torch.Tensor:
            final_hidden_states = torch.zeros_like(hidden_states)
            with torch.no_grad():
                expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
                expert_mask = expert_mask.permute(2, 1, 0)
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_idx_tensor in expert_hit:
                expert_idx = int(expert_idx_tensor[0])
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[token_idx]
                gate_up_weight, down_weight = _cache.weights(expert_idx)
                gate, up = torch.nn.functional.linear(current_state, gate_up_weight).chunk(2, dim=-1)
                current_hidden_states = self.act_fn(gate) * up
                current_hidden_states = torch.nn.functional.linear(current_hidden_states, down_weight)
                current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
                final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
            return final_hidden_states

        experts.forward = MethodType(forward, experts)
    torch.cuda.empty_cache()
    return OffloadRuntimeController(
        caches=tuple(caches),
        state=state,
        expert_bytes=expert_bytes,
        cache_size=cache_size,
    )




def first_valid_prompt_and_continuation_ids(
    samples: tuple[IndexedConversationSample, ...],
    tokenizer: PreTrainedTokenizer,
    max_prompt_tokens: int,
    max_continuation_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    for indexed_sample in samples:
        prompt_ids, continuation_ids, skip_reason = prompt_and_continuation_ids(
            tokenizer=tokenizer,
            sample=indexed_sample.sample,
            max_prompt_tokens=max_prompt_tokens,
            max_continuation_tokens=max_continuation_tokens,
        )
        if skip_reason is None:
            assert prompt_ids is not None
            assert continuation_ids is not None
            return prompt_ids, continuation_ids
    return None


def warmup_variant(
    model: Qwen3_5MoeForCausalLM,
    controller: OffloadRuntimeController,
    samples: tuple[IndexedConversationSample, ...],
    tokenizer: PreTrainedTokenizer,
    max_prompt_tokens: int,
    max_continuation_tokens: int,
    routing_state: SequenceEWMARouter | None = None,
) -> None:
    first_valid_ids = first_valid_prompt_and_continuation_ids(
        samples,
        tokenizer,
        max_prompt_tokens,
        max_continuation_tokens,
    )
    if first_valid_ids is None:
        return
    prompt_ids, continuation_ids = first_valid_ids
    input_device = model.get_input_embeddings().weight.device
    controller.reset_variant()
    if routing_state is not None:
        routing_state.reset_sequence()
    with torch.inference_mode():
        outputs = model(
            input_ids=prompt_ids.to(input_device),
            use_cache=True,
            output_router_logits=False,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        for token_index in range(continuation_ids.shape[1] - 1):
            outputs = model(
                input_ids=continuation_ids[:, token_index : token_index + 1].to(input_device),
                past_key_values=past_key_values,
                use_cache=True,
                output_router_logits=False,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
    del outputs, past_key_values
    if routing_state is not None:
        routing_state.reset_sequence()
    controller.reset_variant()


def evaluate_variant(
    model: Qwen3_5MoeForCausalLM,
    controller: OffloadRuntimeController,
    samples: tuple[IndexedConversationSample, ...],
    tokenizer: PreTrainedTokenizer,
    max_prompt_tokens: int,
    max_continuation_tokens: int,
    routing_state: SequenceEWMARouter | None = None,
) -> tuple[LossAccumulator, TransferStatistics, int, int, int, int, tuple[ProcessedSample, ...]]:
    accumulator = LossAccumulator()
    prompts_processed = 0
    skipped_prompt_too_long = 0
    skipped_continuation_too_long = 0
    skipped_empty_continuation = 0
    continuation_token_count = 0
    processed_samples: list[ProcessedSample] = []
    input_device = model.get_input_embeddings().weight.device
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
        controller.reset_sequence()
        if routing_state is not None:
            routing_state.reset_sequence()
        token_losses: list[torch.Tensor] = []
        with torch.inference_mode():
            with timed_phase(controller.state.prompt):
                outputs = model(
                    input_ids=prompt_ids.to(input_device),
                    use_cache=True,
                    output_router_logits=False,
                    return_dict=True,
                )
            token_losses.append(next_token_nll(outputs.logits[:, -1, :], continuation_ids[:, 0].to(input_device)))
            past_key_values = outputs.past_key_values
            controller.switch_to_continuation()
            with timed_phase(controller.state.continuation):
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
        continuation_token_count += continuation_length
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
        controller.finalize(continuation_token_count),
        prompts_processed,
        skipped_prompt_too_long,
        skipped_continuation_too_long,
        skipped_empty_continuation,
        tuple(processed_samples),
    )


def parse_args() -> OffloadEvalConfig:
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
    parser.add_argument("--ewma-alphas", default="0.2,0.5,0.8")
    parser.add_argument("--cache-sizes", default="8,16,32")
    parser.add_argument("--device-map-mode", choices=("single-gpu", "auto", "balanced-low-0"), default=None)
    args = parser.parse_args()
    device_map_mode = args.device_map_mode
    if device_map_mode is None:
        device_map_mode = "single-gpu" if torch.cuda.is_available() else "auto"
    return OffloadEvalConfig(
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
        cache_sizes=parse_window_sizes(args.cache_sizes),
        device_map_mode=device_map_mode,
    )


def evaluate_cache_budget(
    cache_size: int,
    samples: tuple[IndexedConversationSample, ...],
    tokenizer: PreTrainedTokenizer,
    config: OffloadEvalConfig,
    expert_bytes: int,
) -> tuple[CacheBudgetResult, RuntimeInfo]:
    def run_variant(
        alpha: float,
    ) -> tuple[OffloadVariantResult, int, int, int, int, tuple[ProcessedSample, ...], RuntimeInfo]:
        model = load_model(
            config.model_repo,
            config.device_map_mode,
            auto_headroom_gib=OFFLOAD_AUTO_DEVICE_MAP_HEADROOM_GIB,
        )
        model.eval()
        controller = install_expert_offload(model, cache_size, expert_bytes)
        current_runtime = runtime_info(model, config.device_map_mode)
        routing_context = nullcontext(None) if alpha == 0.0 else patched_ewma_routing(model, alpha)
        with routing_context as routing_state:
            warmup_variant(
                model,
                controller,
                samples,
                tokenizer,
                config.max_prompt_tokens,
                config.max_continuation_tokens,
                routing_state=routing_state,
            )
            (
                accumulator,
                transfer,
                prompts_processed,
                skipped_prompt_too_long,
                skipped_continuation_too_long,
                skipped_empty_continuation,
                processed_samples,
            ) = evaluate_variant(
                model=model,
                controller=controller,
                samples=samples,
                tokenizer=tokenizer,
                max_prompt_tokens=config.max_prompt_tokens,
                max_continuation_tokens=config.max_continuation_tokens,
                routing_state=routing_state,
            )
        result = OffloadVariantResult(
            name="baseline" if alpha == 0.0 else f"ewma_{alpha:.3f}",
            alpha=alpha,
            statistics=accumulator.finalize(),
            transfer=transfer,
            delta_token_weighted_mean_continuation_nll_vs_baseline=0.0,
            delta_sequence_weighted_mean_continuation_nll_vs_baseline=0.0,
        )
        del controller, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return (
            result,
            prompts_processed,
            skipped_prompt_too_long,
            skipped_continuation_too_long,
            skipped_empty_continuation,
            processed_samples,
            current_runtime,
        )

    (
        baseline,
        prompts_processed,
        skipped_prompt_too_long,
        skipped_continuation_too_long,
        skipped_empty_continuation,
        processed_samples,
        current_runtime,
    ) = run_variant(0.0)
    ewma_variants: list[OffloadVariantResult] = []
    for alpha in config.ewma_alphas:
        (
            variant,
            alpha_prompts_processed,
            alpha_skipped_prompt_too_long,
            alpha_skipped_continuation_too_long,
            alpha_skipped_empty_continuation,
            alpha_processed_samples,
            _,
        ) = run_variant(alpha)
        if (
            alpha_prompts_processed != prompts_processed
            or alpha_skipped_prompt_too_long != skipped_prompt_too_long
            or alpha_skipped_continuation_too_long != skipped_continuation_too_long
            or alpha_skipped_empty_continuation != skipped_empty_continuation
            or alpha_processed_samples != processed_samples
        ):
            raise ValueError("EWMA offload evaluation changed the sampled prompt set.")
        ewma_variants.append(
            OffloadVariantResult(
                name=variant.name,
                alpha=variant.alpha,
                statistics=variant.statistics,
                transfer=variant.transfer,
                delta_token_weighted_mean_continuation_nll_vs_baseline=(
                    variant.statistics.token_weighted_mean_continuation_nll
                    - baseline.statistics.token_weighted_mean_continuation_nll
                ),
                delta_sequence_weighted_mean_continuation_nll_vs_baseline=(
                    variant.statistics.sequence_weighted_mean_continuation_nll
                    - baseline.statistics.sequence_weighted_mean_continuation_nll
                ),
            )
        )
    return CacheBudgetResult(
        cache_size=cache_size,
        resident_gib_total=baseline.transfer.resident_gib_total,
        baseline=baseline,
        ewma_variants=tuple(ewma_variants),
        dataset_rows_processed=len(samples),
        prompts_processed=prompts_processed,
        skipped_prompt_too_long=skipped_prompt_too_long,
        skipped_continuation_too_long=skipped_continuation_too_long,
        skipped_empty_continuation=skipped_empty_continuation,
        processed_samples=processed_samples,
    ), current_runtime


def main() -> None:
    config = parse_args()
    routing_config = model_routing_config(config.model_repo)
    rows, dataset_rows_total = load_rows(config.dataset, config.dataset_split, config.max_rows, config.seed)
    samples = indexed_conversation_samples(rows, config.seed)
    assistant_turns_total = len(samples)
    if config.max_prompts is not None:
        samples = samples[: config.max_prompts]
    tokenizer = AutoTokenizer.from_pretrained(config.model_repo)
    cache_budgets: list[CacheBudgetResult] = []
    runtime: RuntimeInfo | None = None
    for cache_size in config.cache_sizes:
        cache_budget, runtime = evaluate_cache_budget(
            cache_size=cache_size,
            samples=samples,
            tokenizer=tokenizer,
            config=config,
            expert_bytes=routing_config.expert_bytes,
        )
        cache_budgets.append(cache_budget)
    assert runtime is not None
    result = OffloadEvalResult(
        config=config,
        model=routing_config,
        runtime=runtime,
        dataset_fingerprint=dataset_fingerprint(rows),
        dataset_rows_total=dataset_rows_total,
        assistant_turns_total=assistant_turns_total,
        cache_budgets=tuple(cache_budgets),
    )
    payload = asdict(result)
    payload["config"]["output_path"] = str(config.output_path)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
