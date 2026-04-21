from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import transformers
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_5_moe import Qwen3_5MoeForCausalLM
from transformers.tokenization_utils_base import BatchEncoding

if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass(frozen=True)
class ChatMessage:
    role: MessageRole
    content: str

    def as_chat_template_dict(self) -> dict[str, str]:
        return {"role": self.role.value, "content": self.content}


@dataclass(frozen=True)
class ExperimentConfig:
    model_repo: str
    dataset: str
    dataset_split: str
    output_path: Path
    seed: int
    max_rows: int | None
    max_prompts: int | None
    max_prompt_tokens: int
    max_continuation_tokens: int
    window_sizes: tuple[int, ...]
    ewma_alphas: tuple[float, ...]
    device_map_mode: str


@dataclass(frozen=True)
class ModelRoutingConfig:
    repo: str
    revision: str
    num_layers: int
    num_experts: int
    num_active_experts: int
    expert_parameters: int
    expert_bytes: int


@dataclass(frozen=True)
class RuntimeInfo:
    transformers_version: str
    torch_version: str
    cuda_available: bool
    device_count: int
    devices: tuple[str, ...]
    model_dtype: str
    model_class: str
    device_map_mode: str
    hf_device_map: dict[str, str] | None


AUTO_DEVICE_MAP_HEADROOM_GIB = 20


@dataclass(frozen=True)
class WindowStatistics:
    window_size: int
    num_windows: int
    sequence_count_with_windows: int
    window_weighted_mean_distinct_experts_per_layer: list[float]
    window_weighted_mean_distinct_experts_overall: float
    window_weighted_mean_distinct_experts_fraction_overall: float
    sequence_weighted_mean_distinct_experts_per_layer: list[float]
    sequence_weighted_mean_distinct_experts_overall: float
    sequence_weighted_mean_distinct_experts_fraction_overall: float
    sequence_weighted_mean_distinct_experts_overall_std: float
    sequence_weighted_mean_distinct_experts_overall_sem: float
    sequence_weighted_mean_distinct_experts_overall_ci95: float
    random_baseline_distinct_experts: float
    random_baseline_fraction: float
    window_weighted_observed_to_random_ratio: float
    sequence_weighted_observed_to_random_ratio: float
    window_weighted_mean_distinct_layer_expert_pairs: float
    window_weighted_mean_distinct_layer_expert_pair_fraction: float
    sequence_weighted_mean_distinct_layer_expert_pairs: float
    sequence_weighted_mean_distinct_layer_expert_pair_fraction: float
    oracle_cache_hit_rates: tuple[CacheHitRateStatistics, ...]


@dataclass(frozen=True)
class CacheHitRateStatistics:
    cache_size: int
    cache_fraction: float
    window_weighted_hit_rate: float
    sequence_weighted_hit_rate: float
    sequence_weighted_hit_rate_std: float
    sequence_weighted_hit_rate_sem: float
    sequence_weighted_hit_rate_ci95: float


@dataclass(frozen=True)
class ResidentBudgetStatistics:
    cache_size: int
    cache_fraction: float
    resident_experts_total: int
    resident_bytes_total: int
    resident_gib_total: float
    token_weighted_hit_rate: float
    sequence_weighted_hit_rate: float
    sequence_weighted_hit_rate_ci95: float
    token_weighted_expert_loads_per_token: float
    sequence_weighted_expert_loads_per_token: float
    sequence_weighted_expert_loads_per_token_ci95: float
    token_weighted_transfer_bytes_per_token: float
    sequence_weighted_transfer_bytes_per_token: float
    sequence_weighted_transfer_bytes_per_token_ci95: float


@dataclass(frozen=True)
class TopKAgreementStatistics:
    token_weighted_mean_retained_fraction_per_layer: list[float]
    token_weighted_mean_retained_fraction_overall: float
    sequence_weighted_mean_retained_fraction_per_layer: list[float]
    sequence_weighted_mean_retained_fraction_overall: float
    sequence_weighted_mean_retained_fraction_overall_std: float
    sequence_weighted_mean_retained_fraction_overall_sem: float
    sequence_weighted_mean_retained_fraction_overall_ci95: float
    token_weighted_exact_match_rate: float
    sequence_weighted_exact_match_rate: float
    sequence_weighted_exact_match_rate_std: float
    sequence_weighted_exact_match_rate_sem: float
    sequence_weighted_exact_match_rate_ci95: float


@dataclass(frozen=True)
class PhaseStatistics:
    sequence_count: int
    token_count: int
    windows: tuple[WindowStatistics, ...]
    resident_budgets: tuple[ResidentBudgetStatistics, ...]


@dataclass(frozen=True)
class EwmaStatistics:
    alpha: float
    prompt_statistics: PhaseStatistics
    continuation_statistics: PhaseStatistics
    prompt_agreement: TopKAgreementStatistics
    continuation_agreement: TopKAgreementStatistics


@dataclass(frozen=True)
class RoutingAnalysisResult:
    config: ExperimentConfig
    model: ModelRoutingConfig
    runtime: RuntimeInfo
    dataset_fingerprint: str
    dataset_rows_total: int
    assistant_turns_total: int
    prompt_statistics: PhaseStatistics
    continuation_statistics: PhaseStatistics
    ewma_statistics: tuple[EwmaStatistics, ...]
    dataset_rows_processed: int
    prompts_processed: int
    skipped_prompt_too_long: int
    skipped_continuation_too_long: int
    skipped_empty_continuation: int
    processed_samples: tuple[ProcessedSample, ...]


@dataclass
class WindowAccumulator:
    window_size: int
    num_layers: int
    num_experts: int
    num_active_experts: int
    cache_sizes: tuple[int, ...]
    num_windows: int = 0
    sum_distinct_per_layer: np.ndarray | None = None
    sum_distinct_layer_expert_pairs: float = 0.0
    sequence_mean_distinct_per_layer: list[np.ndarray] | None = None
    sequence_mean_distinct_layer_expert_pairs: list[float] | None = None
    sum_cache_hit_rates: np.ndarray | None = None
    sequence_mean_cache_hit_rates: list[np.ndarray] | None = None

    def __post_init__(self) -> None:
        self.sum_distinct_per_layer = np.zeros(self.num_layers, dtype=np.float64)
        self.sequence_mean_distinct_per_layer = []
        self.sequence_mean_distinct_layer_expert_pairs = []
        self.sum_cache_hit_rates = np.zeros(len(self.cache_sizes), dtype=np.float64)
        self.sequence_mean_cache_hit_rates = []

    def update(self, active_experts: np.ndarray) -> None:
        distinct_per_layer, num_windows = distinct_experts_in_windows(
            active_experts,
            self.window_size,
            self.num_experts,
        )
        if num_windows == 0:
            return
        assert self.sum_distinct_per_layer is not None
        self.sum_distinct_per_layer += distinct_per_layer.sum(axis=1)
        self.sum_distinct_layer_expert_pairs += float(distinct_per_layer.sum())
        self.num_windows += num_windows
        assert self.sequence_mean_distinct_per_layer is not None
        assert self.sequence_mean_distinct_layer_expert_pairs is not None
        sequence_mean_distinct_per_layer = distinct_per_layer.mean(axis=1)
        self.sequence_mean_distinct_per_layer.append(sequence_mean_distinct_per_layer)
        self.sequence_mean_distinct_layer_expert_pairs.append(float(sequence_mean_distinct_per_layer.sum()))
        cache_hit_rates = oracle_cache_hit_rates_in_windows(
            active_experts,
            self.window_size,
            self.num_experts,
            self.num_active_experts,
            self.cache_sizes,
        )
        assert self.sum_cache_hit_rates is not None
        assert self.sequence_mean_cache_hit_rates is not None
        self.sum_cache_hit_rates += cache_hit_rates.sum(axis=1)
        self.sequence_mean_cache_hit_rates.append(cache_hit_rates.mean(axis=1))

    def finalize(self) -> WindowStatistics:
        assert self.sum_distinct_per_layer is not None
        window_weighted_mean_distinct_per_layer = (
            np.zeros(self.num_layers, dtype=np.float64)
            if self.num_windows == 0
            else self.sum_distinct_per_layer / self.num_windows
        )
        window_weighted_mean_distinct_layer_expert_pairs = (
            0.0 if self.num_windows == 0 else self.sum_distinct_layer_expert_pairs / self.num_windows
        )
        window_weighted_mean_distinct_overall = float(window_weighted_mean_distinct_per_layer.mean())
        assert self.sequence_mean_distinct_per_layer is not None
        assert self.sequence_mean_distinct_layer_expert_pairs is not None
        if self.sequence_mean_distinct_per_layer:
            sequence_weighted_mean_distinct_per_layer = np.stack(self.sequence_mean_distinct_per_layer, axis=0).mean(
                axis=0
            )
            sequence_overall_values = np.asarray(
                [layer_mean.mean() for layer_mean in self.sequence_mean_distinct_per_layer],
                dtype=np.float64,
            )
            sequence_weighted_mean_distinct_layer_expert_pairs = float(
                np.mean(self.sequence_mean_distinct_layer_expert_pairs)
            )
            sequence_weighted_mean_distinct_overall = float(sequence_overall_values.mean())
            sequence_weighted_mean_distinct_overall_std = float(sequence_overall_values.std(ddof=0))
            sequence_weighted_mean_distinct_overall_sem = float(
                sequence_weighted_mean_distinct_overall_std / np.sqrt(len(sequence_overall_values))
            )
        else:
            sequence_weighted_mean_distinct_per_layer = np.zeros(self.num_layers, dtype=np.float64)
            sequence_weighted_mean_distinct_layer_expert_pairs = 0.0
            sequence_weighted_mean_distinct_overall = 0.0
            sequence_weighted_mean_distinct_overall_std = 0.0
            sequence_weighted_mean_distinct_overall_sem = 0.0
        assert self.sum_cache_hit_rates is not None
        assert self.sequence_mean_cache_hit_rates is not None
        oracle_cache_hit_rates = []
        if self.sequence_mean_cache_hit_rates:
            sequence_mean_cache_hit_rates = np.stack(self.sequence_mean_cache_hit_rates, axis=0)
            window_weighted_cache_hit_rates = self.sum_cache_hit_rates / self.num_windows
            sequence_weighted_cache_hit_rates = sequence_mean_cache_hit_rates.mean(axis=0)
            sequence_weighted_cache_hit_rates_std = sequence_mean_cache_hit_rates.std(axis=0, ddof=0)
            sequence_weighted_cache_hit_rates_sem = sequence_weighted_cache_hit_rates_std / np.sqrt(
                sequence_mean_cache_hit_rates.shape[0]
            )
        else:
            window_weighted_cache_hit_rates = np.zeros(len(self.cache_sizes), dtype=np.float64)
            sequence_weighted_cache_hit_rates = np.zeros(len(self.cache_sizes), dtype=np.float64)
            sequence_weighted_cache_hit_rates_std = np.zeros(len(self.cache_sizes), dtype=np.float64)
            sequence_weighted_cache_hit_rates_sem = np.zeros(len(self.cache_sizes), dtype=np.float64)
        for index, cache_size in enumerate(self.cache_sizes):
            oracle_cache_hit_rates.append(
                CacheHitRateStatistics(
                    cache_size=cache_size,
                    cache_fraction=cache_size / self.num_experts,
                    window_weighted_hit_rate=float(window_weighted_cache_hit_rates[index]),
                    sequence_weighted_hit_rate=float(sequence_weighted_cache_hit_rates[index]),
                    sequence_weighted_hit_rate_std=float(sequence_weighted_cache_hit_rates_std[index]),
                    sequence_weighted_hit_rate_sem=float(sequence_weighted_cache_hit_rates_sem[index]),
                    sequence_weighted_hit_rate_ci95=float(1.96 * sequence_weighted_cache_hit_rates_sem[index]),
                )
            )
        random_baseline_distinct_experts = random_distinct_expert_baseline(
            self.window_size,
            self.num_experts,
            self.num_active_experts,
        )
        return WindowStatistics(
            window_size=self.window_size,
            num_windows=self.num_windows,
            sequence_count_with_windows=len(self.sequence_mean_distinct_per_layer),
            window_weighted_mean_distinct_experts_per_layer=window_weighted_mean_distinct_per_layer.tolist(),
            window_weighted_mean_distinct_experts_overall=window_weighted_mean_distinct_overall,
            window_weighted_mean_distinct_experts_fraction_overall=(
                window_weighted_mean_distinct_overall / self.num_experts
            ),
            sequence_weighted_mean_distinct_experts_per_layer=sequence_weighted_mean_distinct_per_layer.tolist(),
            sequence_weighted_mean_distinct_experts_overall=sequence_weighted_mean_distinct_overall,
            sequence_weighted_mean_distinct_experts_fraction_overall=(
                sequence_weighted_mean_distinct_overall / self.num_experts
            ),
            sequence_weighted_mean_distinct_experts_overall_std=sequence_weighted_mean_distinct_overall_std,
            sequence_weighted_mean_distinct_experts_overall_sem=sequence_weighted_mean_distinct_overall_sem,
            sequence_weighted_mean_distinct_experts_overall_ci95=(1.96 * sequence_weighted_mean_distinct_overall_sem),
            random_baseline_distinct_experts=random_baseline_distinct_experts,
            random_baseline_fraction=random_baseline_distinct_experts / self.num_experts,
            window_weighted_observed_to_random_ratio=(
                window_weighted_mean_distinct_overall / random_baseline_distinct_experts
                if random_baseline_distinct_experts
                else 0.0
            ),
            sequence_weighted_observed_to_random_ratio=(
                sequence_weighted_mean_distinct_overall / random_baseline_distinct_experts
                if random_baseline_distinct_experts
                else 0.0
            ),
            window_weighted_mean_distinct_layer_expert_pairs=window_weighted_mean_distinct_layer_expert_pairs,
            window_weighted_mean_distinct_layer_expert_pair_fraction=(
                window_weighted_mean_distinct_layer_expert_pairs / (self.num_layers * self.num_experts)
            ),
            sequence_weighted_mean_distinct_layer_expert_pairs=(sequence_weighted_mean_distinct_layer_expert_pairs),
            sequence_weighted_mean_distinct_layer_expert_pair_fraction=(
                sequence_weighted_mean_distinct_layer_expert_pairs / (self.num_layers * self.num_experts)
            ),
            oracle_cache_hit_rates=tuple(oracle_cache_hit_rates),
        )


@dataclass
class ResidentBudgetAccumulator:
    cache_size: int
    num_layers: int
    num_experts: int
    num_active_experts: int
    expert_bytes: int
    sequence_count: int = 0
    token_count: int = 0
    hit_count: int = 0
    miss_count: int = 0
    sequence_hit_rates: list[float] | None = None
    sequence_expert_loads_per_token: list[float] | None = None

    def __post_init__(self) -> None:
        self.sequence_hit_rates = []
        self.sequence_expert_loads_per_token = []

    def update(self, hit_count: int, miss_count: int, token_count: int) -> None:
        if token_count <= 0:
            raise ValueError(f"token_count must be positive, got {token_count}.")
        if hit_count < 0 or miss_count < 0:
            raise ValueError(f"hit_count and miss_count must be non-negative, got {hit_count} and {miss_count}.")
        denominator = token_count * self.num_layers * self.num_active_experts
        if hit_count + miss_count != denominator:
            raise ValueError(
                "Expected hit_count + miss_count to equal tokens * layers * top_k, got "
                f"{hit_count + miss_count} and {denominator}."
            )
        self.sequence_count += 1
        self.token_count += token_count
        self.hit_count += hit_count
        self.miss_count += miss_count
        assert self.sequence_hit_rates is not None
        assert self.sequence_expert_loads_per_token is not None
        self.sequence_hit_rates.append(hit_count / denominator)
        self.sequence_expert_loads_per_token.append(miss_count / token_count)

    def finalize(self) -> ResidentBudgetStatistics:
        resident_experts_total = self.cache_size * self.num_layers
        resident_bytes_total = resident_experts_total * self.expert_bytes
        if self.token_count == 0:
            token_weighted_hit_rate = 0.0
            token_weighted_expert_loads_per_token = 0.0
        else:
            token_weighted_hit_rate = self.hit_count / (self.token_count * self.num_layers * self.num_active_experts)
            token_weighted_expert_loads_per_token = self.miss_count / self.token_count
        assert self.sequence_hit_rates is not None
        assert self.sequence_expert_loads_per_token is not None
        if self.sequence_hit_rates:
            sequence_hit_rates = np.asarray(self.sequence_hit_rates, dtype=np.float64)
            sequence_expert_loads_per_token = np.asarray(self.sequence_expert_loads_per_token, dtype=np.float64)
            sequence_weighted_hit_rate = float(sequence_hit_rates.mean())
            sequence_weighted_hit_rate_ci95 = float(
                1.96 * sequence_hit_rates.std(ddof=0) / np.sqrt(len(sequence_hit_rates))
            )
            sequence_weighted_expert_loads_per_token = float(sequence_expert_loads_per_token.mean())
            sequence_weighted_expert_loads_per_token_ci95 = float(
                1.96 * sequence_expert_loads_per_token.std(ddof=0) / np.sqrt(len(sequence_expert_loads_per_token))
            )
        else:
            sequence_weighted_hit_rate = 0.0
            sequence_weighted_hit_rate_ci95 = 0.0
            sequence_weighted_expert_loads_per_token = 0.0
            sequence_weighted_expert_loads_per_token_ci95 = 0.0
        return ResidentBudgetStatistics(
            cache_size=self.cache_size,
            cache_fraction=self.cache_size / self.num_experts,
            resident_experts_total=resident_experts_total,
            resident_bytes_total=resident_bytes_total,
            resident_gib_total=resident_bytes_total / (1024**3),
            token_weighted_hit_rate=token_weighted_hit_rate,
            sequence_weighted_hit_rate=sequence_weighted_hit_rate,
            sequence_weighted_hit_rate_ci95=sequence_weighted_hit_rate_ci95,
            token_weighted_expert_loads_per_token=token_weighted_expert_loads_per_token,
            sequence_weighted_expert_loads_per_token=sequence_weighted_expert_loads_per_token,
            sequence_weighted_expert_loads_per_token_ci95=sequence_weighted_expert_loads_per_token_ci95,
            token_weighted_transfer_bytes_per_token=token_weighted_expert_loads_per_token * self.expert_bytes,
            sequence_weighted_transfer_bytes_per_token=sequence_weighted_expert_loads_per_token * self.expert_bytes,
            sequence_weighted_transfer_bytes_per_token_ci95=(
                sequence_weighted_expert_loads_per_token_ci95 * self.expert_bytes
            ),
        )


@dataclass
class PhaseAccumulator:
    window_sizes: tuple[int, ...]
    num_layers: int
    num_experts: int
    num_active_experts: int
    cache_sizes: tuple[int, ...]
    expert_bytes: int
    sequence_count: int = 0
    token_count: int = 0
    windows: tuple[WindowAccumulator, ...] = ()
    resident_budgets: tuple[ResidentBudgetAccumulator, ...] = ()

    def __post_init__(self) -> None:
        self.windows = tuple(
            WindowAccumulator(
                window_size=window_size,
                num_layers=self.num_layers,
                num_experts=self.num_experts,
                num_active_experts=self.num_active_experts,
                cache_sizes=self.cache_sizes,
            )
            for window_size in self.window_sizes
        )
        self.resident_budgets = tuple(
            ResidentBudgetAccumulator(
                cache_size=cache_size,
                num_layers=self.num_layers,
                num_experts=self.num_experts,
                num_active_experts=self.num_active_experts,
                expert_bytes=self.expert_bytes,
            )
            for cache_size in self.cache_sizes
        )

    def update(self, active_experts: np.ndarray) -> None:
        if active_experts.size == 0:
            return
        self.sequence_count += 1
        self.token_count += int(active_experts.shape[1])
        for window in self.windows:
            window.update(active_experts)

    def update_resident_budgets(self, hits: np.ndarray, misses: np.ndarray, token_count: int) -> None:
        if token_count == 0:
            return
        if hits.shape != misses.shape:
            raise ValueError(f"Expected hits and misses to have the same shape, got {hits.shape} and {misses.shape}.")
        if hits.shape != (len(self.resident_budgets),):
            raise ValueError(
                "Expected one hit/miss count per resident budget, got "
                f"{hits.shape} for {len(self.resident_budgets)} budgets."
            )
        for index, budget in enumerate(self.resident_budgets):
            budget.update(int(hits[index]), int(misses[index]), token_count)

    def finalize(self) -> PhaseStatistics:
        return PhaseStatistics(
            sequence_count=self.sequence_count,
            token_count=self.token_count,
            windows=tuple(window.finalize() for window in self.windows),
            resident_budgets=tuple(budget.finalize() for budget in self.resident_budgets),
        )


@dataclass
class AgreementAccumulator:
    num_layers: int
    token_count: int = 0
    sum_retained_fraction_per_layer: np.ndarray | None = None
    sequence_mean_retained_fraction_per_layer: list[np.ndarray] | None = None
    sum_exact_matches: int = 0
    sequence_mean_exact_match_rate: list[float] | None = None

    def __post_init__(self) -> None:
        self.sum_retained_fraction_per_layer = np.zeros(self.num_layers, dtype=np.float64)
        self.sequence_mean_retained_fraction_per_layer = []
        self.sequence_mean_exact_match_rate = []

    def update(self, baseline_active_experts: np.ndarray, comparison_active_experts: np.ndarray) -> None:
        if baseline_active_experts.shape != comparison_active_experts.shape:
            raise ValueError(
                "Expected baseline and comparison active experts to have the same shape, got "
                f"{baseline_active_experts.shape} and {comparison_active_experts.shape}."
            )
        if baseline_active_experts.size == 0:
            return
        overlap_counts = topk_overlap_counts(baseline_active_experts, comparison_active_experts)
        num_tokens = overlap_counts.shape[1]
        retained_fraction = overlap_counts.astype(np.float64, copy=False) / baseline_active_experts.shape[2]
        exact_match = (overlap_counts == baseline_active_experts.shape[2]).astype(np.float64, copy=False)
        self.token_count += num_tokens
        assert self.sum_retained_fraction_per_layer is not None
        self.sum_retained_fraction_per_layer += retained_fraction.sum(axis=1)
        self.sum_exact_matches += int(exact_match.sum())
        assert self.sequence_mean_retained_fraction_per_layer is not None
        assert self.sequence_mean_exact_match_rate is not None
        self.sequence_mean_retained_fraction_per_layer.append(retained_fraction.mean(axis=1))
        self.sequence_mean_exact_match_rate.append(float(exact_match.mean()))

    def finalize(self) -> TopKAgreementStatistics:
        assert self.sum_retained_fraction_per_layer is not None
        assert self.sequence_mean_retained_fraction_per_layer is not None
        assert self.sequence_mean_exact_match_rate is not None
        if self.token_count == 0:
            token_weighted_mean_retained_fraction_per_layer = np.zeros(self.num_layers, dtype=np.float64)
            token_weighted_exact_match_rate = 0.0
        else:
            token_weighted_mean_retained_fraction_per_layer = self.sum_retained_fraction_per_layer / self.token_count
            token_weighted_exact_match_rate = self.sum_exact_matches / (self.token_count * self.num_layers)
        if self.sequence_mean_retained_fraction_per_layer:
            sequence_weighted_mean_retained_fraction_per_layer = np.stack(
                self.sequence_mean_retained_fraction_per_layer, axis=0
            ).mean(axis=0)
            sequence_overall_retained_fraction = np.asarray(
                [value.mean() for value in self.sequence_mean_retained_fraction_per_layer],
                dtype=np.float64,
            )
            sequence_overall_exact_match_rate = np.asarray(self.sequence_mean_exact_match_rate, dtype=np.float64)
            sequence_weighted_mean_retained_fraction_overall = float(sequence_overall_retained_fraction.mean())
            sequence_weighted_mean_retained_fraction_overall_std = float(
                sequence_overall_retained_fraction.std(ddof=0)
            )
            sequence_weighted_mean_retained_fraction_overall_sem = float(
                sequence_weighted_mean_retained_fraction_overall_std / np.sqrt(len(sequence_overall_retained_fraction))
            )
            sequence_weighted_exact_match_rate = float(sequence_overall_exact_match_rate.mean())
            sequence_weighted_exact_match_rate_std = float(sequence_overall_exact_match_rate.std(ddof=0))
            sequence_weighted_exact_match_rate_sem = float(
                sequence_weighted_exact_match_rate_std / np.sqrt(len(sequence_overall_exact_match_rate))
            )
        else:
            sequence_weighted_mean_retained_fraction_per_layer = np.zeros(self.num_layers, dtype=np.float64)
            sequence_weighted_mean_retained_fraction_overall = 0.0
            sequence_weighted_mean_retained_fraction_overall_std = 0.0
            sequence_weighted_mean_retained_fraction_overall_sem = 0.0
            sequence_weighted_exact_match_rate = 0.0
            sequence_weighted_exact_match_rate_std = 0.0
            sequence_weighted_exact_match_rate_sem = 0.0
        return TopKAgreementStatistics(
            token_weighted_mean_retained_fraction_per_layer=token_weighted_mean_retained_fraction_per_layer.tolist(),
            token_weighted_mean_retained_fraction_overall=float(
                token_weighted_mean_retained_fraction_per_layer.mean()
            ),
            sequence_weighted_mean_retained_fraction_per_layer=(
                sequence_weighted_mean_retained_fraction_per_layer.tolist()
            ),
            sequence_weighted_mean_retained_fraction_overall=sequence_weighted_mean_retained_fraction_overall,
            sequence_weighted_mean_retained_fraction_overall_std=(
                sequence_weighted_mean_retained_fraction_overall_std
            ),
            sequence_weighted_mean_retained_fraction_overall_sem=(
                sequence_weighted_mean_retained_fraction_overall_sem
            ),
            sequence_weighted_mean_retained_fraction_overall_ci95=(
                1.96 * sequence_weighted_mean_retained_fraction_overall_sem
            ),
            token_weighted_exact_match_rate=token_weighted_exact_match_rate,
            sequence_weighted_exact_match_rate=sequence_weighted_exact_match_rate,
            sequence_weighted_exact_match_rate_std=sequence_weighted_exact_match_rate_std,
            sequence_weighted_exact_match_rate_sem=sequence_weighted_exact_match_rate_sem,
            sequence_weighted_exact_match_rate_ci95=1.96 * sequence_weighted_exact_match_rate_sem,
        )


@dataclass
class EwmaAccumulator:
    alpha: float
    prompt: PhaseAccumulator
    continuation: PhaseAccumulator
    prompt_agreement: AgreementAccumulator
    continuation_agreement: AgreementAccumulator

    def finalize(self) -> EwmaStatistics:
        return EwmaStatistics(
            alpha=self.alpha,
            prompt_statistics=self.prompt.finalize(),
            continuation_statistics=self.continuation.finalize(),
            prompt_agreement=self.prompt_agreement.finalize(),
            continuation_agreement=self.continuation_agreement.finalize(),
        )


@dataclass(frozen=True)
class ConversationSample:
    prompt: Conversation
    continuation: ChatMessage


@dataclass(frozen=True)
class IndexedConversationSample:
    row_id: int
    conversation_index: int
    assistant_turn_index: int
    sample: ConversationSample


@dataclass(frozen=True)
class ProcessedSample:
    row_id: int
    conversation_index: int
    assistant_turn_index: int
    prompt_tokens: int
    continuation_tokens: int


def parse_window_sizes(text: str) -> tuple[int, ...]:
    window_sizes = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not window_sizes:
        raise ValueError("window_sizes must not be empty.")
    if any(window_size <= 0 for window_size in window_sizes):
        raise ValueError(f"window_sizes must be positive, got {window_sizes}.")
    if tuple(sorted(window_sizes)) != window_sizes:
        raise ValueError(f"window_sizes must be sorted ascending, got {window_sizes}.")
    return window_sizes


def parse_ewma_alphas(text: str) -> tuple[float, ...]:
    if not text.strip():
        return ()
    alphas = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if any(alpha <= 0.0 or alpha >= 1.0 for alpha in alphas):
        raise ValueError(f"ewma_alphas must be strictly between 0 and 1, got {alphas}.")
    if tuple(sorted(alphas)) != alphas:
        raise ValueError(f"ewma_alphas must be sorted ascending, got {alphas}.")
    return alphas


def normalize_role(role: str) -> MessageRole:
    match role:
        case "user" | "human":
            return MessageRole.USER
        case "assistant" | "gpt":
            return MessageRole.ASSISTANT
        case "system" | "developer":
            return MessageRole.SYSTEM
        case other:
            raise ValueError(f"Unsupported message role: {other}.")


def normalize_message(message: dict[str, Any]) -> ChatMessage:
    if "role" in message and "content" in message:
        role = normalize_role(message["role"])
        content = message["content"]
    elif "from" in message and "value" in message:
        role = normalize_role(message["from"])
        content = message["value"]
    else:
        raise ValueError(f"Unsupported message shape: {sorted(message)}.")
    if not isinstance(content, str):
        raise TypeError(f"Message content must be text, got {type(content).__name__}.")
    return ChatMessage(role=role, content=content)


Conversation = tuple[ChatMessage, ...]


def row_conversations(row: dict[str, Any]) -> tuple[Conversation, ...]:
    if "conversation" in row and row["conversation"] is not None:
        return (tuple(normalize_message(message) for message in row["conversation"]),)
    if "conversations" in row and row["conversations"] is not None:
        return (tuple(normalize_message(message) for message in row["conversations"]),)
    if "messages" in row and row["messages"] is not None:
        return (tuple(normalize_message(message) for message in row["messages"]),)
    if "conversation_a" in row and "conversation_b" in row:
        return (
            tuple(normalize_message(message) for message in row["conversation_a"]),
            tuple(normalize_message(message) for message in row["conversation_b"]),
        )
    raise ValueError(f"Unsupported dataset columns: {sorted(row)}.")


def conversation_samples(messages: Conversation) -> tuple[ConversationSample, ...]:
    samples: list[ConversationSample] = []
    current: list[ChatMessage] = []
    for message in messages:
        current.append(message)
        if message.role is not MessageRole.ASSISTANT:
            continue
        prompt = tuple(current[:-1])
        if not prompt or prompt[-1].role is not MessageRole.USER:
            continue
        samples.append(ConversationSample(prompt=prompt, continuation=message))
    return tuple(samples)


def indexed_conversation_samples(rows: Dataset, seed: int) -> tuple[IndexedConversationSample, ...]:
    indexed_samples: list[IndexedConversationSample] = []
    for row in rows:
        row_id = int(row["__row_id__"])
        for conversation_index, conversation in enumerate(row_conversations(row)):
            for assistant_turn_index, sample in enumerate(conversation_samples(conversation)):
                indexed_samples.append(
                    IndexedConversationSample(
                        row_id=row_id,
                        conversation_index=conversation_index,
                        assistant_turn_index=assistant_turn_index,
                        sample=sample,
                    )
                )
    if not indexed_samples:
        return ()
    order = np.random.default_rng(seed).permutation(len(indexed_samples))
    return tuple(indexed_samples[index] for index in order)


def local_dataset_files(dataset_path: Path) -> tuple[str, list[str]]:
    if dataset_path.is_file():
        files = [str(dataset_path)]
    else:
        files = sorted(
            str(path) for suffix in ("*.parquet", "*.jsonl", "*.json") for path in dataset_path.rglob(suffix)
        )
    if not files:
        raise FileNotFoundError(f"No parquet/json/jsonl files found under {dataset_path}.")
    suffixes = {Path(file_path).suffix.lower() for file_path in files}
    if suffixes == {".parquet"}:
        return "parquet", files
    if suffixes <= {".json", ".jsonl"}:
        return "json", files
    raise ValueError(f"Mixed local dataset formats are not supported: {sorted(suffixes)}.")


def load_rows(dataset: str, dataset_split: str, max_rows: int | None, seed: int) -> tuple[Dataset, int]:
    local_path = Path(dataset)
    if local_path.exists():
        loader_name, data_files = local_dataset_files(local_path)
        loaded = load_dataset(loader_name, data_files=data_files, split="train")
    else:
        loaded = load_dataset(dataset, split=dataset_split)
    if isinstance(loaded, DatasetDict):
        raise TypeError(f"Expected a single split dataset, got DatasetDict with splits {sorted(loaded.keys())}.")
    total_rows = len(loaded)
    if "__row_id__" in loaded.column_names:
        raise ValueError("Dataset already has a reserved __row_id__ column.")
    loaded = loaded.add_column("__row_id__", list(range(total_rows)))
    if max_rows is None or total_rows <= max_rows:
        return loaded, total_rows
    sampled = loaded.shuffle(seed=seed).select(range(max_rows))
    return sampled, total_rows


def dataset_fingerprint(rows: Dataset) -> str:
    fingerprint = getattr(rows, "_fingerprint", None)
    if not isinstance(fingerprint, str):
        raise TypeError("Expected Dataset to expose a string fingerprint.")
    return fingerprint


def chat_template_text(
    tokenizer: PreTrainedTokenizer,
    messages: Conversation,
) -> str:
    return tokenizer.apply_chat_template(
        [message.as_chat_template_dict() for message in messages],
        tokenize=False,
        add_generation_prompt=False,
    )


def extract_input_ids(input_ids: torch.Tensor | BatchEncoding) -> torch.Tensor:
    if isinstance(input_ids, BatchEncoding):
        input_ids = input_ids["input_ids"]
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError(f"Expected Tensor from tokenizer, got {type(input_ids).__name__}.")
    if input_ids.ndim != 2:
        raise ValueError(f"Expected input_ids shape [batch, tokens], got {tuple(input_ids.shape)}.")
    return input_ids


def text_input_ids(
    tokenizer: PreTrainedTokenizer,
    text: str,
) -> torch.Tensor:
    return extract_input_ids(
        tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )
    )


def prompt_and_continuation_ids(
    tokenizer: PreTrainedTokenizer,
    sample: ConversationSample,
    max_prompt_tokens: int,
    max_continuation_tokens: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, str | None]:
    prompt_text = tokenizer.apply_chat_template(
        [message.as_chat_template_dict() for message in sample.prompt],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = text_input_ids(tokenizer, prompt_text)
    if prompt_ids.shape[1] > max_prompt_tokens:
        return None, None, "prompt_too_long"
    full_text = chat_template_text(tokenizer, sample.prompt + (sample.continuation,))
    if not full_text.startswith(prompt_text):
        raise ValueError("Prompt text must be a prefix of the prompt+continuation text.")
    continuation_text = full_text[len(prompt_text) :]
    continuation_ids = text_input_ids(tokenizer, continuation_text)
    if continuation_ids.shape[1] == 0:
        return None, None, "empty_continuation"
    if continuation_ids.shape[1] > max_continuation_tokens:
        return None, None, "continuation_too_long"
    return prompt_ids, continuation_ids, None


def make_attention_mask(input_ids: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(input_ids, dtype=torch.long)


def topk_active_experts(router_logits: tuple[torch.Tensor, ...], num_active_experts: int) -> np.ndarray:
    active_per_layer = []
    for layer_logits in router_logits:
        if layer_logits.ndim != 2:
            raise ValueError(f"Expected router logits shape [tokens, experts], got {tuple(layer_logits.shape)}.")
        layer_active = torch.topk(layer_logits, k=num_active_experts, dim=-1).indices
        active_per_layer.append(layer_active.to("cpu", dtype=torch.int16).numpy())
    return np.stack(active_per_layer, axis=0)


def topk_overlap_counts(reference_active_experts: np.ndarray, comparison_active_experts: np.ndarray) -> np.ndarray:
    if reference_active_experts.shape != comparison_active_experts.shape:
        raise ValueError(
            "Expected reference and comparison active experts to have the same shape, got "
            f"{reference_active_experts.shape} and {comparison_active_experts.shape}."
        )
    return (
        (reference_active_experts[..., :, None] == comparison_active_experts[..., None, :]).any(axis=-1).sum(axis=-1)
    ).astype(np.int16, copy=False)


def ewma_topk_active_experts(
    router_logits: tuple[torch.Tensor, ...],
    num_active_experts: int,
    alpha: float,
) -> np.ndarray:
    active_per_layer = []
    for layer_logits in router_logits:
        if layer_logits.ndim != 2:
            raise ValueError(f"Expected router logits shape [tokens, experts], got {tuple(layer_logits.shape)}.")
        smoothed_logits = layer_logits.to("cpu", dtype=torch.float32).contiguous()
        for token_index in range(1, smoothed_logits.shape[0]):
            smoothed_logits[token_index] = (
                alpha * smoothed_logits[token_index] + (1.0 - alpha) * smoothed_logits[token_index - 1]
            )
        layer_active = torch.topk(smoothed_logits, k=num_active_experts, dim=-1).indices
        active_per_layer.append(layer_active.to(dtype=torch.int16).numpy())
    return np.stack(active_per_layer, axis=0)


def active_expert_mask(active_experts: np.ndarray, num_experts: int) -> np.ndarray:
    num_layers, num_tokens, top_k = active_experts.shape
    mask = np.zeros((num_layers, num_tokens, num_experts), dtype=np.bool_)
    for layer_index in range(num_layers):
        for token_index in range(num_tokens):
            mask[layer_index, token_index, active_experts[layer_index, token_index, :top_k]] = True
    return mask


def random_distinct_expert_baseline(window_size: int, num_experts: int, num_active_experts: int) -> float:
    return num_experts * (1.0 - (1.0 - num_active_experts / num_experts) ** window_size)


def distinct_experts_in_windows(
    active_experts: np.ndarray,
    window_size: int,
    num_experts: int,
) -> tuple[np.ndarray, int]:
    if active_experts.ndim != 3:
        raise ValueError(f"active_experts must have shape [layers, tokens, top_k], got {active_experts.shape}.")
    _, num_tokens, _ = active_experts.shape
    if num_tokens < window_size:
        return np.zeros((active_experts.shape[0], 0), dtype=np.float64), 0
    mask = active_expert_mask(active_experts, num_experts).astype(np.int16, copy=False)
    prefix = np.pad(mask.cumsum(axis=1, dtype=np.int32), ((0, 0), (1, 0), (0, 0)))
    window_hits = prefix[:, window_size:, :] - prefix[:, :-window_size, :]
    distinct_per_layer = (window_hits > 0).sum(axis=-1, dtype=np.int32)
    return distinct_per_layer.astype(np.float64, copy=False), distinct_per_layer.shape[1]


def oracle_cache_hit_rates_in_windows(
    active_experts: np.ndarray,
    window_size: int,
    num_experts: int,
    num_active_experts: int,
    cache_sizes: tuple[int, ...],
) -> np.ndarray:
    _, num_tokens, _ = active_experts.shape
    if num_tokens < window_size:
        return np.zeros((len(cache_sizes), 0), dtype=np.float64)
    mask = active_expert_mask(active_experts, num_experts).astype(np.int16, copy=False)
    prefix = np.pad(mask.cumsum(axis=1, dtype=np.int32), ((0, 0), (1, 0), (0, 0)))
    window_counts = prefix[:, window_size:, :] - prefix[:, :-window_size, :]
    sorted_counts = np.sort(window_counts, axis=-1)[..., ::-1]
    cumulative_counts = np.cumsum(sorted_counts, axis=-1, dtype=np.int32)
    denominator = float(window_size * num_active_experts)
    return np.stack(
        [cumulative_counts[..., cache_size - 1].mean(axis=0) / denominator for cache_size in cache_sizes],
        axis=0,
    )


def default_cache_sizes(num_experts: int, num_active_experts: int) -> tuple[int, ...]:
    return tuple(sorted({min(num_experts, multiplier * num_active_experts) for multiplier in (1, 2, 4)}))


def lru_resident_budget_counts(
    active_experts: np.ndarray,
    prompt_length: int,
    cache_sizes: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if active_experts.ndim != 3:
        raise ValueError(f"active_experts must have shape [layers, tokens, top_k], got {active_experts.shape}.")
    num_layers, num_tokens, top_k = active_experts.shape
    if prompt_length < 0 or prompt_length > num_tokens:
        raise ValueError(f"prompt_length must be in [0, {num_tokens}], got {prompt_length}.")
    if any(cache_size < top_k for cache_size in cache_sizes):
        raise ValueError(f"cache_sizes must be at least top_k={top_k}, got {cache_sizes}.")
    prompt_hits = np.zeros(len(cache_sizes), dtype=np.int64)
    prompt_misses = np.zeros(len(cache_sizes), dtype=np.int64)
    continuation_hits = np.zeros(len(cache_sizes), dtype=np.int64)
    continuation_misses = np.zeros(len(cache_sizes), dtype=np.int64)
    for cache_index, cache_size in enumerate(cache_sizes):
        layer_caches = tuple(OrderedDict() for _ in range(num_layers))
        for token_index in range(num_tokens):
            phase_hits = prompt_hits if token_index < prompt_length else continuation_hits
            phase_misses = prompt_misses if token_index < prompt_length else continuation_misses
            for layer_index in range(num_layers):
                layer_cache = layer_caches[layer_index]
                token_experts = active_experts[layer_index, token_index].tolist()
                if len(set(token_experts)) != len(token_experts):
                    raise ValueError(f"Expected unique top-k experts per token, got {token_experts}.")
                misses: list[int] = []
                for expert in token_experts:
                    if expert in layer_cache:
                        layer_cache.move_to_end(expert)
                        phase_hits[cache_index] += 1
                    else:
                        misses.append(expert)
                phase_misses[cache_index] += len(misses)
                for expert in misses:
                    layer_cache[expert] = None
                    layer_cache.move_to_end(expert)
                while len(layer_cache) > cache_size:
                    layer_cache.popitem(last=False)
    return prompt_hits, prompt_misses, continuation_hits, continuation_misses


def sequence_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    position_ids = attention_mask.to(dtype=torch.long).cumsum(dim=-1) - 1
    return position_ids.masked_fill(attention_mask == 0, 0)


def router_logits_for_ids(model: Qwen3_5MoeForCausalLM, input_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
    input_device = model.get_input_embeddings().weight.device
    input_ids = input_ids.to(input_device).contiguous()
    attention_mask = make_attention_mask(input_ids).to(input_device).contiguous()

    with torch.inference_mode():
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=sequence_position_ids(attention_mask).contiguous(),
            output_router_logits=True,
            return_dict=True,
        )
    assert outputs.router_logits is not None
    return outputs.router_logits


def split_active_experts(
    active_experts: np.ndarray,
    prompt_length: int,
    continuation_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    assert active_experts.shape[1] == prompt_length + continuation_length
    return (
        active_experts[:, :prompt_length, :],
        active_experts[:, prompt_length:, :],
    )


def model_routing_config(model_repo: str) -> ModelRoutingConfig:
    config = AutoConfig.from_pretrained(model_repo)
    text_config = config.text_config if hasattr(config, "text_config") else config
    revision = getattr(config, "_commit_hash", None)
    if not isinstance(revision, str):
        raise TypeError("Expected model config to expose a string _commit_hash for reproducibility.")
    hidden_size = int(text_config.hidden_size)
    moe_intermediate_size = int(text_config.moe_intermediate_size)
    dtype = getattr(text_config, "dtype", None)
    if dtype not in {"bfloat16", torch.bfloat16}:
        raise ValueError(f"Expected bfloat16 expert weights for transfer accounting, got {dtype}.")
    return ModelRoutingConfig(
        repo=model_repo,
        revision=revision,
        num_layers=int(text_config.num_hidden_layers),
        num_experts=int(text_config.num_experts),
        num_active_experts=int(text_config.num_experts_per_tok),
        expert_parameters=3 * hidden_size * moe_intermediate_size,
        expert_bytes=3 * hidden_size * moe_intermediate_size * 2,
    )


def auto_device_map_kwargs(
    headroom_gib: int = AUTO_DEVICE_MAP_HEADROOM_GIB,
    *,
    device_map: str = "auto",
) -> dict[str, object]:
    if not torch.cuda.is_available():
        return {
            "torch_dtype": torch.bfloat16,
            "device_map": device_map,
        }
    max_memory: dict[int | str, str] = {"cpu": "512GiB"}
    for device_index in range(torch.cuda.device_count()):
        total_memory_gib = int(torch.cuda.get_device_properties(device_index).total_memory // (1024**3))
        usable_memory_gib = max(1, total_memory_gib - headroom_gib)
        max_memory[device_index] = f"{usable_memory_gib}GiB"
    offload_folder = Path("/dev/shm/qwen_moe_offload")
    if not offload_folder.parent.is_dir():
        offload_folder = Path(".qwen_moe_offload")
    offload_folder.mkdir(parents=True, exist_ok=True)
    return {
        "torch_dtype": torch.bfloat16,
        "device_map": device_map,
        "max_memory": max_memory,
        "offload_folder": str(offload_folder),
        "offload_state_dict": True,
    }


def load_model(
    model_repo: str,
    device_map_mode: str,
    *,
    auto_headroom_gib: int = AUTO_DEVICE_MAP_HEADROOM_GIB,
) -> Qwen3_5MoeForCausalLM:
    def disable_caching_allocator_warmup(*_args: object, **_kwargs: object) -> None:
        return None

    if device_map_mode == "single-gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("single-gpu mode requires CUDA.")
        load_kwargs: dict[str, object] = {
            "torch_dtype": torch.bfloat16,
            "device_map": {"": 0},
        }
    elif device_map_mode in {"auto", "balanced-low-0"}:
        load_kwargs = auto_device_map_kwargs(
            auto_headroom_gib,
            device_map="balanced_low_0" if device_map_mode == "balanced-low-0" else "auto",
        )
    else:
        raise ValueError(f"Unsupported device_map_mode: {device_map_mode}.")

    original_parallel_loading = os.environ.get("HF_ENABLE_PARALLEL_LOADING")
    original_caching_allocator_warmup = transformers.modeling_utils.caching_allocator_warmup
    os.environ["HF_ENABLE_PARALLEL_LOADING"] = "false"
    transformers.modeling_utils.caching_allocator_warmup = disable_caching_allocator_warmup
    try:
        return Qwen3_5MoeForCausalLM.from_pretrained(model_repo, **load_kwargs)
    finally:
        transformers.modeling_utils.caching_allocator_warmup = original_caching_allocator_warmup
        if original_parallel_loading is None:
            del os.environ["HF_ENABLE_PARALLEL_LOADING"]
        else:
            os.environ["HF_ENABLE_PARALLEL_LOADING"] = original_parallel_loading


def run_experiment(config: ExperimentConfig) -> RoutingAnalysisResult:
    routing_config = model_routing_config(config.model_repo)
    cache_sizes = default_cache_sizes(routing_config.num_experts, routing_config.num_active_experts)
    tokenizer = AutoTokenizer.from_pretrained(config.model_repo)
    model = load_model(config.model_repo, config.device_map_mode)
    model.eval()

    prompt_stats = PhaseAccumulator(
        window_sizes=config.window_sizes,
        num_layers=routing_config.num_layers,
        num_experts=routing_config.num_experts,
        num_active_experts=routing_config.num_active_experts,
        cache_sizes=cache_sizes,
        expert_bytes=routing_config.expert_bytes,
    )
    continuation_stats = PhaseAccumulator(
        window_sizes=config.window_sizes,
        num_layers=routing_config.num_layers,
        num_experts=routing_config.num_experts,
        num_active_experts=routing_config.num_active_experts,
        cache_sizes=cache_sizes,
        expert_bytes=routing_config.expert_bytes,
    )
    ewma_stats = tuple(
        EwmaAccumulator(
            alpha=alpha,
            prompt=PhaseAccumulator(
                window_sizes=config.window_sizes,
                num_layers=routing_config.num_layers,
                num_experts=routing_config.num_experts,
                num_active_experts=routing_config.num_active_experts,
                cache_sizes=cache_sizes,
                expert_bytes=routing_config.expert_bytes,
            ),
            continuation=PhaseAccumulator(
                window_sizes=config.window_sizes,
                num_layers=routing_config.num_layers,
                num_experts=routing_config.num_experts,
                num_active_experts=routing_config.num_active_experts,
                cache_sizes=cache_sizes,
                expert_bytes=routing_config.expert_bytes,
            ),
            prompt_agreement=AgreementAccumulator(num_layers=routing_config.num_layers),
            continuation_agreement=AgreementAccumulator(num_layers=routing_config.num_layers),
        )
        for alpha in config.ewma_alphas
    )

    rows, dataset_rows_total = load_rows(config.dataset, config.dataset_split, config.max_rows, config.seed)
    samples = indexed_conversation_samples(rows, config.seed)
    assistant_turns_total = len(samples)
    prompts_processed = 0
    skipped_prompt_too_long = 0
    skipped_continuation_too_long = 0
    skipped_empty_continuation = 0
    processed_samples: list[ProcessedSample] = []

    for indexed_sample in samples:
        prompt_ids, continuation_ids, skip_reason = prompt_and_continuation_ids(
            tokenizer=tokenizer,
            sample=indexed_sample.sample,
            max_prompt_tokens=config.max_prompt_tokens,
            max_continuation_tokens=config.max_continuation_tokens,
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
        full_router_logits = router_logits_for_ids(
            model,
            torch.cat([prompt_ids, continuation_ids], dim=1),
        )
        prompt_active_experts, continuation_active_experts = split_active_experts(
            topk_active_experts(full_router_logits, routing_config.num_active_experts),
            prompt_length,
            continuation_length,
        )
        prompt_stats.update(prompt_active_experts)
        continuation_stats.update(continuation_active_experts)
        prompt_hits, prompt_misses, continuation_hits, continuation_misses = lru_resident_budget_counts(
            np.concatenate((prompt_active_experts, continuation_active_experts), axis=1),
            prompt_length,
            cache_sizes,
        )
        prompt_stats.update_resident_budgets(prompt_hits, prompt_misses, prompt_length)
        continuation_stats.update_resident_budgets(continuation_hits, continuation_misses, continuation_length)
        for ewma_stat in ewma_stats:
            ewma_prompt_active_experts, ewma_continuation_active_experts = split_active_experts(
                ewma_topk_active_experts(
                    full_router_logits,
                    routing_config.num_active_experts,
                    ewma_stat.alpha,
                ),
                prompt_length,
                continuation_length,
            )
            ewma_stat.prompt.update(ewma_prompt_active_experts)
            ewma_stat.continuation.update(ewma_continuation_active_experts)
            ewma_prompt_hits, ewma_prompt_misses, ewma_continuation_hits, ewma_continuation_misses = (
                lru_resident_budget_counts(
                    np.concatenate((ewma_prompt_active_experts, ewma_continuation_active_experts), axis=1),
                    prompt_length,
                    cache_sizes,
                )
            )
            ewma_stat.prompt.update_resident_budgets(ewma_prompt_hits, ewma_prompt_misses, prompt_length)
            ewma_stat.continuation.update_resident_budgets(
                ewma_continuation_hits,
                ewma_continuation_misses,
                continuation_length,
            )
            ewma_stat.prompt_agreement.update(prompt_active_experts, ewma_prompt_active_experts)
            ewma_stat.continuation_agreement.update(
                continuation_active_experts,
                ewma_continuation_active_experts,
            )
        prompts_processed += 1
        processed_samples.append(
            ProcessedSample(
                row_id=indexed_sample.row_id,
                conversation_index=indexed_sample.conversation_index,
                assistant_turn_index=indexed_sample.assistant_turn_index,
                prompt_tokens=prompt_active_experts.shape[1],
                continuation_tokens=continuation_active_experts.shape[1],
            )
        )
        print(
            "processed "
            f"row_id={indexed_sample.row_id} "
            f"conversation={indexed_sample.conversation_index} "
            f"assistant_turn={indexed_sample.assistant_turn_index} "
            f"prompt={prompts_processed} "
            f"prompt_tokens={prompt_active_experts.shape[1]} "
            f"continuation_tokens={continuation_active_experts.shape[1]}",
            file=sys.stderr,
        )
        if config.max_prompts is not None and prompts_processed >= config.max_prompts:
            return RoutingAnalysisResult(
                config=config,
                model=routing_config,
                runtime=runtime_info(model, config.device_map_mode),
                dataset_fingerprint=dataset_fingerprint(rows),
                dataset_rows_total=dataset_rows_total,
                assistant_turns_total=assistant_turns_total,
                prompt_statistics=prompt_stats.finalize(),
                continuation_statistics=continuation_stats.finalize(),
                ewma_statistics=tuple(ewma_stat.finalize() for ewma_stat in ewma_stats),
                dataset_rows_processed=len(rows),
                prompts_processed=prompts_processed,
                skipped_prompt_too_long=skipped_prompt_too_long,
                skipped_continuation_too_long=skipped_continuation_too_long,
                skipped_empty_continuation=skipped_empty_continuation,
                processed_samples=tuple(processed_samples),
            )

    return RoutingAnalysisResult(
        config=config,
        model=routing_config,
        runtime=runtime_info(model, config.device_map_mode),
        dataset_fingerprint=dataset_fingerprint(rows),
        dataset_rows_total=dataset_rows_total,
        assistant_turns_total=assistant_turns_total,
        prompt_statistics=prompt_stats.finalize(),
        continuation_statistics=continuation_stats.finalize(),
        ewma_statistics=tuple(ewma_stat.finalize() for ewma_stat in ewma_stats),
        dataset_rows_processed=len(rows),
        prompts_processed=prompts_processed,
        skipped_prompt_too_long=skipped_prompt_too_long,
        skipped_continuation_too_long=skipped_continuation_too_long,
        skipped_empty_continuation=skipped_empty_continuation,
        processed_samples=tuple(processed_samples),
    )


def runtime_info(model: Qwen3_5MoeForCausalLM, device_map_mode: str) -> RuntimeInfo:
    raw_device_map = getattr(model, "hf_device_map", None)
    device_map = None if raw_device_map is None else {key: str(value) for key, value in raw_device_map.items()}
    return RuntimeInfo(
        transformers_version=transformers.__version__,
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        device_count=torch.cuda.device_count(),
        devices=tuple(torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())),
        model_dtype=str(next(model.parameters()).dtype),
        model_class=type(model).__name__,
        device_map_mode=device_map_mode,
        hf_device_map=device_map,
    )


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Local dataset path or Hugging Face dataset repo id.")
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--model-repo", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-continuation-tokens", type=int, default=256)
    parser.add_argument("--window-sizes", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--ewma-alphas", default="")
    parser.add_argument("--device-map-mode", choices=("single-gpu", "auto", "balanced-low-0"), default=None)
    args = parser.parse_args()
    device_map_mode = args.device_map_mode
    if device_map_mode is None:
        device_map_mode = "single-gpu" if torch.cuda.is_available() else "auto"
    return ExperimentConfig(
        model_repo=args.model_repo,
        dataset=args.dataset,
        dataset_split=args.dataset_split,
        output_path=args.output_path,
        seed=args.seed,
        max_rows=args.max_rows,
        max_prompts=args.max_prompts,
        max_prompt_tokens=args.max_prompt_tokens,
        max_continuation_tokens=args.max_continuation_tokens,
        window_sizes=parse_window_sizes(args.window_sizes),
        ewma_alphas=parse_ewma_alphas(args.ewma_alphas),
        device_map_mode=device_map_mode,
    )


def main() -> None:
    config = parse_args()
    result = run_experiment(config)
    payload = asdict(result)
    payload["config"]["output_path"] = str(config.output_path)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
