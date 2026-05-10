from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Annotated

from annotated_types import Ge

from lalamo.data.completion_features import FeatureRequest, LalamoCompletionFeatures
from lalamo.data.lalamo_completions import LalamoCompletion
from lalamo.models.completion_feature_extractor import FeatureQueue, OnlineCompletionFeatureExtractor
from lalamo.modules.decoder import Decoder
from lalamo.speculator.common import Speculator
from lalamo.speculator.proposal import GumbelSampler

__all__ = [
    "SpeculatorBatchResult",
    "SpeculatorTrainer",
    "SpeculatorTrainingConfig",
    "SpeculatorTrainingContext",
    "SpeculatorTrainingEvent",
    "SpeculatorTrainingProgress",
    "TrainingFeatureRequest",
    "train_speculator",
]


@dataclass(frozen=True)
class SpeculatorTrainingConfig:
    batch_size: Annotated[int, Ge(1)] = 8
    max_prefetch: Annotated[int, Ge(1)] = 2
    top_k_logits: Annotated[int, Ge(1)] = 128
    epochs: Annotated[int, Ge(1)] = 1
    eval_every_epochs: Annotated[int, Ge(1)] = 1
    early_stopping_patience: Annotated[int, Ge(1)] | None = None
    tokens_to_train: int | None = None


@dataclass(frozen=True)
class SpeculatorTrainingProgress:
    trained_sequences: int
    trained_tokens: int
    trained_steps: int
    evaluated_sequences: int
    evaluated_tokens: int
    evaluated_steps: int

    @staticmethod
    def create() -> "SpeculatorTrainingProgress":
        return SpeculatorTrainingProgress(
            trained_sequences=0,
            trained_tokens=0,
            trained_steps=0,
            evaluated_sequences=0,
            evaluated_tokens=0,
            evaluated_steps=0,
        )

    def after_train(self, features: LalamoCompletionFeatures) -> "SpeculatorTrainingProgress":
        sequences, tokens = count_feature_tokens(features)
        return SpeculatorTrainingProgress(
            trained_sequences=self.trained_sequences + sequences,
            trained_tokens=self.trained_tokens + tokens,
            trained_steps=self.trained_steps + 1,
            evaluated_sequences=self.evaluated_sequences,
            evaluated_tokens=self.evaluated_tokens,
            evaluated_steps=self.evaluated_steps,
        )

    def after_evaluate(self, features: LalamoCompletionFeatures) -> "SpeculatorTrainingProgress":
        sequences, tokens = count_feature_tokens(features)
        return SpeculatorTrainingProgress(
            trained_sequences=self.trained_sequences,
            trained_tokens=self.trained_tokens,
            trained_steps=self.trained_steps,
            evaluated_sequences=self.evaluated_sequences + sequences,
            evaluated_tokens=self.evaluated_tokens + tokens,
            evaluated_steps=self.evaluated_steps + 1,
        )


@dataclass(frozen=True)
class SpeculatorTrainingEvent:
    epoch: int
    progress: SpeculatorTrainingProgress
    loss: float | None = None


@dataclass(frozen=True)
class SpeculatorTrainingContext:
    epoch: int
    step: int
    progress: SpeculatorTrainingProgress


@dataclass(frozen=True)
class SpeculatorBatchResult:
    loss: float | None = None
    loss_weight: float = 1.0


@dataclass(frozen=True)
class TrainingFeatureRequest:
    output_features: bool = False
    layer_indices: tuple[int, ...] = ()


class SpeculatorTrainer[SpeculatorT: Speculator, StateT](ABC):
    @property
    @abstractmethod
    def feature_request(self) -> TrainingFeatureRequest: ...

    @abstractmethod
    def init_state(self) -> StateT: ...

    @abstractmethod
    def train(
        self,
        state: StateT,
        features: LalamoCompletionFeatures,
        context: SpeculatorTrainingContext,
    ) -> tuple[StateT, SpeculatorBatchResult]: ...

    @abstractmethod
    def evaluate(
        self,
        state: StateT,
        features: LalamoCompletionFeatures,
        context: SpeculatorTrainingContext,
    ) -> SpeculatorBatchResult: ...

    @abstractmethod
    def finish(
        self,
        state: StateT,
        target_decoder: Decoder,
        sampler: GumbelSampler,
    ) -> SpeculatorT: ...

    @abstractmethod
    def save(
        self,
        state: StateT,
        target_decoder: Decoder,
        sampler: GumbelSampler,
        event: SpeculatorTrainingEvent,
    ) -> None: ...


def train_speculator[SpeculatorT: Speculator, StateT](
    trainer: SpeculatorTrainer[SpeculatorT, StateT],
    extractor: OnlineCompletionFeatureExtractor,
    train_completions: Callable[[], Iterable[LalamoCompletion]],
    eval_completions: Callable[[], Iterable[LalamoCompletion]] | None,
    target_decoder: Decoder,
    sampler: GumbelSampler,
    config: SpeculatorTrainingConfig,
    progress_callback: Callable[[SpeculatorTrainingEvent], None] | None = None,
) -> SpeculatorT:
    state = trainer.init_state()
    best_state = state
    best_event: SpeculatorTrainingEvent | None = None
    best_loss: float | None = None
    bad_eval_count = 0
    last_event: SpeculatorTrainingEvent | None = None
    progress = SpeculatorTrainingProgress.create()

    for epoch in range(1, config.epochs + 1):
        token_budget = None
        if config.tokens_to_train is not None:
            token_budget = max(config.tokens_to_train - progress.trained_tokens, 0)
            if token_budget == 0:
                break

        state, progress, last_event = run_training_epoch(
            trainer,
            extractor,
            train_completions,
            config,
            state,
            epoch,
            progress,
            token_budget,
            progress_callback,
        )

        if eval_completions is None or epoch % config.eval_every_epochs != 0:
            continue

        progress, eval_event = run_evaluation_epoch(
            trainer,
            extractor,
            eval_completions,
            config,
            state,
            epoch,
            progress,
            progress_callback,
        )
        if eval_event is None or eval_event.loss is None:
            continue

        last_event = eval_event
        if best_loss is None or eval_event.loss < best_loss:
            best_state = state
            best_event = eval_event
            best_loss = eval_event.loss
            bad_eval_count = 0
            trainer.save(best_state, target_decoder, sampler, best_event)
        else:
            bad_eval_count += 1
            if (
                config.early_stopping_patience is not None
                and bad_eval_count >= config.early_stopping_patience
            ):
                break

    if best_event is None:
        best_state = state
        best_event = last_event or SpeculatorTrainingEvent(
            epoch=0,
            progress=progress,
        )
        trainer.save(best_state, target_decoder, sampler, best_event)

    return trainer.finish(best_state, target_decoder, sampler)


def run_training_epoch[SpeculatorT: Speculator, StateT](
    trainer: SpeculatorTrainer[SpeculatorT, StateT],
    extractor: OnlineCompletionFeatureExtractor,
    train_completions: Callable[[], Iterable[LalamoCompletion]],
    config: SpeculatorTrainingConfig,
    state: StateT,
    epoch: int,
    progress: SpeculatorTrainingProgress,
    token_budget: int | None,
    progress_callback: Callable[[SpeculatorTrainingEvent], None] | None,
) -> tuple[StateT, SpeculatorTrainingProgress, SpeculatorTrainingEvent | None]:
    last_event = None
    requests = iter_training_feature_requests(
        train_completions,
        trainer.feature_request,
        config,
        token_budget,
    )
    feature_queue = FeatureQueue(extractor, config.max_prefetch)

    for features in feature_queue.iter_features(requests):
        context = SpeculatorTrainingContext(
            epoch=epoch,
            step=progress.trained_steps + 1,
            progress=progress,
        )
        state, result = trainer.train(state, features, context)
        progress = progress.after_train(features)
        last_event = SpeculatorTrainingEvent(
            epoch=epoch,
            progress=progress,
            loss=result.loss,
        )
        if progress_callback is not None:
            progress_callback(last_event)

    return state, progress, last_event


def run_evaluation_epoch[SpeculatorT: Speculator, StateT](
    trainer: SpeculatorTrainer[SpeculatorT, StateT],
    extractor: OnlineCompletionFeatureExtractor,
    eval_completions: Callable[[], Iterable[LalamoCompletion]],
    config: SpeculatorTrainingConfig,
    state: StateT,
    epoch: int,
    progress: SpeculatorTrainingProgress,
    progress_callback: Callable[[SpeculatorTrainingEvent], None] | None,
) -> tuple[SpeculatorTrainingProgress, SpeculatorTrainingEvent | None]:
    total_loss = 0.0
    total_loss_weight = 0.0
    last_event = None
    requests = iter_evaluation_feature_requests(
        eval_completions,
        trainer.feature_request,
        config,
    )
    feature_queue = FeatureQueue(extractor, config.max_prefetch)

    for features in feature_queue.iter_features(requests):
        context = SpeculatorTrainingContext(
            epoch=epoch,
            step=progress.evaluated_steps + 1,
            progress=progress,
        )
        result = trainer.evaluate(state, features, context)
        if result.loss is not None:
            total_loss += result.loss * result.loss_weight
            total_loss_weight += result.loss_weight

        progress = progress.after_evaluate(features)
        last_event = SpeculatorTrainingEvent(
            epoch=epoch,
            progress=progress,
            loss=total_loss / total_loss_weight if total_loss_weight > 0.0 else None,
        )
        if progress_callback is not None:
            progress_callback(last_event)

    return progress, last_event


def iter_training_feature_requests(
    completions: Callable[[], Iterable[LalamoCompletion]],
    training_request: TrainingFeatureRequest,
    config: SpeculatorTrainingConfig,
    token_budget: int | None,
) -> Iterator[FeatureRequest]:
    requested_tokens = 0
    batch: list[LalamoCompletion] = []

    for sequence_index, completion in enumerate(completions()):
        if token_budget is not None and requested_tokens >= token_budget:
            break

        requested_tokens += len(completion.completion_token_ids)
        batch.append(completion)

        if len(batch) == config.batch_size:
            yield make_feature_request(tuple(batch), training_request, config)
            batch.clear()

    if batch:
        yield make_feature_request(tuple(batch), training_request, config)


def iter_evaluation_feature_requests(
    completions: Callable[[], Iterable[LalamoCompletion]],
    training_request: TrainingFeatureRequest,
    config: SpeculatorTrainingConfig,
) -> Iterator[FeatureRequest]:
    batch: list[LalamoCompletion] = []

    for completion in completions():
        batch.append(completion)

        if len(batch) == config.batch_size:
            yield make_feature_request(tuple(batch), training_request, config)
            batch.clear()

    if batch:
        yield make_feature_request(tuple(batch), training_request, config)


def make_feature_request(
    completions: tuple[LalamoCompletion, ...],
    training_request: TrainingFeatureRequest,
    config: SpeculatorTrainingConfig,
) -> FeatureRequest:
    return FeatureRequest(
        completions=completions,
        top_k_logits=config.top_k_logits,
        layer_indices=training_request.layer_indices,
        output_features=training_request.output_features,
    )

def count_feature_tokens(features: LalamoCompletionFeatures) -> tuple[int, int]:
    return features.completion_batch.target_mask.shape[0], int(features.completion_batch.target_mask.sum())
