from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from itertools import batched
from typing import Annotated

import jax
from annotated_types import Ge

from lalamo.data.completion_features import FeatureRequest, LalamoCompletionFeatures  # noqa: TC001
from lalamo.data.lalamo_completions import LalamoCompletion  # noqa: TC001
from lalamo.models.completion_feature_extractor import FeatureQueue, OnlineCompletionFeatureExtractor, jax_device
from lalamo.speculator.common import Speculator

__all__ = [
    "SpeculatorBatchResult",
    "SpeculatorTrainer",
    "SpeculatorTrainingConfig",
    "SpeculatorTrainingContext",
    "SpeculatorTrainingEvent",
    "SpeculatorTrainingPhase",
    "SpeculatorTrainingProgress",
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


@dataclass(frozen=True)
class SpeculatorTrainingProgress:
    trained_sequences: int
    trained_tokens: int
    trained_steps: int
    evaluated_sequences: int
    evaluated_tokens: int
    evaluated_steps: int

    @staticmethod
    def create() -> SpeculatorTrainingProgress:
        return SpeculatorTrainingProgress(
            trained_sequences=0,
            trained_tokens=0,
            trained_steps=0,
            evaluated_sequences=0,
            evaluated_tokens=0,
            evaluated_steps=0,
        )

    def after_train(self, features: LalamoCompletionFeatures) -> SpeculatorTrainingProgress:
        sequences, tokens = count_feature_tokens(features)
        return SpeculatorTrainingProgress(
            trained_sequences=self.trained_sequences + sequences,
            trained_tokens=self.trained_tokens + tokens,
            trained_steps=self.trained_steps + 1,
            evaluated_sequences=self.evaluated_sequences,
            evaluated_tokens=self.evaluated_tokens,
            evaluated_steps=self.evaluated_steps,
        )

    def after_evaluate(self, features: LalamoCompletionFeatures) -> SpeculatorTrainingProgress:
        sequences, tokens = count_feature_tokens(features)
        return SpeculatorTrainingProgress(
            trained_sequences=self.trained_sequences,
            trained_tokens=self.trained_tokens,
            trained_steps=self.trained_steps,
            evaluated_sequences=self.evaluated_sequences + sequences,
            evaluated_tokens=self.evaluated_tokens + tokens,
            evaluated_steps=self.evaluated_steps + 1,
        )


class SpeculatorTrainingPhase(str, Enum):
    TRAIN = "train"
    EVAL = "eval"


@dataclass(frozen=True)
class SpeculatorTrainingSchedule:
    total_epochs: int
    train_batches_per_epoch: int
    eval_batches_per_epoch: int
    total_batches: int

    @classmethod
    def create(
        cls,
        config: SpeculatorTrainingConfig,
        train_sequences: int,
        eval_sequences: int,
    ) -> SpeculatorTrainingSchedule:
        train_batches_per_epoch = count_batches(train_sequences, config.batch_size)
        eval_batches_per_epoch = count_batches(eval_sequences, config.batch_size)
        eval_epochs = config.epochs // config.eval_every_epochs if eval_sequences > 0 else 0
        return cls(
            total_epochs=config.epochs,
            train_batches_per_epoch=train_batches_per_epoch,
            eval_batches_per_epoch=eval_batches_per_epoch,
            total_batches=config.epochs * train_batches_per_epoch + eval_epochs * eval_batches_per_epoch,
        )


@dataclass(frozen=True)
class SpeculatorTrainingEvent:
    epoch: int
    progress: SpeculatorTrainingProgress
    phase: SpeculatorTrainingPhase = SpeculatorTrainingPhase.TRAIN
    total_epochs: int = 1
    batch_index: int = 0
    total_epoch_batches: int = 0
    completed_batches: int = 0
    total_batches: int = 0
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


class SpeculatorTrainer[SpeculatorT: Speculator, StateT](ABC):
    @abstractmethod
    def make_feature_request(
        self,
        completions: tuple[LalamoCompletion, ...],
        config: SpeculatorTrainingConfig,
    ) -> FeatureRequest: ...

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
    ) -> SpeculatorT: ...

    @abstractmethod
    def save(
        self,
        state: StateT,
        event: SpeculatorTrainingEvent,
    ) -> None: ...


def train_speculator[SpeculatorT: Speculator, StateT](
    trainer: SpeculatorTrainer[SpeculatorT, StateT],
    extractor: OnlineCompletionFeatureExtractor,
    train_completions: Callable[[], Iterable[LalamoCompletion]],
    eval_completions: Callable[[], Iterable[LalamoCompletion]] | None,
    config: SpeculatorTrainingConfig,
    training_device_id: int = 0,
    progress_callback: Callable[[SpeculatorTrainingEvent], None] | None = None,
) -> SpeculatorT:
    train_sequences = count_completions(train_completions)
    eval_sequences = count_completions(eval_completions) if eval_completions is not None else 0
    schedule = SpeculatorTrainingSchedule.create(config, train_sequences, eval_sequences)
    progress = SpeculatorTrainingProgress.create()
    last_event: SpeculatorTrainingEvent | None = SpeculatorTrainingEvent(
        epoch=1,
        progress=progress,
        phase=SpeculatorTrainingPhase.TRAIN,
        total_epochs=schedule.total_epochs,
        batch_index=0,
        total_epoch_batches=schedule.train_batches_per_epoch,
        completed_batches=0,
        total_batches=schedule.total_batches,
    )
    if progress_callback is not None:
        progress_callback(last_event)

    training_device = jax_device(training_device_id)
    with jax.default_device(training_device):
        state = trainer.init_state()
    best_state = state
    best_event: SpeculatorTrainingEvent | None = None
    best_loss: float | None = None
    bad_eval_count = 0

    for epoch in range(1, config.epochs + 1):
        state, progress, last_event = run_training_epoch(
            trainer,
            extractor,
            train_completions,
            config,
            state,
            epoch,
            progress,
            training_device,
            training_device_id,
            schedule,
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
            training_device,
            training_device_id,
            schedule,
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
            trainer.save(best_state, best_event)
        else:
            bad_eval_count += 1
            if (
                config.early_stopping_patience is not None
                and bad_eval_count >= config.early_stopping_patience
            ):
                break

    if best_event is None:
        best_state = state
        assert last_event is not None
        best_event = last_event
        trainer.save(best_state, best_event)

    return trainer.finish(best_state)


def run_training_epoch[SpeculatorT: Speculator, StateT](
    trainer: SpeculatorTrainer[SpeculatorT, StateT],
    extractor: OnlineCompletionFeatureExtractor,
    train_completions: Callable[[], Iterable[LalamoCompletion]],
    config: SpeculatorTrainingConfig,
    state: StateT,
    epoch: int,
    progress: SpeculatorTrainingProgress,
    training_device: jax.Device,
    training_device_id: int,
    schedule: SpeculatorTrainingSchedule,
    progress_callback: Callable[[SpeculatorTrainingEvent], None] | None,
) -> tuple[StateT, SpeculatorTrainingProgress, SpeculatorTrainingEvent | None]:
    last_event = None
    requests = (
        trainer.make_feature_request(completion_batch, config)
        for completion_batch in batched(train_completions(), config.batch_size)
    )
    feature_queue = FeatureQueue(
        extractor=extractor,
        max_prefetch=config.max_prefetch,
        target_device_id=training_device_id,
    )

    for batch_index, features in enumerate(feature_queue.iter_features(requests), start=1):
        context = SpeculatorTrainingContext(
            epoch=epoch,
            step=progress.trained_steps + 1,
            progress=progress,
        )
        with jax.default_device(training_device):
            state, result = trainer.train(state, features, context)
        progress = progress.after_train(features)
        last_event = SpeculatorTrainingEvent(
            epoch=epoch,
            progress=progress,
            phase=SpeculatorTrainingPhase.TRAIN,
            total_epochs=schedule.total_epochs,
            batch_index=batch_index,
            total_epoch_batches=schedule.train_batches_per_epoch,
            completed_batches=progress.trained_steps + progress.evaluated_steps,
            total_batches=schedule.total_batches,
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
    training_device: jax.Device,
    training_device_id: int,
    schedule: SpeculatorTrainingSchedule,
    progress_callback: Callable[[SpeculatorTrainingEvent], None] | None,
) -> tuple[SpeculatorTrainingProgress, SpeculatorTrainingEvent | None]:
    total_loss = 0.0
    total_loss_weight = 0.0
    last_event = None
    requests = (
        trainer.make_feature_request(completion_batch, config)
        for completion_batch in batched(eval_completions(), config.batch_size)
    )
    feature_queue = FeatureQueue(
        extractor=extractor,
        max_prefetch=config.max_prefetch,
        target_device_id=training_device_id,
    )

    for batch_index, features in enumerate(feature_queue.iter_features(requests), start=1):
        context = SpeculatorTrainingContext(
            epoch=epoch,
            step=progress.evaluated_steps + 1,
            progress=progress,
        )
        with jax.default_device(training_device):
            result = trainer.evaluate(state, features, context)
        if result.loss is not None:
            total_loss += result.loss * result.loss_weight
            total_loss_weight += result.loss_weight

        progress = progress.after_evaluate(features)
        last_event = SpeculatorTrainingEvent(
            epoch=epoch,
            progress=progress,
            phase=SpeculatorTrainingPhase.EVAL,
            total_epochs=schedule.total_epochs,
            batch_index=batch_index,
            total_epoch_batches=schedule.eval_batches_per_epoch,
            completed_batches=progress.trained_steps + progress.evaluated_steps,
            total_batches=schedule.total_batches,
            loss=total_loss / total_loss_weight if total_loss_weight > 0.0 else None,
        )
        if progress_callback is not None:
            progress_callback(last_event)

    return progress, last_event


def count_feature_tokens(features: LalamoCompletionFeatures) -> tuple[int, int]:
    return features.completion_batch.target_mask.shape[0], int(features.completion_batch.target_mask.sum())


def count_completions(completions: Callable[[], Iterable[LalamoCompletion]]) -> int:
    return sum(1 for _ in completions())


def count_batches(num_sequences: int, batch_size: int) -> int:
    return (num_sequences + batch_size - 1) // batch_size
