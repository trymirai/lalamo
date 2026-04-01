from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import optax
import polars as pl
import pytest

from lalamo.distill_runner import (
    DistillConfig,
    LoadedDistillBatches,
    OptimizerName,
    _load_tokenized_conversations,
    _shard_distill_batches,
    _validate_distill_models,
    distill,
)
from lalamo.distillation import (
    DistillBatch,
    DistillBatchMetrics,
    DistillOptimizerState,
    DistillParameterSummary,
    DistillTrainingState,
)
from lalamo.quantization import QuantizationMode


@dataclass(frozen=True)
class _StubModelConfig:
    weight_quantization_mode: QuantizationMode


@dataclass(frozen=True)
class _StubLanguageModelConfig:
    model_config: _StubModelConfig
    message_processor_config: str = "default"


_EMPTY_PARAMETER_SUMMARY = DistillParameterSummary(
    total_parameters=0,
    trainable_parameters=0,
    total_master_bytes=0,
    by_group={},
)


def _make_config(tmp_path: Path) -> DistillConfig:
    return DistillConfig(
        teacher_path=tmp_path / "teacher",
        student_path=tmp_path / "student",
        dataset_path=tmp_path / "dataset.parquet",
        output_dir=tmp_path / "output",
        batch_size=1,
        train_examples=1,
        eval_examples=1,
        num_steps=1,
        optimizer_name=OptimizerName.SGD,
        quantization_mode=QuantizationMode.UINT4,
    )


def _make_language_model(
    *,
    tokenizer_signature: str = "shared-tokenizer",
    quantization_mode: QuantizationMode = QuantizationMode.UINT4,
    message_processor_config: str = "default",
    tokenize_request: Callable[[object], list[int]] | None = None,
) -> SimpleNamespace:
    if tokenize_request is None:

        def tokenize_request(_messages: object) -> list[int]:
            return [1, 2, 3]

    return SimpleNamespace(
        config=_StubLanguageModelConfig(
            model_config=_StubModelConfig(weight_quantization_mode=quantization_mode),
            message_processor_config=message_processor_config,
        ),
        model=SimpleNamespace(activation_precision=jnp.float32, vocab_size=32),
        message_processor=SimpleNamespace(
            tokenize_request=tokenize_request,
            tokenizer=SimpleNamespace(to_str=lambda: tokenizer_signature),
        ),
    )


def test_load_tokenized_conversations_skips_short_rows_and_keeps_later_examples() -> None:
    dataframe = pl.DataFrame(
        {
            "messages": [
                [{"role": "user", "content": "drop"}],
                [{"role": "user", "content": "keep-one"}],
                [{"role": "user", "content": "keep-two"}],
            ],
        },
    )
    language_model = _make_language_model(
        tokenize_request=lambda messages: {
            "drop": [1],
            "keep-one": [1, 2, 3],
            "keep-two": [4, 5, 6, 7],
        }[messages[0].content],
    )

    with (
        patch("lalamo.distill_runner.load_hf_parquet", return_value=dataframe),
        patch("lalamo.distill_runner.shuffle_dataset", return_value=dataframe),
    ):
        tokenized = _load_tokenized_conversations(
            Path("dataset.parquet"),
            seed=0,
            language_model=language_model,
            max_sequence_length=3,
            num_examples=2,
        )

    assert [sequence.tolist() for sequence in tokenized] == [[1, 2, 3], [4, 5, 6]]


def test_shard_distill_batches_rejects_partial_batches() -> None:
    dataset = LoadedDistillBatches(
        train_examples=3,
        eval_examples=2,
        train_batches=[
            DistillBatch(
                token_ids=jnp.ones((3, 4), dtype=jnp.int32),
                lengths_without_padding=jnp.ones((3,), dtype=jnp.int32),
            ),
        ],
        eval_batches=[
            DistillBatch(
                token_ids=jnp.ones((2, 4), dtype=jnp.int32),
                lengths_without_padding=jnp.ones((2,), dtype=jnp.int32),
            ),
        ],
    )

    with (
        patch("lalamo.distill_runner.NamedSharding", return_value=None),
        pytest.raises(ValueError, match="Batch size 3 is not divisible by num_devices=2"),
    ):
        _shard_distill_batches(dataset, object(), num_devices=2)


def test_shard_distill_batches_keeps_truthful_counts() -> None:
    dataset = LoadedDistillBatches(
        train_examples=999,
        eval_examples=999,
        train_batches=[
            DistillBatch(
                token_ids=jnp.ones((2, 4), dtype=jnp.int32),
                lengths_without_padding=jnp.ones((2,), dtype=jnp.int32),
            ),
        ],
        eval_batches=[
            DistillBatch(
                token_ids=jnp.ones((4, 4), dtype=jnp.int32),
                lengths_without_padding=jnp.ones((4,), dtype=jnp.int32),
            ),
        ],
    )

    with (
        patch("lalamo.distill_runner.NamedSharding", return_value=None),
        patch("lalamo.distill_runner.eqx.filter_shard", side_effect=lambda batch, _sharding: batch),
    ):
        sharded = _shard_distill_batches(dataset, object(), num_devices=2)

    assert sharded.train_examples == 2
    assert sharded.eval_examples == 4


@pytest.mark.parametrize(
    ("student_model", "teacher_model", "quantization_mode", "error_message"),
    [
        (
            _make_language_model(tokenizer_signature="student"),
            _make_language_model(tokenizer_signature="teacher"),
            QuantizationMode.UINT4,
            "identical tokenizers",
        ),
        (
            _make_language_model(quantization_mode=QuantizationMode.UINT8),
            _make_language_model(quantization_mode=QuantizationMode.UINT8),
            QuantizationMode.UINT4,
            "does not match the student model quantization mode",
        ),
    ],
)
def test_validate_distill_models_rejects_invalid_inputs(
    student_model: SimpleNamespace,
    teacher_model: SimpleNamespace,
    quantization_mode: QuantizationMode,
    error_message: str,
) -> None:
    with pytest.raises(ValueError, match=error_message):
        _validate_distill_models(student_model, teacher_model, quantization_mode=quantization_mode)


@pytest.mark.parametrize(
    ("config", "error_message"),
    [
        (lambda config: replace(config, eval_examples=0), "eval_examples must be at least 1"),
        (lambda config: replace(config, max_sequence_length=1), "max_sequence_length >= 2"),
    ],
)
def test_distill_rejects_invalid_config(
    tmp_path: Path,
    config: Callable[[DistillConfig], DistillConfig],
    error_message: str,
) -> None:
    with pytest.raises(ValueError, match=error_message):
        distill(config(_make_config(tmp_path)))


def test_distill_restores_best_in_memory_state_when_not_saving_checkpoints(tmp_path: Path) -> None:
    config = replace(_make_config(tmp_path), num_steps=2, eval_every_steps=1, save_checkpoints=False)
    teacher_model = _make_language_model()
    student_model = _make_language_model()
    batch = DistillBatch(
        token_ids=jnp.asarray([[0]], dtype=jnp.int32),
        lengths_without_padding=jnp.asarray([1], dtype=jnp.int32),
    )
    initial_state = DistillTrainingState(
        master_weights={"weight": jnp.asarray(0.0, dtype=jnp.float32)},
        muon_weight_dimension_numbers={"weight": None},
    )
    exported_students: list[object] = []
    train_states = [
        DistillTrainingState(
            master_weights={"weight": jnp.asarray(1.0, dtype=jnp.float32)},
            muon_weight_dimension_numbers={"weight": None},
        ),
        DistillTrainingState(
            master_weights={"weight": jnp.asarray(2.0, dtype=jnp.float32)},
            muon_weight_dimension_numbers={"weight": None},
        ),
    ]

    def accumulate_train_step(
        optimizer_state: DistillOptimizerState,
        optimizer: optax.GradientTransformation,
        microbatches: object,
        train_key: object,
        *,
        stochastic_rounding: bool,
        compute_step_gradients: object,
    ) -> tuple[DistillOptimizerState, DistillBatchMetrics, object]:
        del optimizer, microbatches, stochastic_rounding, compute_step_gradients
        next_training_state = train_states.pop(0)
        return (
            DistillOptimizerState(
                training_state=next_training_state,
                optimizer_state=optimizer_state.optimizer_state,
            ),
            DistillBatchMetrics(
                loss=jnp.asarray(0.1, dtype=jnp.float32),
                valid_tokens=jnp.asarray(1, dtype=jnp.int32),
                top1_matches=jnp.asarray(0, dtype=jnp.int32),
            ),
            train_key,
        )

    def materialize_trainable_module(
        module: object,
        master_weights: object,
        config: object,
    ) -> object:
        del module, config
        return master_weights

    def compute_distill_batch_metrics(
        student: dict[str, jax.Array],
        teacher: object,
        batch: object,
    ) -> DistillBatchMetrics:
        del teacher, batch
        losses = {0.0: 1.0, 1.0: 0.5, 2.0: 1.5}
        return DistillBatchMetrics(
            loss=jnp.asarray(losses[float(student["weight"])], dtype=jnp.float32),
            valid_tokens=jnp.asarray(1, dtype=jnp.int32),
            top1_matches=jnp.asarray(0, dtype=jnp.int32),
        )

    with (
        patch("lalamo.distill_runner.LanguageModelConfig.load_model", side_effect=[teacher_model, student_model]),
        patch(
            "lalamo.distill_runner._load_distill_batches",
            return_value=LoadedDistillBatches(
                train_examples=1,
                eval_examples=1,
                train_batches=[batch],
                eval_batches=[batch],
            ),
        ),
        patch("lalamo.distill_runner.initialize_distill_training_state", return_value=initial_state),
        patch("lalamo.distill_runner.iter_parameter_leaves", return_value=[]),
        patch("lalamo.distill_runner.summarize_distill_parameters", return_value=_EMPTY_PARAMETER_SUMMARY),
        patch("lalamo.distill_runner.materialize_trainable_module", side_effect=materialize_trainable_module),
        patch("lalamo.distill_runner.compute_distill_batch_metrics", side_effect=compute_distill_batch_metrics),
        patch("lalamo.distill_runner._accumulate_train_step", side_effect=accumulate_train_step),
        patch(
            "lalamo.distill_runner._save_materialized_student",
            side_effect=lambda *_args: exported_students.append(_args[-1]),
        ),
    ):
        result = distill(config)

    assert float(exported_students[0]["weight"]) == 1.0
    assert result.best_step == 1
    assert result.final_eval.kl_divergence == 0.5


def test_distill_uses_final_eval_as_best_step_without_periodic_eval(tmp_path: Path) -> None:
    config = replace(_make_config(tmp_path), num_steps=2, eval_every_steps=0, save_checkpoints=True)
    teacher_model = _make_language_model()
    student_model = _make_language_model()
    batch = DistillBatch(
        token_ids=jnp.asarray([[0, 1]], dtype=jnp.int32),
        lengths_without_padding=jnp.asarray([2], dtype=jnp.int32),
    )
    initial_state = DistillTrainingState(
        master_weights={"weight": jnp.asarray(0.0, dtype=jnp.float32)},
        muon_weight_dimension_numbers={"weight": None},
    )
    final_state = DistillTrainingState(
        master_weights={"weight": jnp.asarray(2.0, dtype=jnp.float32)},
        muon_weight_dimension_numbers={"weight": None},
    )
    saved_checkpoints: list[tuple[Path, object]] = []

    def accumulate_train_step(
        optimizer_state: DistillOptimizerState,
        optimizer: optax.GradientTransformation,
        microbatches: object,
        train_key: object,
        *,
        stochastic_rounding: bool,
        compute_step_gradients: object,
    ) -> tuple[DistillOptimizerState, DistillBatchMetrics, object]:
        del optimizer, microbatches, train_key, stochastic_rounding, compute_step_gradients
        return (
            DistillOptimizerState(
                training_state=final_state,
                optimizer_state=optimizer_state.optimizer_state,
            ),
            DistillBatchMetrics(
                loss=jnp.asarray(0.1, dtype=jnp.float32),
                valid_tokens=jnp.asarray(1, dtype=jnp.int32),
                top1_matches=jnp.asarray(0, dtype=jnp.int32),
            ),
            jax.random.key(1),
        )

    def materialize_trainable_module(module: object, master_weights: object, config: object) -> object:
        del module, config
        return master_weights

    def compute_distill_batch_metrics(
        student: dict[str, jax.Array],
        teacher: object,
        batch: object,
    ) -> DistillBatchMetrics:
        del teacher, batch
        losses = {0.0: 1.0, 2.0: 0.5}
        return DistillBatchMetrics(
            loss=jnp.asarray(losses[float(student["weight"])], dtype=jnp.float32),
            valid_tokens=jnp.asarray(1, dtype=jnp.int32),
            top1_matches=jnp.asarray(0, dtype=jnp.int32),
        )

    def save_checkpoint(checkpoint_dir: Path, checkpoint: object, metadata: object) -> None:
        del checkpoint
        saved_checkpoints.append((checkpoint_dir, metadata))

    with (
        patch("lalamo.distill_runner.LanguageModelConfig.load_model", side_effect=[teacher_model, student_model]),
        patch(
            "lalamo.distill_runner._load_distill_batches",
            return_value=LoadedDistillBatches(
                train_examples=1,
                eval_examples=1,
                train_batches=[batch],
                eval_batches=[batch],
            ),
        ),
        patch("lalamo.distill_runner.initialize_distill_training_state", return_value=initial_state),
        patch("lalamo.distill_runner.iter_parameter_leaves", return_value=[]),
        patch("lalamo.distill_runner.summarize_distill_parameters", return_value=_EMPTY_PARAMETER_SUMMARY),
        patch("lalamo.distill_runner._accumulate_train_step", side_effect=accumulate_train_step),
        patch("lalamo.distill_runner.materialize_trainable_module", side_effect=materialize_trainable_module),
        patch("lalamo.distill_runner.compute_distill_batch_metrics", side_effect=compute_distill_batch_metrics),
        patch("lalamo.distill_runner._save_checkpoint", side_effect=save_checkpoint),
        patch("lalamo.distill_runner._save_materialized_student"),
    ):
        result = distill(config)

    assert result.best_step == 2
    assert result.final_eval.kl_divergence == 0.5
    assert saved_checkpoints[-1][0].name == "best-checkpoint"
    assert saved_checkpoints[-1][1].best_step == 2
