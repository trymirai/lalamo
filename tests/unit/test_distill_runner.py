from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import optax
import pytest

from lalamo.distill_runner import CheckpointMetadata, DistillConfig, EvaluationMetrics, OptimizerName, distill
from lalamo.distillation import DistillParameterSummary, DistillStepMetrics, DistillTrainingState
from lalamo.quantization import QuantizationMode


def _make_config(tmp_path: Path, **overrides: object) -> DistillConfig:
    config_arguments = {
        "teacher_path": tmp_path / "teacher",
        "student_path": tmp_path / "student",
        "dataset_path": tmp_path / "dataset.parquet",
        "output_dir": tmp_path / "output",
        "batch_size": 1,
        "train_examples": 1,
        "eval_examples": 1,
        "num_steps": 1,
        "optimizer_name": OptimizerName.SGD,
        "quantization_mode": QuantizationMode.UINT4,
        **overrides,
    }
    return DistillConfig(**config_arguments)


def _make_language_model() -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(activation_precision=jnp.float32),
        message_processor=SimpleNamespace(tokenize_request=lambda _messages: [1, 2, 3]),
    )


def _make_parameter_summary() -> DistillParameterSummary:
    return DistillParameterSummary(
        total_parameters=0,
        trainable_parameters=0,
        total_master_bytes=0,
        by_group={},
    )


def _fake_accumulate_train_step(
    optimizer_state: object,
    optimizer: optax.GradientTransformation,
    microbatches: object,
    train_key: object,
    *,
    stochastic_rounding: bool,
    compute_step_gradients: object,
) -> tuple[object, DistillStepMetrics, object]:
    del optimizer, microbatches, stochastic_rounding, compute_step_gradients
    return (
        optimizer_state,
        DistillStepMetrics(
            loss=jnp.asarray(0.1, dtype=jnp.float32),
            valid_tokens=jnp.asarray(2, dtype=jnp.int32),
        ),
        train_key,
    )


def test_distill_skips_best_checkpoint_restore_without_periodic_eval(tmp_path: Path) -> None:
    config = _make_config(tmp_path, eval_every_steps=0, save_checkpoints=True)
    training_state = DistillTrainingState(trainable_parameters=())
    language_models = [_make_language_model(), _make_language_model()]

    def save_checkpoint(checkpoint_dir: Path | str, checkpoint: object, metadata: CheckpointMetadata) -> None:
        del checkpoint, metadata
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def fail_load_checkpoint(checkpoint_dir: Path | str, like: object) -> object:
        del like
        raise AssertionError(f"Unexpected checkpoint restore from {checkpoint_dir}")

    with (
        patch(
            "lalamo.distill_runner.LanguageModelConfig.load_model",
            side_effect=language_models,
        ),
        patch("lalamo.distill_runner._load_conversations", return_value=[object(), object()]),
        patch(
            "lalamo.distill_runner._tokenize_conversations",
            return_value=[np.array([1, 2, 3], dtype=np.int32), np.array([1, 2, 3], dtype=np.int32)],
        ),
        patch("lalamo.distill_runner.initialize_distill_training_state", return_value=training_state),
        patch("lalamo.distill_runner.iter_parameter_leaves", return_value=[]),
        patch("lalamo.distill_runner.summarize_distill_parameters", return_value=_make_parameter_summary()),
        patch("lalamo.distill_runner.materialize_trainable_module", return_value=SimpleNamespace()),
        patch(
            "lalamo.distill_runner._evaluate",
            side_effect=[
                EvaluationMetrics(kl_divergence=1.0, top1_agreement=0.0, valid_tokens=1),
                EvaluationMetrics(kl_divergence=0.5, top1_agreement=0.0, valid_tokens=1),
            ],
        ),
        patch("lalamo.distill_runner._accumulate_train_step", side_effect=_fake_accumulate_train_step),
        patch("lalamo.distill_runner._save_checkpoint", side_effect=save_checkpoint),
        patch("lalamo.distill_runner._load_checkpoint", side_effect=fail_load_checkpoint),
        patch("lalamo.distill_runner._save_materialized_student"),
    ):
        result = distill(config)

    assert result.best_step == 0
    assert result.completed_steps == 1


def test_distill_restores_best_checkpoint_after_periodic_eval(tmp_path: Path) -> None:
    config = _make_config(tmp_path, eval_every_steps=1, save_checkpoints=True)
    training_state = DistillTrainingState(trainable_parameters=())
    restored_paths: list[Path] = []
    language_models = [_make_language_model(), _make_language_model()]

    def save_checkpoint(checkpoint_dir: Path | str, checkpoint: object, metadata: CheckpointMetadata) -> None:
        del checkpoint, metadata
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def load_checkpoint(checkpoint_dir: Path | str, like: object) -> tuple[object, CheckpointMetadata]:
        restored_paths.append(Path(checkpoint_dir))
        return like, CheckpointMetadata(step=0, best_eval_kl=1.0, best_step=0, evaluations_without_improvement=1)

    with (
        patch(
            "lalamo.distill_runner.LanguageModelConfig.load_model",
            side_effect=language_models,
        ),
        patch("lalamo.distill_runner._load_conversations", return_value=[object(), object()]),
        patch(
            "lalamo.distill_runner._tokenize_conversations",
            return_value=[np.array([1, 2, 3], dtype=np.int32), np.array([1, 2, 3], dtype=np.int32)],
        ),
        patch("lalamo.distill_runner.initialize_distill_training_state", return_value=training_state),
        patch("lalamo.distill_runner.iter_parameter_leaves", return_value=[]),
        patch("lalamo.distill_runner.summarize_distill_parameters", return_value=_make_parameter_summary()),
        patch("lalamo.distill_runner.materialize_trainable_module", return_value=SimpleNamespace()),
        patch(
            "lalamo.distill_runner._evaluate",
            side_effect=[
                EvaluationMetrics(kl_divergence=1.0, top1_agreement=0.0, valid_tokens=1),
                EvaluationMetrics(kl_divergence=2.0, top1_agreement=0.0, valid_tokens=1),
                EvaluationMetrics(kl_divergence=1.0, top1_agreement=0.0, valid_tokens=1),
            ],
        ),
        patch("lalamo.distill_runner._accumulate_train_step", side_effect=_fake_accumulate_train_step),
        patch("lalamo.distill_runner._save_checkpoint", side_effect=save_checkpoint),
        patch("lalamo.distill_runner._load_checkpoint", side_effect=load_checkpoint),
        patch("lalamo.distill_runner._save_materialized_student"),
    ):
        result = distill(config)

    assert result.best_step == 0
    assert restored_paths == [config.output_dir / "best-checkpoint"]


def test_distill_fails_when_sharding_filters_every_batch(tmp_path: Path) -> None:
    config = _make_config(tmp_path, num_devices=2, batch_size=1)
    language_models = [_make_language_model(), _make_language_model()]

    with (
        patch(
            "lalamo.distill_runner.LanguageModelConfig.load_model",
            side_effect=language_models,
        ),
        patch("lalamo.distill_runner._load_conversations", return_value=[object(), object()]),
        patch(
            "lalamo.distill_runner._tokenize_conversations",
            return_value=[np.array([1, 2, 3], dtype=np.int32), np.array([1, 2, 3], dtype=np.int32)],
        ),
        patch("lalamo.distill_runner._setup_device_mesh", return_value=object()),
        patch("lalamo.distill_runner.NamedSharding", return_value=None),
        pytest.raises(
            ValueError,
            match="No train batches remain after data sharding; use more train examples or fewer devices",
        ),
    ):
        distill(config)
