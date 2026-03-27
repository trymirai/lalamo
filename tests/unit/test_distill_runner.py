from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from lalamo.distill_runner import (
    CheckpointMetadata,
    DistillConfig,
    EvaluationMetrics,
    OptimizerName,
    _accumulate_train_step,
    _build_optimizer,
    distill,
)
from lalamo.distillation import (
    DistillBatchMetrics,
    DistillOptimizerState,
    DistillParameterSummary,
    DistillTrainingState,
)
from lalamo.quantization import QuantizationMode


class _CallableWeights(eqx.Module):
    leaf: jax.Array | optax.contrib.MuonDimensionNumbers | None

    def __call__(self, token_positions: object) -> object:
        return token_positions


@dataclass(frozen=True)
class _StubModelConfig:
    weight_quantization_mode: QuantizationMode


@dataclass(frozen=True)
class _StubLanguageModelConfig:
    model_config: _StubModelConfig
    message_processor_config: str = "default"


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


def _make_language_model(
    *,
    tokenizer_signature: str = "shared-tokenizer",
    quantization_mode: QuantizationMode = QuantizationMode.UINT4,
    message_processor_config: str = "default",
) -> SimpleNamespace:
    return SimpleNamespace(
        config=_StubLanguageModelConfig(
            model_config=_StubModelConfig(weight_quantization_mode=quantization_mode),
            message_processor_config=message_processor_config,
        ),
        model=SimpleNamespace(activation_precision=jnp.float32, vocab_size=32),
        message_processor=SimpleNamespace(
            tokenize_request=lambda _messages: [1, 2, 3],
            tokenizer=SimpleNamespace(to_str=lambda: tokenizer_signature),
        ),
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
) -> tuple[object, DistillBatchMetrics, object]:
    del optimizer, microbatches, stochastic_rounding, compute_step_gradients
    return (
        optimizer_state,
        DistillBatchMetrics(
            loss=jnp.asarray(0.1, dtype=jnp.float32),
            valid_tokens=jnp.asarray(2, dtype=jnp.int32),
            top1_matches=jnp.asarray(0, dtype=jnp.int32),
        ),
        train_key,
    )


def test_accumulate_train_step_handles_nested_gradient_pytrees() -> None:
    optimizer = optax.sgd(1.0)
    optimizer_state = DistillOptimizerState(
        training_state=DistillTrainingState(
            master_weights={"left": jnp.array([1.0, 2.0]), "right": None},
            muon_weight_dimension_numbers={"left": None, "right": None},
        ),
        optimizer_state=optimizer.init({"left": jnp.array([1.0, 2.0]), "right": None}),
    )

    def compute_step_gradients(
        _batch: int,
        _quantization_key: object,
    ) -> tuple[object, DistillBatchMetrics]:
        return (
            {"left": jnp.array([2.0, 4.0]), "right": None},
            DistillBatchMetrics(
                loss=jnp.asarray(0.5, dtype=jnp.float32),
                valid_tokens=jnp.asarray(2, dtype=jnp.int32),
                top1_matches=jnp.asarray(1, dtype=jnp.int32),
            ),
        )

    updated_state, metrics, _ = _accumulate_train_step(
        optimizer_state,
        optimizer,
        microbatches=(1, 2),
        train_key=jax.random.key(0),
        stochastic_rounding=False,
        compute_step_gradients=compute_step_gradients,
    )

    assert jnp.allclose(updated_state.training_state.master_weights["left"], jnp.array([-1.0, -2.0]))
    assert updated_state.training_state.master_weights["right"] is None
    assert metrics.valid_tokens == 4
    assert jnp.isclose(metrics.loss, 0.5)
    assert metrics.top1_matches == 2


def test_build_optimizer_muon_does_not_call_callable_training_state_tree() -> None:
    training_state = DistillTrainingState(
        master_weights=_CallableWeights(jnp.ones((2, 2), dtype=jnp.float32)),
        muon_weight_dimension_numbers=_CallableWeights(
            optax.contrib.MuonDimensionNumbers(reduction_axis=1, output_axis=0),
        ),
    )

    optimizer = _build_optimizer(
        OptimizerName.MUON,
        1e-3,
        training_state,
        warmup_steps=0,
        gradient_clip_norm=None,
    )
    optimizer_state = optimizer.init(training_state.master_weights)

    assert optimizer_state is not None


def test_distill_skips_best_checkpoint_restore_without_periodic_eval(tmp_path: Path) -> None:
    config = _make_config(tmp_path, eval_every_steps=0, save_checkpoints=True)
    training_state = DistillTrainingState(master_weights=(), muon_weight_dimension_numbers=())
    language_models = [_make_language_model(), _make_language_model()]

    def save_checkpoint(checkpoint_dir: Path, checkpoint: object, metadata: CheckpointMetadata) -> None:
        del checkpoint, metadata
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def fail_load_checkpoint(checkpoint_dir: Path, like: object) -> object:
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
            "lalamo.distill_runner._evaluate_with",
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
    training_state = DistillTrainingState(master_weights=(), muon_weight_dimension_numbers=())
    restored_paths: list[Path] = []
    language_models = [_make_language_model(), _make_language_model()]

    def save_checkpoint(checkpoint_dir: Path, checkpoint: object, metadata: CheckpointMetadata) -> None:
        del checkpoint, metadata
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def load_checkpoint(checkpoint_dir: Path, like: object) -> tuple[object, CheckpointMetadata]:
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
            "lalamo.distill_runner._evaluate_with",
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


def test_distill_rejects_mismatched_tokenizers(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    language_models = [
        _make_language_model(tokenizer_signature="teacher"),
        _make_language_model(tokenizer_signature="student"),
    ]

    with (
        patch("lalamo.distill_runner.LanguageModelConfig.load_model", side_effect=language_models),
        pytest.raises(ValueError, match="identical tokenizers"),
    ):
        distill(config)


def test_distill_rejects_quantization_mode_mismatch(tmp_path: Path) -> None:
    config = _make_config(tmp_path, quantization_mode=QuantizationMode.UINT4)
    language_models = [
        _make_language_model(quantization_mode=QuantizationMode.UINT8),
        _make_language_model(quantization_mode=QuantizationMode.UINT8),
    ]

    with (
        patch("lalamo.distill_runner.LanguageModelConfig.load_model", side_effect=language_models),
        pytest.raises(ValueError, match="does not match the student model quantization mode"),
    ):
        distill(config)


def test_distill_uses_later_usable_sequences(tmp_path: Path) -> None:
    config = _make_config(tmp_path, train_examples=2, eval_examples=1)
    training_state = DistillTrainingState(master_weights=(), muon_weight_dimension_numbers=())
    language_models = [_make_language_model(), _make_language_model()]
    tokenized_sequences = [
        np.array([1], dtype=np.int32),
        np.array([1, 2, 3], dtype=np.int32),
        np.array([4, 5, 6], dtype=np.int32),
        np.array([7, 8, 9], dtype=np.int32),
    ]

    def load_conversations(_dataset_path: Path, *, num_examples: int | None, seed: int) -> list[object]:
        del seed
        assert num_examples is None
        return [object()] * 4

    with (
        patch("lalamo.distill_runner.LanguageModelConfig.load_model", side_effect=language_models),
        patch("lalamo.distill_runner._load_conversations", side_effect=load_conversations),
        patch("lalamo.distill_runner._tokenize_conversations", return_value=tokenized_sequences),
        patch("lalamo.distill_runner.initialize_distill_training_state", return_value=training_state),
        patch("lalamo.distill_runner.iter_parameter_leaves", return_value=[]),
        patch("lalamo.distill_runner.summarize_distill_parameters", return_value=_make_parameter_summary()),
        patch("lalamo.distill_runner.materialize_trainable_module", return_value=SimpleNamespace()),
        patch(
            "lalamo.distill_runner._evaluate_with",
            side_effect=[
                EvaluationMetrics(kl_divergence=1.0, top1_agreement=0.0, valid_tokens=1),
                EvaluationMetrics(kl_divergence=0.5, top1_agreement=0.0, valid_tokens=1),
            ],
        ),
        patch("lalamo.distill_runner._accumulate_train_step", side_effect=_fake_accumulate_train_step),
        patch("lalamo.distill_runner._save_materialized_student"),
    ):
        result = distill(config)

    assert result.train_examples == 2
    assert result.eval_examples == 1
