from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import equinox as eqx
import jax
import jax.numpy as jnp
import polars as pl
import pytest

from lalamo.distill_runner import (
    DistillConfig,
    OptimizerName,
    _load_tokenized_conversations,
    _validate_distill_models,
    distill,
)
from lalamo.modules.common import field
from lalamo.quantization import QuantizationMode


@dataclass(frozen=True)
class _StubModelConfig:
    weight_quantization_mode: QuantizationMode

    @property
    def quantization(self) -> QuantizationMode:
        return self.weight_quantization_mode


@dataclass(frozen=True)
class _StubLanguageModelConfig:
    model_config: _StubModelConfig
    message_processor_config: str = "default"


class _StubQuantizedModel(eqx.Module):
    config: _StubModelConfig = eqx.field(static=True)
    weights: jax.Array = field(quantized=True)

    @property
    def activation_precision(self) -> jnp.dtype:
        return self.weights.dtype

    @property
    def vocab_size(self) -> int:
        return 32


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
        model=_StubQuantizedModel(
            config=_StubModelConfig(weight_quantization_mode=quantization_mode),
            weights=jnp.zeros((1, 1), dtype=jnp.float32),
        ),
        message_processor=SimpleNamespace(
            tokenize_requests=lambda conversations: [tokenize_request(msgs) for msgs in conversations],
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
