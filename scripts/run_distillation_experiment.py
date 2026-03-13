from dataclasses import asdict, dataclass
from itertools import cycle, islice
import json
from pathlib import Path
import shutil
import time
from enum import StrEnum

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import typer
from jaxtyping import Array, Float

from lalamo.common import flatten_parameters
from lalamo.data import load_hf_parquet, shuffle_dataset
from lalamo.data.huggingface_message import HFMessage
from lalamo.distillation import (
    DistillBatch,
    DistillTrainConfig,
    compute_distill_kl_loss,
    distill_train_step,
    initialize_distill_optimizer_state,
    initialize_distill_training_state,
    materialize_trainable_module,
    summarize_distill_parameters,
)
from lalamo.models import LanguageModel, LanguageModelConfig
from lalamo.modules.common import iter_parameter_leaves
from lalamo.modules.decoder import Decoder
from lalamo.safetensors import safe_write


@dataclass(frozen=True)
class EvaluationMetrics:
    kl_divergence: float
    top1_agreement: float
    valid_tokens: int


@dataclass(frozen=True)
class ExperimentResult:
    teacher_path: str
    student_path: str
    dataset_path: str
    train_examples: int
    eval_examples: int
    max_sequence_length: int
    batch_size: int
    num_steps: int
    learning_rate: float
    optimizer: str
    distill_config: dict[str, Any]
    parameter_summary: dict[str, Any]
    initial_eval: dict[str, Any]
    final_eval: dict[str, Any]
    elapsed_seconds: float
    seconds_per_step: float
    output_model_path: str


class OptimizerName(StrEnum):
    ADAMW = "adamw"
    SGD = "sgd"


def _load_conversations(dataset_path: Path, *, num_examples: int, seed: int) -> list[list[HFMessage]]:
    dataframe = shuffle_dataset(load_hf_parquet(dataset_path), seed=seed)
    if "conversation" in dataframe.columns:
        column_name = "conversation"
    elif "messages" in dataframe.columns:
        column_name = "messages"
    else:
        raise ValueError(f"{dataset_path} must contain a 'conversation' or 'messages' column")

    conversations = dataframe.get_column(column_name).to_list()[:num_examples]
    return [
        [HFMessage.from_dict(message) for message in conversation]
        for conversation in conversations
    ]


def _tokenize_conversations(
    conversations: list[list[HFMessage]],
    *,
    language_model: LanguageModel,
    max_sequence_length: int,
) -> list[jax.Array]:
    tokenized_sequences: list[jax.Array] = []

    for conversation in conversations:
        messages = [message.as_message() for message in conversation]
        token_ids = language_model.message_processor.tokenize_request(messages)
        if len(token_ids) < 2:
            continue
        tokenized_sequences.append(jnp.array(token_ids[:max_sequence_length], dtype=jnp.int32))

    return tokenized_sequences


def _make_batches(sequences: list[jax.Array], *, batch_size: int) -> list[DistillBatch]:
    batches: list[DistillBatch] = []
    for start in range(0, len(sequences), batch_size):
        batch_sequences = sequences[start : start + batch_size]
        if not batch_sequences:
            continue

        max_length = max(sequence.shape[0] for sequence in batch_sequences)
        padded = jnp.zeros((len(batch_sequences), max_length), dtype=jnp.int32)
        lengths = jnp.array([sequence.shape[0] for sequence in batch_sequences], dtype=jnp.int32)

        for row, sequence in enumerate(batch_sequences):
            padded = padded.at[row, : sequence.shape[0]].set(sequence)

        batches.append(DistillBatch(token_ids=padded, lengths_without_padding=lengths))

    return batches


def _decoder_logits(
    decoder: Decoder,
    batch: DistillBatch,
) -> Float[Array, "batch prediction_tokens vocabulary"]:
    _, num_tokens = batch.token_ids.shape
    token_positions = jnp.broadcast_to(jnp.arange(num_tokens, dtype=jnp.int32), batch.token_ids.shape)
    decoder_result = decoder(
        token_ids=batch.token_ids,
        token_positions=token_positions,
    )
    return decoder_result.logits[:, :-1, :].astype(jnp.float32)


def _prediction_mask(batch: DistillBatch) -> jax.Array:
    if batch.lengths_without_padding is None:
        _, num_tokens = batch.token_ids.shape
        lengths_without_padding = jnp.full((batch.token_ids.shape[0],), num_tokens, dtype=jnp.int32)
    else:
        lengths_without_padding = batch.lengths_without_padding

    prediction_tokens = max(batch.token_ids.shape[1] - 1, 0)
    return jnp.arange(prediction_tokens, dtype=jnp.int32)[None, :] < (lengths_without_padding[:, None] - 1)


def _evaluate(
    student: Decoder,
    teacher: Decoder,
    batches: list[DistillBatch],
) -> EvaluationMetrics:
    total_kl = 0.0
    total_valid_tokens = 0
    total_matches = 0

    for batch in batches:
        metrics = compute_distill_kl_loss(student, teacher, batch)
        student_logits = _decoder_logits(student, batch)
        teacher_logits = _decoder_logits(teacher, batch)
        prediction_mask = _prediction_mask(batch)

        student_top1 = jnp.argmax(student_logits, axis=-1)
        teacher_top1 = jnp.argmax(teacher_logits, axis=-1)
        total_matches += int(jnp.sum((student_top1 == teacher_top1) & prediction_mask))

        valid_tokens = int(metrics.valid_tokens)
        total_kl += float(metrics.loss) * valid_tokens
        total_valid_tokens += valid_tokens

    return EvaluationMetrics(
        kl_divergence=total_kl / total_valid_tokens,
        top1_agreement=total_matches / total_valid_tokens,
        valid_tokens=total_valid_tokens,
    )


def _save_materialized_student(
    student_path: Path,
    output_path: Path,
    materialized_student: LanguageModel,
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(student_path / "config.json", output_path / "config.json")
    shutil.copy2(student_path / "tokenizer.json", output_path / "tokenizer.json")

    with Path(output_path / "model.safetensors").open("wb") as fd:
        safe_write(fd, flatten_parameters(materialized_student.export_weights()))


def _build_optimizer(name: OptimizerName, learning_rate: float) -> optax.GradientTransformation:
    match name:
        case OptimizerName.ADAMW:
            return optax.adamw(learning_rate)
        case OptimizerName.SGD:
            return optax.sgd(learning_rate)


def main(
    teacher_path: Path = typer.Option(..., exists=True, dir_okay=True, file_okay=False),
    student_path: Path = typer.Option(..., exists=True, dir_okay=True, file_okay=False),
    dataset_path: Path = typer.Option(..., exists=True, dir_okay=False, file_okay=True),
    output_dir: Path = typer.Option(...),
    train_examples: int = typer.Option(256),
    eval_examples: int = typer.Option(64),
    max_sequence_length: int = typer.Option(256),
    batch_size: int = typer.Option(4),
    num_steps: int = typer.Option(25),
    learning_rate: float = typer.Option(3e-4),
    optimizer_name: OptimizerName = typer.Option(OptimizerName.ADAMW),
    seed: int = typer.Option(0),
    train_bias: bool = typer.Option(False),
    train_norm: bool = typer.Option(False),
    train_embedding: bool = typer.Option(False),
    train_quant_aux: bool = typer.Option(False),
    train_base_weight: bool = typer.Option(True),
) -> None:
    teacher_model = LanguageModelConfig.load_model(teacher_path)
    student_model = LanguageModelConfig.load_model(student_path)

    conversations = _load_conversations(
        dataset_path,
        num_examples=train_examples + eval_examples,
        seed=seed,
    )
    tokenized = _tokenize_conversations(
        conversations,
        language_model=student_model,
        max_sequence_length=max_sequence_length,
    )

    if len(tokenized) < train_examples + eval_examples:
        raise ValueError(
            f"Requested {train_examples + eval_examples} usable sequences, got {len(tokenized)} from {dataset_path}",
        )

    train_sequences = tokenized[:train_examples]
    eval_sequences = tokenized[train_examples : train_examples + eval_examples]
    train_batches = _make_batches(train_sequences, batch_size=batch_size)
    eval_batches = _make_batches(eval_sequences, batch_size=batch_size)

    distill_config = DistillTrainConfig(
        train_bias=train_bias,
        train_norm=train_norm,
        train_embedding=train_embedding,
        train_adapter=False,
        train_quant_aux=train_quant_aux,
        train_base_weight=train_base_weight,
        master_dtype=jnp.float32,
        compute_dtype=student_model.model.activation_precision,
    )
    training_state = initialize_distill_training_state(student_model.model, distill_config)
    optimizer = _build_optimizer(optimizer_name, learning_rate)
    optimizer_state = initialize_distill_optimizer_state(training_state, optimizer)
    parameter_summary = summarize_distill_parameters(
        iter_parameter_leaves(student_model.model),
        distill_config,
    )

    initial_student = materialize_trainable_module(student_model.model, optimizer_state.training_state, distill_config)
    initial_eval = _evaluate(initial_student, teacher_model.model, eval_batches)

    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"
    step_iterator = cycle(train_batches)
    start_time = time.perf_counter()

    with history_path.open("w") as history_file:
        for step, batch in enumerate(islice(step_iterator, num_steps), start=1):
            optimizer_state, metrics = distill_train_step(
                optimizer_state,
                optimizer,
                student_model.model,
                teacher_model.model,
                batch,
                distill_config,
            )
            history_file.write(
                json.dumps(
                    {
                        "step": step,
                        "train_kl_divergence": float(metrics.loss),
                        "valid_tokens": int(metrics.valid_tokens),
                    }
                )
                + "\n"
            )

    elapsed_seconds = time.perf_counter() - start_time
    final_student = materialize_trainable_module(student_model.model, optimizer_state.training_state, distill_config)
    final_eval = _evaluate(final_student, teacher_model.model, eval_batches)

    model_output_path = output_dir / "student-distilled"
    _save_materialized_student(
        student_path,
        model_output_path,
        eqx.tree_at(lambda model: model.model, student_model, final_student),
    )

    result = ExperimentResult(
        teacher_path=str(teacher_path),
        student_path=str(student_path),
        dataset_path=str(dataset_path),
        train_examples=len(train_sequences),
        eval_examples=len(eval_sequences),
        max_sequence_length=max_sequence_length,
        batch_size=batch_size,
        num_steps=num_steps,
        learning_rate=learning_rate,
        optimizer=optimizer_name.value,
        distill_config={
            "train_bias": distill_config.train_bias,
            "train_norm": distill_config.train_norm,
            "train_embedding": distill_config.train_embedding,
            "train_quant_aux": distill_config.train_quant_aux,
            "train_base_weight": distill_config.train_base_weight,
            "master_dtype": str(jnp.dtype(distill_config.master_dtype)),
            "compute_dtype": str(jnp.dtype(distill_config.compute_dtype)),
        },
        parameter_summary=asdict(parameter_summary),
        initial_eval=asdict(initial_eval),
        final_eval=asdict(final_eval),
        elapsed_seconds=elapsed_seconds,
        seconds_per_step=elapsed_seconds / num_steps,
        output_model_path=str(model_output_path),
    )

    with (output_dir / "metrics.json").open("w") as metrics_file:
        json.dump(asdict(result), metrics_file, indent=4)

    print(json.dumps(asdict(result), indent=4))


if __name__ == "__main__":
    typer.run(main)
