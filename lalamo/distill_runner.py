import json
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Key

from lalamo.common import flatten_parameters
from lalamo.data import load_hf_parquet, shuffle_dataset
from lalamo.data.huggingface_message import HFMessage
from lalamo.distillation import (
    DistillBatch,
    DistillBatchMetrics,
    DistillOptimizerState,
    DistillTrainConfig,
    DistillTrainingState,
    apply_distill_gradients,
    apply_donated_distill_gradients,
    compute_distill_batch_metrics,
    compute_distill_step_gradients,
    initialize_distill_training_state,
    inject_lora_adapter_configs,
    inject_lora_adapters,
    lora_trainable_filter,
    materialize_trainable_module,
)
from lalamo.model_import import ModelMetadata
from lalamo.models.language_model import LanguageModel, LanguageModelConfig
from lalamo.modules.common import (
    ParameterLeafInfo,
    ShardingConfig,
    config_converter,
    iter_parameter_leaves,
    pad_and_apply_data_sharding,
)
from lalamo.modules.decoder import Decoder
from lalamo.quantization import QuantizationMode
from lalamo.safetensors import safe_write

__all__ = [
    "ComputeDTypeName",
    "DistillConfig",
    "DistillResult",
    "EvaluationMetrics",
    "OptimizerName",
    "distill",
]


class OptimizerName(StrEnum):
    ADAMW = "adamw"
    MUON = "muon"


class ComputeDTypeName(StrEnum):
    AUTO = "auto"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"


@dataclass(frozen=True)
class EvaluationMetrics:
    kl_divergence: float
    top1_agreement: float
    valid_tokens: int


@dataclass(frozen=True)
class LoadedDistillBatches:
    train_batches: tuple[DistillBatch, ...]
    eval_batches: tuple[DistillBatch, ...]


@dataclass(frozen=True)
class DistillConfig:
    teacher_path: Path
    student_path: Path
    dataset_path: Path
    output_dir: Path
    train_examples: int = 256
    eval_examples: int = 64
    max_sequence_length: int = 256
    batch_size: int = 4
    num_steps: int = 25
    learning_rate: float = 3e-7
    warmup_steps: int = 0
    gradient_clip_norm: float | None = None
    gradient_accumulation_steps: int = 1
    optimizer_name: OptimizerName = OptimizerName.MUON
    quantization_mode: QuantizationMode = QuantizationMode.UINT4
    lora_rank: int | None = None
    lora_scale: float = 1.0
    compute_dtype_name: ComputeDTypeName = ComputeDTypeName.AUTO
    seed: int = 0
    stochastic_rounding: bool = True


@dataclass(frozen=True)
class DistillResult:
    completed_steps: int
    initial_eval: EvaluationMetrics
    final_eval: EvaluationMetrics
    compilation_seconds: float
    elapsed_seconds: float
    seconds_per_step: float
    output_model_path: Path


def _load_tokenized_conversations(
    dataset_path: Path,
    *,
    seed: int,
    language_model: LanguageModel,
    max_sequence_length: int,
    num_examples: int,
) -> list[np.ndarray]:
    dataframe = shuffle_dataset(load_hf_parquet(dataset_path), seed=seed)
    if "conversation" not in dataframe.columns:
        raise ValueError(f"{dataset_path} must contain a 'conversation' column")

    conversations = dataframe.get_column("conversation").to_list()
    all_token_ids = language_model.message_processor.tokenize_requests(
        [HFMessage.from_dict(message).as_message() for message in conversation] for conversation in conversations
    )
    tokenized_sequences = [
        np.array(ids[:max_sequence_length], dtype=np.int32)
        for ids in all_token_ids
        if len(ids[:max_sequence_length]) >= 2
    ]
    return tokenized_sequences[:num_examples]


def _same_tokenizers(
    student_model: LanguageModel,
    teacher_model: LanguageModel,
) -> bool:
    return (
        student_model.config.message_processor_config == teacher_model.config.message_processor_config
        and student_model.message_processor.tokenizer.to_str() == teacher_model.message_processor.tokenizer.to_str()
        and student_model.model.vocab_size == teacher_model.model.vocab_size
    )


def _single_quantization_mode(module: eqx.Module) -> QuantizationMode | None:
    quantization_modes = {
        info.quantization_mode for info in iter_parameter_leaves(module) if info.quantization_mode is not None
    }
    if not quantization_modes:
        return None
    if len(quantization_modes) > 1:
        raise ValueError(
            "Distillation requires a single quantization bitness per student model;"
            f" got {sorted(mode.value for mode in quantization_modes)}",
        )
    return next(iter(quantization_modes))


def _validate_distill_models(
    student_model: LanguageModel,
    teacher_model: LanguageModel,
    *,
    quantization_mode: QuantizationMode,
    stochastic_rounding: bool,
) -> None:
    if not _same_tokenizers(student_model, teacher_model):
        raise ValueError("Teacher and student must use identical tokenizers and message processor configs")

    if not stochastic_rounding:
        return

    student_quantization_mode = _single_quantization_mode(student_model.model)
    if student_quantization_mode is None:
        raise ValueError("Stochastic rounding requires a quantized student model")
    if quantization_mode != student_quantization_mode:
        raise ValueError(
            f"Requested stochastic rounding mode {quantization_mode.value} does not match the student model"
            f" quantization mode {student_quantization_mode.value}",
        )


def _make_batches(
    sequences: list[np.ndarray],
    *,
    batch_size: int,
    fixed_sequence_length: int,
) -> tuple[DistillBatch, ...]:
    batches: list[DistillBatch] = []
    for start in range(0, len(sequences), batch_size):
        batch_sequences = sequences[start : start + batch_size]

        padded = np.zeros((len(batch_sequences), fixed_sequence_length), dtype=np.int32)
        lengths = np.array([seq.shape[0] for seq in batch_sequences], dtype=np.int32)

        for row, seq in enumerate(batch_sequences):
            padded[row, : seq.shape[0]] = seq

        batches.append(DistillBatch(token_ids=jnp.array(padded), lengths_without_padding=jnp.array(lengths)))

    return tuple(batches)


def _load_distill_batches(
    config: DistillConfig,
    student_model: LanguageModel,
    *,
    device_batch_size: int,
) -> LoadedDistillBatches:
    requested_examples = config.train_examples + config.eval_examples
    tokenized = _load_tokenized_conversations(
        config.dataset_path,
        seed=config.seed,
        language_model=student_model,
        max_sequence_length=config.max_sequence_length,
        num_examples=requested_examples,
    )
    if len(tokenized) < requested_examples:
        raise ValueError(
            f"Requested {requested_examples} usable sequences, got {len(tokenized)} from {config.dataset_path}",
        )
    return LoadedDistillBatches(
        train_batches=_make_batches(
            tokenized[: config.train_examples],
            batch_size=device_batch_size,
            fixed_sequence_length=config.max_sequence_length,
        ),
        eval_batches=_make_batches(
            tokenized[config.train_examples : requested_examples],
            batch_size=device_batch_size,
            fixed_sequence_length=config.max_sequence_length,
        ),
    )


def _evaluate_with(
    student: Decoder,
    teacher: Decoder,
    batches: Sequence[DistillBatch],
) -> EvaluationMetrics:
    total_kl = 0.0
    total_valid_tokens = 0
    total_matches = 0

    for batch in batches:
        batch_metrics = compute_distill_batch_metrics(student, teacher, batch)
        total_kl += float(batch_metrics.loss) * int(batch_metrics.valid_tokens)
        total_valid_tokens += int(batch_metrics.valid_tokens)
        total_matches += int(batch_metrics.top1_matches)

    assert total_valid_tokens > 0, "Evaluation requires at least one valid token"
    return EvaluationMetrics(
        kl_divergence=total_kl / total_valid_tokens,
        top1_agreement=total_matches / total_valid_tokens,
        valid_tokens=total_valid_tokens,
    )


def _build_lora_student_model(
    student_model: LanguageModel,
    lora_rank: int,
    lora_scale: float,
    *,
    key: Key[Array, ""],
) -> LanguageModel:
    return replace(
        student_model,
        config=replace(
            student_model.config,
            model_config=inject_lora_adapter_configs(student_model.config.model_config, lora_rank, lora_scale),
        ),
        model=inject_lora_adapters(student_model.model, lora_rank, lora_scale, key),
    )


def _save_materialized_student(
    student_path: Path,
    output_path: Path,
    student_model: LanguageModel,
    materialized_student: Decoder,
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)

    with (student_path / "config.json").open() as config_file:
        source_config_json = json.load(config_file)
    if not isinstance(source_config_json, dict):
        raise TypeError(f"{student_path / 'config.json'} must contain a JSON object")

    updated_model = replace(student_model, model=materialized_student)
    metadata = config_converter.structure(source_config_json, ModelMetadata)
    updated_metadata = replace(metadata, model_config=updated_model.config)

    for filename in (
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ):
        source = student_path / filename
        if source.exists():
            shutil.copy2(source, output_path / filename)

    with (output_path / "config.json").open("w") as config_file:
        json.dump(
            config_converter.unstructure(updated_metadata, ModelMetadata),
            config_file,
            indent=4,
        )
    with (output_path / "model.safetensors").open("wb") as weights_file:
        safe_write(weights_file, flatten_parameters(updated_model.export_weights()))


def _build_optimizer(
    name: OptimizerName,
    learning_rate: float,
    training_state: DistillTrainingState,
    *,
    warmup_steps: int,
    gradient_clip_norm: float | None,
) -> optax.GradientTransformation:
    learning_rate_schedule = (
        learning_rate if warmup_steps <= 0 else optax.linear_schedule(0.0, learning_rate, warmup_steps)
    )

    match name:
        case OptimizerName.ADAMW:
            optimizer = optax.adamw(learning_rate_schedule)
        case OptimizerName.MUON:
            optimizer = optax.contrib.muon(
                learning_rate=learning_rate_schedule,
                muon_weight_dimension_numbers=lambda _params: training_state.muon_weight_dimension_numbers,
            )

    if gradient_clip_norm is None:
        return optimizer
    return optax.chain(optax.clip_by_global_norm(gradient_clip_norm), optimizer)


def _accumulate_train_step(
    optimizer_state: DistillOptimizerState,
    optimizer: optax.GradientTransformation,
    microbatches: Sequence[DistillBatch],
    train_key: Key[Array, ""],
    *,
    optimizer_name: OptimizerName,
    student: Decoder,
    teacher: Decoder,
    distill_config: DistillTrainConfig,
    quantization_mode: QuantizationMode,
    stochastic_rounding: bool,
) -> tuple[DistillOptimizerState, DistillBatchMetrics, Key[Array, ""]]:
    assert microbatches, "Gradient accumulation requires at least one microbatch"
    accumulated_grads: object | None = None
    total_loss = jnp.asarray(0.0, dtype=jnp.float32)
    total_valid_tokens = jnp.asarray(0, dtype=jnp.int32)
    total_top1_matches = jnp.asarray(0, dtype=jnp.int32)

    for microbatch in microbatches:
        train_key, step_key = jax.random.split(train_key)
        grads, metrics = compute_distill_step_gradients(
            optimizer_state.training_state,
            student,
            teacher,
            microbatch,
            distill_config,
            quantization_mode,
            stochastic_rounding=stochastic_rounding,
            quantization_key=step_key,
        )
        valid_tokens = metrics.valid_tokens
        weighted_grads = jax.tree.map(lambda grad, scale=valid_tokens: grad * scale, grads)
        accumulated_grads = (
            weighted_grads
            if accumulated_grads is None
            else jax.tree.map(lambda total, grad: total + grad, accumulated_grads, weighted_grads)
        )
        total_loss = total_loss + metrics.loss * valid_tokens.astype(jnp.float32)
        total_valid_tokens = total_valid_tokens + valid_tokens
        total_top1_matches = total_top1_matches + metrics.top1_matches

    assert accumulated_grads is not None
    averaged_grads = jax.tree.map(lambda grad: grad / total_valid_tokens, accumulated_grads)
    step_metrics = DistillBatchMetrics(
        loss=total_loss / total_valid_tokens.astype(jnp.float32),
        valid_tokens=total_valid_tokens,
        top1_matches=total_top1_matches,
    )
    apply_gradients = (
        apply_distill_gradients if optimizer_name == OptimizerName.MUON else apply_donated_distill_gradients
    )
    return apply_gradients(optimizer_state, optimizer, averaged_grads), step_metrics, train_key


def _config_json(config: DistillConfig) -> dict[str, object]:
    result = asdict(config)
    for key in ("teacher_path", "student_path", "dataset_path", "output_dir"):
        result[key] = str(result[key])
    result["optimizer_name"] = config.optimizer_name.value
    result["quantization_mode"] = config.quantization_mode.value
    result["compute_dtype_name"] = config.compute_dtype_name.value
    return result


def _metrics_json(metrics: EvaluationMetrics) -> dict[str, float | int]:
    return {
        "kl_divergence": metrics.kl_divergence,
        "top1_agreement": metrics.top1_agreement,
        "valid_tokens": metrics.valid_tokens,
    }


def _validate_config(config: DistillConfig) -> None:
    if config.train_examples < 1:
        raise ValueError("train_examples must be at least 1")
    if config.eval_examples < 1:
        raise ValueError("eval_examples must be at least 1")
    if config.max_sequence_length < 2:
        raise ValueError("Online distillation requires max_sequence_length >= 2")
    if config.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if config.num_steps < 1:
        raise ValueError("num_steps must be at least 1")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if config.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    if config.gradient_clip_norm is not None and config.gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive when provided")
    if config.lora_rank is not None and config.lora_rank < 1:
        raise ValueError("lora_rank must be positive when provided")
    if config.lora_scale <= 0:
        raise ValueError("lora_scale must be positive")


def distill(
    config: DistillConfig,
    *,
    trainable_filter: Callable[[ParameterLeafInfo], bool] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> DistillResult:
    _validate_config(config)

    sharding_config = ShardingConfig.build() if jax.device_count() > 1 else None
    device_batch_size = config.batch_size * jax.device_count()

    teacher_model = LanguageModelConfig.load_model(config.teacher_path)
    student_model = LanguageModelConfig.load_model(config.student_path)

    effective_trainable_filter = trainable_filter
    if config.lora_rank is not None:
        student_model = _build_lora_student_model(
            student_model,
            config.lora_rank,
            config.lora_scale,
            key=jax.random.key(config.seed),
        )
        if effective_trainable_filter is None:
            effective_trainable_filter = lora_trainable_filter

    _validate_distill_models(
        student_model,
        teacher_model,
        quantization_mode=config.quantization_mode,
        stochastic_rounding=config.stochastic_rounding,
    )

    dataset = _load_distill_batches(
        config,
        student_model,
        device_batch_size=device_batch_size,
    )

    def shard(batch: DistillBatch) -> DistillBatch:
        return pad_and_apply_data_sharding(batch, sharding_config=sharding_config, batch_axis=0)

    dataset = replace(
        dataset,
        train_batches=tuple(shard(batch) for batch in dataset.train_batches),
        eval_batches=tuple(shard(batch) for batch in dataset.eval_batches),
    )

    match config.compute_dtype_name:
        case ComputeDTypeName.AUTO:
            compute_dtype = student_model.model.activation_precision
        case ComputeDTypeName.BFLOAT16:
            compute_dtype = jnp.bfloat16
        case ComputeDTypeName.FLOAT32:
            compute_dtype = jnp.float32

    distill_config = DistillTrainConfig(compute_dtype=compute_dtype)
    training_state = initialize_distill_training_state(
        student_model.model,
        distill_config,
        trainable_filter=effective_trainable_filter,
    )
    optimizer = _build_optimizer(
        config.optimizer_name,
        config.learning_rate,
        training_state,
        warmup_steps=config.warmup_steps,
        gradient_clip_norm=config.gradient_clip_norm,
    )
    optimizer_state = DistillOptimizerState(
        training_state=training_state,
        optimizer_state=optimizer.init(training_state.master_weights),
    )

    if sharding_config is not None:
        replicated_sharding = sharding_config.make_sharding(jax.sharding.PartitionSpec())
        student_model = eqx.filter_shard(student_model, replicated_sharding)
        teacher_model = eqx.filter_shard(teacher_model, replicated_sharding)
        optimizer_state = eqx.filter_shard(optimizer_state, replicated_sharding)

    train_key = jax.random.key(config.seed)
    run_start = time.perf_counter()
    initial_student = materialize_trainable_module(
        student_model.model,
        optimizer_state.training_state.master_weights,
        distill_config,
    )
    initial_eval = _evaluate_with(initial_student, teacher_model.model, dataset.eval_batches)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = config.output_dir / "history.jsonl"
    next_microbatch_index = 0
    compilation_seconds = 0.0
    executed_steps = 0

    with history_path.open("w") as history_file:
        for step in range(1, config.num_steps + 1):
            microbatches = tuple(
                dataset.train_batches[(next_microbatch_index + offset) % len(dataset.train_batches)]
                for offset in range(config.gradient_accumulation_steps)
            )
            next_microbatch_index += config.gradient_accumulation_steps

            if executed_steps == 0:
                warmup_start = time.perf_counter()

            optimizer_state, train_metrics, train_key = _accumulate_train_step(
                optimizer_state,
                optimizer,
                microbatches,
                train_key,
                optimizer_name=config.optimizer_name,
                student=student_model.model,
                teacher=teacher_model.model,
                distill_config=distill_config,
                quantization_mode=config.quantization_mode,
                stochastic_rounding=config.stochastic_rounding,
            )

            if executed_steps == 0:
                compilation_seconds = time.perf_counter() - warmup_start

            history_file.write(
                json.dumps(
                    {
                        "step": step,
                        "train_kl_divergence": float(train_metrics.loss),
                        "valid_tokens": int(train_metrics.valid_tokens),
                    }
                )
                + "\n"
            )
            history_file.flush()
            executed_steps += 1
            if progress_callback is not None:
                progress_callback(step, config.num_steps)

    elapsed_seconds = time.perf_counter() - run_start
    measured_steps = max(executed_steps - 1, 0)

    final_student = materialize_trainable_module(
        student_model.model,
        optimizer_state.training_state.master_weights,
        distill_config,
    )
    final_eval = _evaluate_with(final_student, teacher_model.model, dataset.eval_batches)

    model_output_path = config.output_dir / "student-distilled"
    export_config = DistillTrainConfig(compute_dtype=student_model.model.activation_precision)
    export_student = materialize_trainable_module(
        student_model.model,
        optimizer_state.training_state.master_weights,
        export_config,
    )
    _save_materialized_student(
        config.student_path,
        model_output_path,
        student_model,
        export_student,
    )

    result = DistillResult(
        completed_steps=config.num_steps,
        initial_eval=initial_eval,
        final_eval=final_eval,
        compilation_seconds=compilation_seconds,
        elapsed_seconds=elapsed_seconds,
        seconds_per_step=0.0 if measured_steps == 0 else elapsed_seconds / measured_steps,
        output_model_path=model_output_path,
    )

    with (config.output_dir / "metrics.json").open("w") as metrics_file:
        json.dump(
            {
                "config": _config_json(config),
                "result": {
                    "completed_steps": result.completed_steps,
                    "initial_eval": _metrics_json(result.initial_eval),
                    "final_eval": _metrics_json(result.final_eval),
                    "compilation_seconds": result.compilation_seconds,
                    "elapsed_seconds": result.elapsed_seconds,
                    "seconds_per_step": result.seconds_per_step,
                    "output_model_path": str(result.output_model_path),
                },
            },
            metrics_file,
            indent=4,
        )

    return result
