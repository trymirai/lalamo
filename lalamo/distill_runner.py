import dataclasses
import json
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from itertools import cycle, islice
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from cattrs import Converter
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Key

from lalamo.common import flatten_parameters
from lalamo.data import load_hf_parquet, shuffle_dataset
from lalamo.data.huggingface_message import HFMessage
from lalamo.data.lalamo_completions import LalamoCompletion
from lalamo.distillation import (
    DistillBatch,
    DistillBatchMetrics,
    DistillOptimizerState,
    DistillParameterSummary,
    DistillTrainConfig,
    DistillTrainingState,
    TraceDistillBatch,
    apply_distill_gradients,
    compute_distill_batch_metrics,
    compute_distill_step_gradients,
    compute_trace_distill_batch_metrics,
    compute_trace_distill_step_gradients,
    initialize_distill_training_state,
    inject_lora_adapter_configs,
    inject_lora_adapters,
    lora_trainable_filter,
    make_trace_distill_batch,
    materialize_trainable_module,
    summarize_distill_parameters,
)
from lalamo.model_import import ModelMetadata
from lalamo.models import LanguageModel, LanguageModelConfig
from lalamo.modules import config_converter
from lalamo.modules.common import ParameterLeafInfo, iter_parameter_leaves
from lalamo.modules.decoder import Decoder
from lalamo.quantization import QuantizationMode
from lalamo.safetensors import safe_write

__all__ = [
    "ComputeDTypeName",
    "DistillCallbacks",
    "DistillConfig",
    "DistillResult",
    "EvaluationMetrics",
    "OptimizerName",
    "TrainingMode",
    "distill",
]


_CHECKPOINT_CONVERTER = Converter()


class TrainingMode(StrEnum):
    ONLINE_EXACT = "online_exact"
    TRACE_TOPK = "trace_topk"


class OptimizerName(StrEnum):
    ADAMW = "adamw"
    MUON = "muon"
    SGD = "sgd"


class ComputeDTypeName(StrEnum):
    AUTO = "auto"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"


@dataclass(frozen=True)
class EvaluationMetrics:
    kl_divergence: float
    top1_agreement: float
    valid_tokens: int


type DistillBatchLike = DistillBatch | TraceDistillBatch


@dataclass(frozen=True)
class LoadedDistillBatches:
    train_batches: Sequence[DistillBatchLike]
    eval_batches: Sequence[DistillBatchLike]


@dataclass(frozen=True)
class CheckpointMetadata:
    step: int
    best_eval_kl: float | None
    best_step: int | None
    evaluations_without_improvement: int


@dataclass(frozen=True)
class DistillConfig:
    teacher_path: Path
    student_path: Path
    dataset_path: Path
    output_dir: Path
    training_mode: TrainingMode = TrainingMode.ONLINE_EXACT
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
    eval_every_steps: int = 0
    checkpoint_every_steps: int = 0
    early_stop_patience: int = 0
    resume_from: Path | None = None
    num_devices: int = 1
    seed: int = 0
    stochastic_rounding: bool = True
    save_checkpoints: bool = True


@dataclass(frozen=True)
class DistillResult:
    teacher_path: str
    student_path: str
    dataset_path: str
    training_mode: TrainingMode
    train_examples: int
    eval_examples: int
    max_sequence_length: int
    batch_size: int
    num_steps: int
    completed_steps: int
    learning_rate: float
    warmup_steps: int
    gradient_clip_norm: float | None
    gradient_accumulation_steps: int
    seed: int
    optimizer: OptimizerName
    quantization_mode: str
    master_dtype: str
    compute_dtype: str
    parameter_summary: DistillParameterSummary
    initial_eval: EvaluationMetrics
    final_eval: EvaluationMetrics
    best_step: int | None
    stopped_early: bool
    compilation_seconds: float
    elapsed_seconds: float
    seconds_per_step: float
    output_model_path: str


class ExperimentCheckpoint(eqx.Module):
    optimizer_state: DistillOptimizerState
    train_key_data: Array


@dataclass
class DistillCallbacks:
    config: DistillConfig

    def started(self) -> None:
        pass

    def loading_models(self) -> None:
        pass

    def finished_loading_models(self) -> None:
        pass

    def loading_dataset(self) -> None:
        pass

    def finished_loading_dataset(self) -> None:
        pass

    def distillation_progress(self, step: int, num_steps: int) -> None:
        pass

    def saving_model(self) -> None:
        pass

    def finished_saving_model(self, output_model_path: Path) -> None:
        pass

    def finished(self, result: DistillResult) -> None:
        pass


def _load_tokenized_conversations(
    dataset_path: Path,
    *,
    seed: int,
    language_model: LanguageModel,
    max_sequence_length: int,
    num_examples: int,
) -> list[np.ndarray]:
    dataframe = shuffle_dataset(load_hf_parquet(dataset_path), seed=seed)
    if "conversation" in dataframe.columns:
        column_name = "conversation"
    elif "messages" in dataframe.columns:
        column_name = "messages"
    else:
        raise ValueError(f"{dataset_path} must contain a 'conversation' or 'messages' column")

    tokenized_sequences: list[np.ndarray] = []
    for conversation in dataframe.get_column(column_name).to_list():
        token_ids = language_model.message_processor.tokenize_request(
            [HFMessage.from_dict(message).as_message() for message in conversation]
        )
        token_ids = token_ids[:max_sequence_length]
        if len(token_ids) < 2:
            continue
        tokenized_sequences.append(np.array(token_ids, dtype=np.int32))
        if len(tokenized_sequences) == num_examples:
            break
    return tokenized_sequences


def _load_traces(dataset_path: Path, *, num_examples: int) -> list[LalamoCompletion]:
    with dataset_path.open("rb") as trace_fd:
        return list(islice(LalamoCompletion.deserialize_many(trace_fd), num_examples))


def _has_tokenizer_compatibility(
    student_model: LanguageModel,
    teacher_model: LanguageModel,
) -> bool:
    student_tokenizer = student_model.message_processor.tokenizer
    teacher_tokenizer = teacher_model.message_processor.tokenizer
    student_vocab_size = student_tokenizer.get_vocab_size()
    teacher_vocab_size = teacher_tokenizer.get_vocab_size()
    return (
        student_tokenizer.to_str() == teacher_tokenizer.to_str()
        and student_vocab_size == teacher_vocab_size
        and student_model.model.vocab_size == student_vocab_size
        and teacher_model.model.vocab_size >= teacher_vocab_size
    )


def _iter_quantization_modes(config: object) -> set[QuantizationMode]:
    quantization_modes: set[QuantizationMode] = set()

    def visit(value: object) -> None:
        if dataclasses.is_dataclass(value):
            for config_field in dataclasses.fields(value):
                field_value = getattr(value, config_field.name)
                if config_field.name in {"weight_quantization_mode", "embedding_quantization_mode"}:
                    if isinstance(field_value, QuantizationMode):
                        quantization_modes.add(field_value)
                    continue
                visit(field_value)
            return
        if isinstance(value, tuple | list):
            for item in value:
                visit(item)
            return
        if isinstance(value, dict):
            for item in value.values():
                visit(item)

    visit(config)
    return quantization_modes


def _single_quantization_mode(config: object) -> QuantizationMode | None:
    quantization_modes = _iter_quantization_modes(config)
    if not quantization_modes:
        return None
    if len(quantization_modes) > 1:
        raise ValueError(
            "Distillation currently supports a single quantization bitness per student model;"
            f" got {sorted(mode.value for mode in quantization_modes)}",
        )
    return next(iter(quantization_modes))


def _validate_distill_models(
    student_model: LanguageModel,
    teacher_model: LanguageModel,
    *,
    quantization_mode: QuantizationMode,
) -> None:
    if not _has_tokenizer_compatibility(student_model, teacher_model):
        raise ValueError("Teacher and student must use identical tokenizers and message processor configs")

    student_quantization_mode = _single_quantization_mode(student_model.config.model_config)
    if student_quantization_mode is not None and quantization_mode != student_quantization_mode:
        raise ValueError(
            f"Requested stochastic rounding mode {quantization_mode.value} does not match the student model"
            f" quantization mode {student_quantization_mode.value}",
        )


def _make_batches(
    sequences: list[np.ndarray],
    *,
    batch_size: int,
    fixed_sequence_length: int,
) -> list[DistillBatch]:
    batches: list[DistillBatch] = []
    for start in range(0, len(sequences), batch_size):
        batch_sequences = sequences[start : start + batch_size]

        padded = np.zeros((len(batch_sequences), fixed_sequence_length), dtype=np.int32)
        lengths = np.array([seq.shape[0] for seq in batch_sequences], dtype=np.int32)

        for row, seq in enumerate(batch_sequences):
            padded[row, : seq.shape[0]] = seq

        batches.append(DistillBatch(token_ids=jnp.array(padded), lengths_without_padding=jnp.array(lengths)))

    return batches


def _load_distill_batches(
    config: DistillConfig,
    student_model: LanguageModel,
    *,
    device_batch_size: int,
) -> LoadedDistillBatches:
    requested_examples = config.train_examples + config.eval_examples
    match config.training_mode:
        case TrainingMode.ONLINE_EXACT:
            tokenized = _load_tokenized_conversations(
                config.dataset_path,
                seed=config.seed,
                language_model=student_model,
                max_sequence_length=config.max_sequence_length,
                num_examples=requested_examples,
            )
            if len(tokenized) < requested_examples:
                raise ValueError(
                    "Requested"
                    f" {requested_examples} usable sequences, got {len(tokenized)} from {config.dataset_path}",
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
        case TrainingMode.TRACE_TOPK:
            traces = _load_traces(config.dataset_path, num_examples=requested_examples)
            if len(traces) < requested_examples:
                raise ValueError(
                    f"Requested {requested_examples} traces, got {len(traces)} from {config.dataset_path}",
                )
            return LoadedDistillBatches(
                train_batches=[
                    make_trace_distill_batch(traces[start : start + device_batch_size], pad_token_id=0)
                    for start in range(0, config.train_examples, device_batch_size)
                ],
                eval_batches=[
                    make_trace_distill_batch(traces[start : start + device_batch_size], pad_token_id=0)
                    for start in range(config.train_examples, requested_examples, device_batch_size)
                ],
            )


def _shard_distill_batches(
    dataset: LoadedDistillBatches,
    mesh: Mesh | None,
    *,
    num_devices: int,
) -> LoadedDistillBatches:
    if mesh is None:
        return dataset

    batch_sharding = NamedSharding(mesh, P("batch"))
    for batch in [*dataset.train_batches, *dataset.eval_batches]:
        if batch.token_ids.shape[0] % num_devices == 0:
            continue
        raise ValueError(f"Batch size {batch.token_ids.shape[0]} is not divisible by num_devices={num_devices}")

    return replace(
        dataset,
        train_batches=[eqx.filter_shard(batch, batch_sharding) for batch in dataset.train_batches],
        eval_batches=[eqx.filter_shard(batch, batch_sharding) for batch in dataset.eval_batches],
    )


def _evaluate_with(
    batches: Sequence[DistillBatchLike],
    compute_batch_metrics: Callable[[DistillBatchLike], DistillBatchMetrics],
) -> EvaluationMetrics:
    total_kl = 0.0
    total_valid_tokens = 0
    total_matches = 0

    for batch in batches:
        batch_metrics = compute_batch_metrics(batch)
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


def _save_checkpoint(
    checkpoint_dir: Path,
    checkpoint: ExperimentCheckpoint,
    metadata: CheckpointMetadata,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(checkpoint_dir / "state.eqx", checkpoint)
    with (checkpoint_dir / "metadata.json").open("w") as metadata_file:
        json.dump(_CHECKPOINT_CONVERTER.unstructure(metadata), metadata_file, indent=4)


def _load_checkpoint(
    checkpoint_dir: Path,
    like: ExperimentCheckpoint,
) -> tuple[ExperimentCheckpoint, CheckpointMetadata]:
    checkpoint = eqx.tree_deserialise_leaves(checkpoint_dir / "state.eqx", like)
    with (checkpoint_dir / "metadata.json").open() as metadata_file:
        metadata = _CHECKPOINT_CONVERTER.structure(json.load(metadata_file), CheckpointMetadata)
    return checkpoint, metadata


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
        case OptimizerName.SGD:
            optimizer = optax.sgd(learning_rate_schedule)
        case _:
            raise ValueError(f"Unknown optimizer: {name}")

    if gradient_clip_norm is None:
        return optimizer
    return optax.chain(optax.clip_by_global_norm(gradient_clip_norm), optimizer)


def _accumulate_train_step(
    optimizer_state: DistillOptimizerState,
    optimizer: optax.GradientTransformation,
    microbatches: Sequence[DistillBatchLike],
    train_key: Key[Array, ""],
    *,
    stochastic_rounding: bool,
    compute_step_gradients: Callable[
        [DistillTrainingState, DistillBatchLike, Key[Array, ""] | None],
        tuple[object, DistillBatchMetrics],
    ],
) -> tuple[DistillOptimizerState, DistillBatchMetrics, Key[Array, ""]]:
    assert microbatches, "Gradient accumulation requires at least one microbatch"
    accumulated_grads: object | None = None
    total_loss = jnp.asarray(0.0, dtype=jnp.float32)
    total_valid_tokens = jnp.asarray(0, dtype=jnp.int32)
    total_top1_matches = jnp.asarray(0, dtype=jnp.int32)

    for microbatch in microbatches:
        train_key, step_key = jax.random.split(train_key)
        quantization_key = step_key if stochastic_rounding else None
        grads, metrics = compute_step_gradients(
            optimizer_state.training_state,
            microbatch,
            quantization_key,
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
    return apply_distill_gradients(optimizer_state, optimizer, averaged_grads), step_metrics, train_key


def _setup_device_mesh(num_devices: int) -> Mesh | None:
    if num_devices <= 1:
        return None
    devices = jax.devices()[:num_devices]
    if len(devices) < num_devices:
        raise ValueError(f"Requested {num_devices} devices but only {len(devices)} available")
    return Mesh(devices, axis_names=("batch",))


def distill(
    config: DistillConfig,
    *,
    trainable_filter: Callable[[ParameterLeafInfo], bool] | None = None,
    callbacks: DistillCallbacks | None = None,
) -> DistillResult:
    if config.train_examples < 1:
        raise ValueError("train_examples must be at least 1")
    if config.eval_examples < 1:
        raise ValueError("eval_examples must be at least 1")
    if config.training_mode == TrainingMode.ONLINE_EXACT and config.max_sequence_length < 2:
        raise ValueError("Online distillation requires max_sequence_length >= 2")
    if config.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if config.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    if config.gradient_clip_norm is not None and config.gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive when provided")
    if config.num_devices < 1:
        raise ValueError("num_devices must be at least 1")

    mesh = _setup_device_mesh(config.num_devices)
    device_batch_size = config.batch_size * config.num_devices

    callbacks = DistillCallbacks(config) if callbacks is None else callbacks
    callbacks.started()

    # Models
    callbacks.loading_models()
    teacher_model = LanguageModelConfig.load_model(config.teacher_path)
    student_model = LanguageModelConfig.load_model(config.student_path)
    callbacks.finished_loading_models()

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

    _validate_distill_models(student_model, teacher_model, quantization_mode=config.quantization_mode)

    # Dataset
    callbacks.loading_dataset()
    dataset = _load_distill_batches(
        config,
        student_model,
        device_batch_size=device_batch_size,
    )
    callbacks.finished_loading_dataset()

    dataset = _shard_distill_batches(dataset, mesh, num_devices=config.num_devices)

    # Training state
    match config.compute_dtype_name:
        case ComputeDTypeName.AUTO:
            compute_dtype = student_model.model.activation_precision
        case ComputeDTypeName.BFLOAT16:
            compute_dtype = jnp.bfloat16
        case ComputeDTypeName.FLOAT32:
            compute_dtype = jnp.float32
        case _:
            raise ValueError(f"Unknown compute dtype: {config.compute_dtype_name}")

    distill_config = DistillTrainConfig(
        master_dtype=jnp.float32,
        compute_dtype=compute_dtype,
    )
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
    parameter_summary = summarize_distill_parameters(
        iter_parameter_leaves(student_model.model),
        distill_config,
        trainable_filter=effective_trainable_filter,
    )

    if mesh is not None:
        replicated_sharding = NamedSharding(mesh, P())
        student_model = eqx.filter_shard(student_model, replicated_sharding)
        teacher_model = eqx.filter_shard(teacher_model, replicated_sharding)
        optimizer_state = eqx.filter_shard(optimizer_state, replicated_sharding)

    # Checkpoint state
    train_key = jax.random.key(config.seed)
    completed_steps = 0
    best_eval_kl: float | None = None
    best_step: int | None = None
    best_optimizer_state = optimizer_state
    evaluations_without_improvement = 0
    resume_best_checkpoint_dir: Path | None = None
    if config.resume_from is not None:
        checkpoint, checkpoint_metadata = _load_checkpoint(
            config.resume_from,
            ExperimentCheckpoint(
                optimizer_state=optimizer_state,
                train_key_data=jax.random.key_data(train_key),
            ),
        )
        optimizer_state = checkpoint.optimizer_state
        train_key = jax.random.wrap_key_data(checkpoint.train_key_data)
        completed_steps = checkpoint_metadata.step
        best_eval_kl = checkpoint_metadata.best_eval_kl
        best_step = checkpoint_metadata.best_step
        evaluations_without_improvement = checkpoint_metadata.evaluations_without_improvement
        if best_step == completed_steps:
            best_optimizer_state = optimizer_state
        if best_step is not None and best_step != completed_steps:
            resume_best_checkpoint_dir = Path(config.resume_from).parent / "best-checkpoint"
            if not resume_best_checkpoint_dir.exists():
                raise ValueError(
                    f"Resume checkpoint {config.resume_from} is missing sibling best-checkpoint directory",
                )
            best_checkpoint, _ = _load_checkpoint(
                resume_best_checkpoint_dir,
                ExperimentCheckpoint(
                    optimizer_state=optimizer_state,
                    train_key_data=jax.random.key_data(train_key),
                ),
            )
            best_optimizer_state = best_checkpoint.optimizer_state

    match config.training_mode:
        case TrainingMode.ONLINE_EXACT:

            def evaluate_batch(student: Decoder, batch: DistillBatchLike) -> DistillBatchMetrics:
                assert isinstance(batch, DistillBatch)
                return compute_distill_batch_metrics(student, teacher_model.model, batch)

            def compute_step_gradients(
                training_state: DistillTrainingState,
                batch: DistillBatchLike,
                quantization_key: Key[Array, ""] | None,
            ) -> tuple[object, DistillBatchMetrics]:
                assert isinstance(batch, DistillBatch)
                return compute_distill_step_gradients(
                    training_state,
                    student_model.model,
                    teacher_model.model,
                    batch,
                    distill_config,
                    quantization_key=quantization_key,
                    quantization_mode=config.quantization_mode,
                )

        case TrainingMode.TRACE_TOPK:

            def evaluate_batch(student: Decoder, batch: DistillBatchLike) -> DistillBatchMetrics:
                assert isinstance(batch, TraceDistillBatch)
                return compute_trace_distill_batch_metrics(student, batch)

            def compute_step_gradients(
                training_state: DistillTrainingState,
                batch: DistillBatchLike,
                quantization_key: Key[Array, ""] | None,
            ) -> tuple[object, DistillBatchMetrics]:
                assert isinstance(batch, TraceDistillBatch)
                return compute_trace_distill_step_gradients(
                    training_state,
                    student_model.model,
                    batch,
                    distill_config,
                    quantization_key=quantization_key,
                    quantization_mode=config.quantization_mode,
                )

    def evaluate_model(student: Decoder) -> EvaluationMetrics:
        return _evaluate_with(dataset.eval_batches, lambda batch: evaluate_batch(student, batch))

    # Initial evaluation
    run_start = time.perf_counter()
    initial_student = materialize_trainable_module(
        student_model.model,
        optimizer_state.training_state.master_weights,
        distill_config,
    )
    initial_eval = evaluate_model(initial_student)

    if best_eval_kl is None:
        best_eval_kl = initial_eval.kl_divergence
        best_step = completed_steps
        best_optimizer_state = optimizer_state

    config.output_dir.mkdir(parents=True, exist_ok=True)
    latest_checkpoint_dir = config.output_dir / "latest-checkpoint"
    best_checkpoint_dir = config.output_dir / "best-checkpoint"

    def save_checkpoint(checkpoint_dir: Path, step: int) -> None:
        assert best_eval_kl is not None
        assert best_step is not None
        _save_checkpoint(
            checkpoint_dir,
            ExperimentCheckpoint(
                optimizer_state=optimizer_state,
                train_key_data=jax.random.key_data(train_key),
            ),
            CheckpointMetadata(
                step=step,
                best_eval_kl=best_eval_kl,
                best_step=best_step,
                evaluations_without_improvement=evaluations_without_improvement,
            ),
        )

    def update_best(step: int, eval_kl: float) -> bool:
        nonlocal best_eval_kl, best_step, best_optimizer_state
        if best_eval_kl is not None and eval_kl >= best_eval_kl:
            return False
        best_eval_kl = eval_kl
        best_step = step
        best_optimizer_state = optimizer_state
        if config.save_checkpoints:
            save_checkpoint(best_checkpoint_dir, step)
        return True

    if config.save_checkpoints:
        if resume_best_checkpoint_dir is not None and resume_best_checkpoint_dir != best_checkpoint_dir:
            shutil.copytree(resume_best_checkpoint_dir, best_checkpoint_dir, dirs_exist_ok=True)
        elif best_step == completed_steps:
            save_checkpoint(best_checkpoint_dir, completed_steps)
    history_path = config.output_dir / "history.jsonl"
    step_iterator = cycle(dataset.train_batches)
    compilation_seconds = 0.0
    executed_steps = 0
    stopped_early = False
    performed_periodic_evaluation = False

    # Training loop
    with history_path.open("a" if completed_steps > 0 else "w") as history_file:
        microbatch_iterator = islice(
            step_iterator,
            completed_steps * config.gradient_accumulation_steps,
            None,
        )
        for step in range(completed_steps + 1, config.num_steps + 1):
            microbatches = tuple(next(microbatch_iterator) for _ in range(config.gradient_accumulation_steps))

            if executed_steps == 0:
                warmup_start = time.perf_counter()

            optimizer_state, train_metrics, train_key = _accumulate_train_step(
                optimizer_state,
                optimizer,
                microbatches,
                train_key,
                stochastic_rounding=config.stochastic_rounding,
                compute_step_gradients=compute_step_gradients,
            )

            if executed_steps == 0:
                compilation_seconds = time.perf_counter() - warmup_start

            eval_metrics: EvaluationMetrics | None = None
            if config.eval_every_steps > 0 and step % config.eval_every_steps == 0:
                performed_periodic_evaluation = True
                materialized_student = materialize_trainable_module(
                    student_model.model,
                    optimizer_state.training_state.master_weights,
                    distill_config,
                )
                eval_metrics = evaluate_model(materialized_student)
                if update_best(step, eval_metrics.kl_divergence):
                    evaluations_without_improvement = 0
                else:
                    evaluations_without_improvement += 1
                    patience_exceeded = evaluations_without_improvement >= config.early_stop_patience
                    if config.early_stop_patience > 0 and patience_exceeded:
                        stopped_early = True

            history_file.write(
                json.dumps(
                    {
                        "step": step,
                        "train_kl_divergence": float(train_metrics.loss),
                        "valid_tokens": int(train_metrics.valid_tokens),
                        "eval_kl_divergence": None if eval_metrics is None else eval_metrics.kl_divergence,
                        "eval_top1_agreement": None if eval_metrics is None else eval_metrics.top1_agreement,
                        "eval_valid_tokens": None if eval_metrics is None else eval_metrics.valid_tokens,
                    }
                )
                + "\n"
            )
            history_file.flush()
            executed_steps += 1
            completed_steps = step
            callbacks.distillation_progress(step, config.num_steps)

            should_save_checkpoint = config.checkpoint_every_steps > 0 and step % config.checkpoint_every_steps == 0
            if config.save_checkpoints and (should_save_checkpoint or stopped_early):
                save_checkpoint(latest_checkpoint_dir, step)
            if stopped_early:
                break

    if config.save_checkpoints:
        save_checkpoint(latest_checkpoint_dir, completed_steps)

    elapsed_seconds = time.perf_counter() - run_start
    measured_steps = max(executed_steps - 1, 0)

    should_restore_best_checkpoint = (
        performed_periodic_evaluation and best_step is not None and best_step != completed_steps
    )
    if should_restore_best_checkpoint:
        if config.save_checkpoints and best_checkpoint_dir.exists():
            best_checkpoint, _ = _load_checkpoint(
                best_checkpoint_dir,
                ExperimentCheckpoint(
                    optimizer_state=optimizer_state,
                    train_key_data=jax.random.key_data(train_key),
                ),
            )
            optimizer_state = best_checkpoint.optimizer_state
        else:
            optimizer_state = best_optimizer_state

    # Final evaluation and export
    final_student = materialize_trainable_module(
        student_model.model,
        optimizer_state.training_state.master_weights,
        distill_config,
    )
    final_eval = evaluate_model(final_student)
    update_best(completed_steps, final_eval.kl_divergence)

    model_output_path = config.output_dir / "student-distilled"
    export_config = DistillTrainConfig(
        master_dtype=distill_config.master_dtype,
        compute_dtype=student_model.model.activation_precision,
    )
    export_student = materialize_trainable_module(
        student_model.model,
        optimizer_state.training_state.master_weights,
        export_config,
    )
    callbacks.saving_model()
    _save_materialized_student(
        config.student_path,
        model_output_path,
        student_model,
        export_student,
    )
    callbacks.finished_saving_model(model_output_path)

    result = DistillResult(
        teacher_path=str(config.teacher_path),
        student_path=str(config.student_path),
        dataset_path=str(config.dataset_path),
        training_mode=config.training_mode,
        train_examples=config.train_examples,
        eval_examples=config.eval_examples,
        max_sequence_length=config.max_sequence_length,
        batch_size=config.batch_size,
        num_steps=config.num_steps,
        completed_steps=completed_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        gradient_clip_norm=config.gradient_clip_norm,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        seed=config.seed,
        optimizer=config.optimizer_name,
        quantization_mode=config.quantization_mode.value,
        master_dtype=str(jnp.dtype(distill_config.master_dtype)),
        compute_dtype=str(jnp.dtype(distill_config.compute_dtype)),
        parameter_summary=parameter_summary,
        initial_eval=initial_eval,
        final_eval=final_eval,
        best_step=best_step,
        stopped_early=stopped_early,
        compilation_seconds=compilation_seconds,
        elapsed_seconds=elapsed_seconds,
        seconds_per_step=0.0 if measured_steps == 0 else elapsed_seconds / measured_steps,
        output_model_path=str(model_output_path),
    )

    with (config.output_dir / "metrics.json").open("w") as metrics_file:
        json.dump(asdict(result), metrics_file, indent=4)

    callbacks.finished(result)
    return result
