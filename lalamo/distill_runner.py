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
from jaxtyping import Array, DTypeLike, Key

from lalamo.common import flatten_parameters
from lalamo.data import load_hf_parquet, shuffle_dataset
from lalamo.data.huggingface_message import HFMessage
from lalamo.data.lalamo_completions import LalamoCompletion
from lalamo.distillation import (
    DistillBatch,
    DistillOptimizerState,
    DistillParameterSummary,
    DistillStepMetrics,
    DistillTrainConfig,
    DistillTrainingState,
    TraceDistillBatch,
    apply_distill_gradients,
    compute_distill_batch_metrics,
    compute_distill_step_gradients,
    compute_trace_distill_batch_metrics,
    compute_trace_distill_step_gradients,
    get_muon_weight_dimension_numbers,
    initialize_distill_optimizer_state,
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
    "DistillConfigSnapshot",
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


@dataclass(frozen=True)
class DistillConfigSnapshot:
    master_dtype: str
    compute_dtype: str


@dataclass(frozen=True)
class CheckpointMetadata:
    step: int
    best_eval_kl: float | None
    best_step: int | None
    evaluations_without_improvement: int


@dataclass(frozen=True)
class HistoryEntry:
    step: int
    train_kl_divergence: float
    valid_tokens: int
    eval_kl_divergence: float | None = None
    eval_top1_agreement: float | None = None
    eval_valid_tokens: int | None = None


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
    distill_config: DistillConfigSnapshot
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


def _load_conversations(dataset_path: Path | str, *, num_examples: int, seed: int) -> list[list[HFMessage]]:
    dataset_path = Path(dataset_path)
    dataframe = shuffle_dataset(load_hf_parquet(dataset_path), seed=seed)
    if "conversation" in dataframe.columns:
        column_name = "conversation"
    elif "messages" in dataframe.columns:
        column_name = "messages"
    else:
        raise ValueError(f"{dataset_path} must contain a 'conversation' or 'messages' column")

    conversations = dataframe.get_column(column_name).to_list()[:num_examples]
    return [[HFMessage.from_dict(message) for message in conversation] for conversation in conversations]


def _load_traces(dataset_path: Path | str, *, num_examples: int) -> list[LalamoCompletion]:
    dataset_path = Path(dataset_path)
    with dataset_path.open("rb") as trace_fd:
        return list(islice(LalamoCompletion.deserialize_many(trace_fd), num_examples))


def _tokenize_conversations(
    conversations: list[list[HFMessage]],
    *,
    language_model: LanguageModel,
    max_sequence_length: int,
) -> list[np.ndarray]:
    tokenized_sequences: list[np.ndarray] = []

    for conversation in conversations:
        messages = [message.as_message() for message in conversation]
        token_ids = language_model.message_processor.tokenize_request(messages)
        if len(token_ids) < 2:
            continue
        tokenized_sequences.append(np.array(token_ids[:max_sequence_length], dtype=np.int32))

    return tokenized_sequences


def _make_batches(
    sequences: list[np.ndarray],
    *,
    batch_size: int,
    fixed_sequence_length: int | None = None,
) -> list[DistillBatch]:
    batches: list[DistillBatch] = []
    for start in range(0, len(sequences), batch_size):
        batch_sequences = sequences[start : start + batch_size]
        if not batch_sequences:
            continue

        pad_length = fixed_sequence_length or max(seq.shape[0] for seq in batch_sequences)
        padded = np.zeros((len(batch_sequences), pad_length), dtype=np.int32)
        lengths = np.array([seq.shape[0] for seq in batch_sequences], dtype=np.int32)

        for row, seq in enumerate(batch_sequences):
            padded[row, : seq.shape[0]] = seq

        batches.append(DistillBatch(token_ids=jnp.array(padded), lengths_without_padding=jnp.array(lengths)))

    return batches


def _make_trace_batches(
    traces: Sequence[LalamoCompletion],
    *,
    batch_size: int,
    pad_token_id: int,
) -> list[TraceDistillBatch]:
    return [
        make_trace_distill_batch(traces[start : start + batch_size], pad_token_id=pad_token_id)
        for start in range(0, len(traces), batch_size)
    ]


def _evaluate(student: Decoder, teacher: Decoder, batches: list[DistillBatch]) -> EvaluationMetrics:
    total_kl = 0.0
    total_valid_tokens = 0
    total_matches = 0

    for batch in batches:
        batch_metrics = compute_distill_batch_metrics(student, teacher, batch)
        total_kl += float(batch_metrics.loss) * int(batch_metrics.valid_tokens)
        total_valid_tokens += int(batch_metrics.valid_tokens)
        total_matches += int(batch_metrics.top1_matches)

    return EvaluationMetrics(
        kl_divergence=total_kl / total_valid_tokens,
        top1_agreement=total_matches / total_valid_tokens,
        valid_tokens=total_valid_tokens,
    )


def _evaluate_trace(student: Decoder, batches: list[TraceDistillBatch]) -> EvaluationMetrics:
    total_kl = 0.0
    total_valid_tokens = 0
    total_matches = 0

    for batch in batches:
        batch_metrics = compute_trace_distill_batch_metrics(student, batch)
        total_kl += float(batch_metrics.loss) * int(batch_metrics.valid_tokens)
        total_valid_tokens += int(batch_metrics.valid_tokens)
        total_matches += int(batch_metrics.top1_matches)

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
    model_config = inject_lora_adapter_configs(student_model.config.model_config, lora_rank, lora_scale)
    model = inject_lora_adapters(student_model.model, lora_rank, lora_scale, key)
    return LanguageModel(
        config=LanguageModelConfig(
            model_config=model_config,
            message_processor_config=student_model.config.message_processor_config,
            generation_config=student_model.config.generation_config,
        ),
        model=model,
        message_processor=student_model.message_processor,
    )


def _load_model_config_json(model_path: Path | str) -> dict[str, object]:
    model_path = Path(model_path)
    with (model_path / "config.json").open() as config_file:
        config_json = json.load(config_file)
    if not isinstance(config_json, dict):
        raise TypeError(f"{model_path / 'config.json'} must contain a JSON object")
    return config_json


def _updated_config_json(
    source_config_json: dict[str, object],
    updated_model: LanguageModel,
) -> dict[str, object]:
    metadata = config_converter.structure(source_config_json, ModelMetadata)
    updated_metadata = replace(metadata, model_config=updated_model.config)
    return config_converter.unstructure(updated_metadata, ModelMetadata)


def _save_materialized_student(
    student_path: Path | str,
    output_path: Path | str,
    student_model: LanguageModel,
    materialized_student: Decoder,
) -> None:
    student_path = Path(student_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    source_config_json = _load_model_config_json(student_path)
    updated_model = LanguageModel(
        config=student_model.config,
        model=materialized_student,
        message_processor=student_model.message_processor,
    )

    shutil.copy2(student_path / "tokenizer.json", output_path / "tokenizer.json")

    with (output_path / "config.json").open("w") as config_file:
        json.dump(_updated_config_json(source_config_json, updated_model), config_file, indent=4)
    with (output_path / "model.safetensors").open("wb") as weights_file:
        safe_write(weights_file, flatten_parameters(updated_model.export_weights()))


def _save_checkpoint(
    checkpoint_dir: Path | str,
    checkpoint: ExperimentCheckpoint,
    metadata: CheckpointMetadata,
) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(checkpoint_dir / "state.eqx", checkpoint)
    with (checkpoint_dir / "metadata.json").open("w") as metadata_file:
        json.dump(_CHECKPOINT_CONVERTER.unstructure(metadata), metadata_file, indent=4)


def _load_checkpoint(
    checkpoint_dir: Path | str,
    like: ExperimentCheckpoint,
) -> tuple[ExperimentCheckpoint, CheckpointMetadata]:
    checkpoint_dir = Path(checkpoint_dir)
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
                muon_weight_dimension_numbers=get_muon_weight_dimension_numbers(training_state),
            )
        case OptimizerName.SGD:
            optimizer = optax.sgd(learning_rate_schedule)
        case _:
            raise ValueError(f"Unknown optimizer: {name}")

    if gradient_clip_norm is None:
        return optimizer
    return optax.chain(optax.clip_by_global_norm(gradient_clip_norm), optimizer)


def _resolve_compute_dtype(name: ComputeDTypeName, native_dtype: DTypeLike) -> DTypeLike:
    match name:
        case ComputeDTypeName.AUTO:
            return native_dtype
        case ComputeDTypeName.BFLOAT16:
            return jnp.bfloat16
        case ComputeDTypeName.FLOAT32:
            return jnp.float32
        case _:
            raise ValueError(f"Unknown compute dtype: {name}")


def _accumulate_train_step[BatchT: eqx.Module](
    optimizer_state: DistillOptimizerState,
    optimizer: optax.GradientTransformation,
    microbatches: Sequence[BatchT],
    train_key: Key[Array, ""],
    *,
    stochastic_rounding: bool,
    compute_step_gradients: Callable[
        [BatchT, Key[Array, ""] | None],
        tuple[tuple[Array, ...], DistillStepMetrics],
    ],
) -> tuple[DistillOptimizerState, DistillStepMetrics, Key[Array, ""]]:
    accumulated_grads: tuple[Array, ...] | None = None
    accumulated_metrics: DistillStepMetrics | None = None
    total_loss = 0.0
    total_valid_tokens = 0

    for microbatch in microbatches:
        train_key, step_key = jax.random.split(train_key)
        quantization_key = step_key if stochastic_rounding else None
        grads, metrics = compute_step_gradients(microbatch, quantization_key)
        accumulated_metrics = metrics
        valid_tokens = int(metrics.valid_tokens)
        weighted_grads = tuple(grad * valid_tokens for grad in grads)
        accumulated_grads = (
            weighted_grads
            if accumulated_grads is None
            else jax.tree.map(lambda total, grad: total + grad, accumulated_grads, weighted_grads)
        )
        total_loss += float(metrics.loss) * valid_tokens
        total_valid_tokens += valid_tokens

    if accumulated_grads is None or accumulated_metrics is None:
        raise ValueError("Gradient accumulation requires at least one microbatch")

    averaged_grads = tuple(grad / total_valid_tokens for grad in accumulated_grads)
    accumulated_metrics = DistillStepMetrics(
        loss=jnp.asarray(total_loss / total_valid_tokens, dtype=jnp.float32),
        valid_tokens=jnp.asarray(total_valid_tokens, dtype=jnp.int32),
    )
    return apply_distill_gradients(optimizer_state, optimizer, averaged_grads), accumulated_metrics, train_key


def _setup_device_mesh(num_devices: int) -> Mesh | None:
    if num_devices <= 1:
        return None
    devices = jax.devices()[:num_devices]
    if len(devices) < num_devices:
        raise ValueError(f"Requested {num_devices} devices but only {len(devices)} available")
    return Mesh(devices, axis_names=("batch",))


def _shard_batch[B: eqx.Module](batch: B, mesh: Mesh) -> B:
    return eqx.filter_shard(batch, NamedSharding(mesh, P("batch")))


def _replicate[M: eqx.Module](module: M, mesh: Mesh) -> M:
    return eqx.filter_shard(module, NamedSharding(mesh, P()))


def distill(
    config: DistillConfig,
    *,
    trainable_filter: Callable[[ParameterLeafInfo], bool] | None = None,
    callbacks_type: Callable[[DistillConfig], DistillCallbacks] = DistillCallbacks,
) -> DistillResult:
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

    callbacks = callbacks_type(config)
    callbacks.started()

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

    callbacks.loading_dataset()
    match config.training_mode:
        case TrainingMode.ONLINE_EXACT:
            conversations = _load_conversations(
                config.dataset_path,
                num_examples=config.train_examples + config.eval_examples,
                seed=config.seed,
            )
            tokenized = _tokenize_conversations(
                conversations,
                language_model=student_model,
                max_sequence_length=config.max_sequence_length,
            )
            if len(tokenized) < config.train_examples + config.eval_examples:
                raise ValueError(
                    f"Requested {config.train_examples + config.eval_examples} usable sequences, "
                    f"got {len(tokenized)} from {config.dataset_path}",
                )
            train_items = tokenized[: config.train_examples]
            eval_items = tokenized[config.train_examples : config.train_examples + config.eval_examples]
            fixed_seq_len = config.max_sequence_length if mesh is not None else None
            train_batches = _make_batches(
                train_items, batch_size=device_batch_size, fixed_sequence_length=fixed_seq_len,
            )
            eval_batches = _make_batches(
                eval_items, batch_size=device_batch_size, fixed_sequence_length=fixed_seq_len,
            )
        case TrainingMode.TRACE_TOPK:
            traces = _load_traces(config.dataset_path, num_examples=config.train_examples + config.eval_examples)
            if len(traces) < config.train_examples + config.eval_examples:
                raise ValueError(
                    f"Requested {config.train_examples + config.eval_examples} traces, "
                    f"got {len(traces)} from {config.dataset_path}",
                )
            train_items = traces[: config.train_examples]
            eval_items = traces[config.train_examples : config.train_examples + config.eval_examples]
            train_batches = _make_trace_batches(train_items, batch_size=device_batch_size, pad_token_id=0)
            eval_batches = _make_trace_batches(eval_items, batch_size=device_batch_size, pad_token_id=0)
    callbacks.finished_loading_dataset()

    if mesh is not None:
        is_shardable = lambda b: b.token_ids.shape[0] % config.num_devices == 0  # noqa: E731
        train_batches = [_shard_batch(b, mesh) for b in train_batches if is_shardable(b)]
        eval_batches = [_shard_batch(b, mesh) for b in eval_batches if is_shardable(b)]

    distill_config = DistillTrainConfig(
        master_dtype=jnp.float32,
        compute_dtype=_resolve_compute_dtype(config.compute_dtype_name, student_model.model.activation_precision),
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
    optimizer_state = initialize_distill_optimizer_state(training_state, optimizer)
    parameter_summary = summarize_distill_parameters(
        iter_parameter_leaves(student_model.model),
        distill_config,
        trainable_filter=effective_trainable_filter,
    )

    if mesh is not None:
        student_model = _replicate(student_model, mesh)
        teacher_model = _replicate(teacher_model, mesh)
        optimizer_state = _replicate(optimizer_state, mesh)

    train_key = jax.random.key(config.seed)
    completed_steps = 0
    best_eval_kl: float | None = None
    best_step: int | None = None
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
        if best_step is not None and best_step != completed_steps:
            resume_best_checkpoint_dir = Path(config.resume_from).parent / "best-checkpoint"
            if not resume_best_checkpoint_dir.exists():
                raise ValueError(
                    f"Resume checkpoint {config.resume_from} is missing sibling best-checkpoint directory",
                )

    initial_student = materialize_trainable_module(student_model.model, optimizer_state.training_state, distill_config)
    match config.training_mode:
        case TrainingMode.ONLINE_EXACT:
            initial_eval = _evaluate(initial_student, teacher_model.model, eval_batches)
        case TrainingMode.TRACE_TOPK:
            initial_eval = _evaluate_trace(initial_student, eval_batches)

    if best_eval_kl is None:
        best_eval_kl = initial_eval.kl_divergence
        best_step = max(0, completed_steps)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    latest_checkpoint_dir = config.output_dir / "latest-checkpoint"
    best_checkpoint_dir = config.output_dir / "best-checkpoint"
    if config.save_checkpoints:
        if resume_best_checkpoint_dir is not None:
            shutil.copytree(resume_best_checkpoint_dir, best_checkpoint_dir, dirs_exist_ok=True)
        elif best_step == completed_steps:
            _save_checkpoint(
                best_checkpoint_dir,
                ExperimentCheckpoint(
                    optimizer_state=optimizer_state,
                    train_key_data=jax.random.key_data(train_key),
                ),
                CheckpointMetadata(
                    step=completed_steps,
                    best_eval_kl=best_eval_kl,
                    best_step=best_step,
                    evaluations_without_improvement=evaluations_without_improvement,
                ),
            )
    history_path = config.output_dir / "history.jsonl"
    step_iterator = cycle(train_batches)
    compilation_seconds = 0.0
    start_time: float | None = None
    executed_steps = 0
    stopped_early = False

    with history_path.open("a" if completed_steps > 0 else "w") as history_file:
        microbatch_iterator = islice(
            step_iterator,
            completed_steps * config.gradient_accumulation_steps,
            None,
        )
        for step in range(completed_steps + 1, config.num_steps + 1):
            microbatches = tuple(next(microbatch_iterator) for _ in range(config.gradient_accumulation_steps))
            current_training_state = optimizer_state.training_state

            if executed_steps == 0:
                warmup_start = time.perf_counter()

            match config.training_mode:
                case TrainingMode.ONLINE_EXACT:
                    optimizer_state, train_metrics, train_key = _accumulate_train_step(
                        optimizer_state,
                        optimizer,
                        microbatches,
                        train_key,
                        stochastic_rounding=config.stochastic_rounding,
                        compute_step_gradients=lambda batch,
                        quantization_key,
                        training_state=current_training_state: compute_distill_step_gradients(
                            training_state,
                            student_model.model,
                            teacher_model.model,
                            batch,
                            distill_config,
                            quantization_key=quantization_key,
                            quantization_mode=config.quantization_mode,
                        ),
                    )
                case TrainingMode.TRACE_TOPK:
                    optimizer_state, train_metrics, train_key = _accumulate_train_step(
                        optimizer_state,
                        optimizer,
                        microbatches,
                        train_key,
                        stochastic_rounding=config.stochastic_rounding,
                        compute_step_gradients=lambda batch,
                        quantization_key,
                        training_state=current_training_state: compute_trace_distill_step_gradients(
                            training_state,
                            student_model.model,
                            batch,
                            distill_config,
                            quantization_key=quantization_key,
                            quantization_mode=config.quantization_mode,
                        ),
                    )

            if executed_steps == 0:
                compilation_seconds = time.perf_counter() - warmup_start
                start_time = time.perf_counter()

            history_entry = HistoryEntry(
                step=step,
                train_kl_divergence=float(train_metrics.loss),
                valid_tokens=int(train_metrics.valid_tokens),
            )

            if config.eval_every_steps > 0 and step % config.eval_every_steps == 0:
                materialized_student = materialize_trainable_module(
                    student_model.model,
                    optimizer_state.training_state,
                    distill_config,
                )
                match config.training_mode:
                    case TrainingMode.ONLINE_EXACT:
                        eval_metrics = _evaluate(materialized_student, teacher_model.model, eval_batches)
                    case TrainingMode.TRACE_TOPK:
                        eval_metrics = _evaluate_trace(materialized_student, eval_batches)

                history_entry = HistoryEntry(
                    step=step,
                    train_kl_divergence=float(train_metrics.loss),
                    valid_tokens=int(train_metrics.valid_tokens),
                    eval_kl_divergence=eval_metrics.kl_divergence,
                    eval_top1_agreement=eval_metrics.top1_agreement,
                    eval_valid_tokens=eval_metrics.valid_tokens,
                )
                if best_eval_kl is None or eval_metrics.kl_divergence < best_eval_kl:
                    best_eval_kl = eval_metrics.kl_divergence
                    best_step = step
                    evaluations_without_improvement = 0
                    if config.save_checkpoints:
                        _save_checkpoint(
                            best_checkpoint_dir,
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
                else:
                    evaluations_without_improvement += 1
                    patience_exceeded = evaluations_without_improvement >= config.early_stop_patience
                    if config.early_stop_patience > 0 and patience_exceeded:
                        stopped_early = True

            history_file.write(json.dumps(asdict(history_entry)) + "\n")
            history_file.flush()
            executed_steps += 1
            completed_steps = step
            callbacks.distillation_progress(step, config.num_steps)

            should_save_checkpoint = config.checkpoint_every_steps > 0 and step % config.checkpoint_every_steps == 0
            if config.save_checkpoints and (should_save_checkpoint or stopped_early):
                _save_checkpoint(
                    latest_checkpoint_dir,
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
            if stopped_early:
                break

    if config.save_checkpoints:
        _save_checkpoint(
            latest_checkpoint_dir,
            ExperimentCheckpoint(
                optimizer_state=optimizer_state,
                train_key_data=jax.random.key_data(train_key),
            ),
            CheckpointMetadata(
                step=completed_steps,
                best_eval_kl=best_eval_kl,
                best_step=best_step,
                evaluations_without_improvement=evaluations_without_improvement,
            ),
        )

    elapsed_seconds = 0.0 if start_time is None else time.perf_counter() - start_time
    measured_steps = max(executed_steps - 1, 0)

    should_restore_best_checkpoint = (
        config.save_checkpoints
        and best_checkpoint_dir.exists()
        and best_step is not None
        and best_step != completed_steps
    )
    if should_restore_best_checkpoint:
        best_checkpoint, _ = _load_checkpoint(
            best_checkpoint_dir,
            ExperimentCheckpoint(
                optimizer_state=optimizer_state,
                train_key_data=jax.random.key_data(train_key),
            ),
        )
        optimizer_state = best_checkpoint.optimizer_state

    final_student = materialize_trainable_module(student_model.model, optimizer_state.training_state, distill_config)
    match config.training_mode:
        case TrainingMode.ONLINE_EXACT:
            final_eval = _evaluate(final_student, teacher_model.model, eval_batches)
        case TrainingMode.TRACE_TOPK:
            final_eval = _evaluate_trace(final_student, eval_batches)

    model_output_path = config.output_dir / "student-distilled"
    export_config = DistillTrainConfig(
        master_dtype=distill_config.master_dtype,
        compute_dtype=student_model.model.activation_precision,
    )
    export_student = materialize_trainable_module(student_model.model, optimizer_state.training_state, export_config)
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
        train_examples=len(train_items),
        eval_examples=len(eval_items),
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
        distill_config=DistillConfigSnapshot(
            master_dtype=str(jnp.dtype(distill_config.master_dtype)),
            compute_dtype=str(jnp.dtype(distill_config.compute_dtype)),
        ),
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
