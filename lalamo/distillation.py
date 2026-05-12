from collections.abc import Sequence
from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from jax import Array
from jaxtyping import Bool, DTypeLike, Float, Int

from lalamo.data.lalamo_completions import LalamoCompletion
from lalamo.compressed.hybrid import HybridSpec
from lalamo.compressed.low_rank import LowRankMatrix, LowRankSpec
from lalamo.module import Keychain, ParameterNorm
from lalamo.modules.normalization import Normalization
from lalamo.modules.decoder import Decoder, DecoderForwardPassConfig
from lalamo.utils.field_metadata import field_metadata_for_leaf, partition_trainable
from lalamo.utils.surgery import map_nodes_of_type
from lalamo.weight_matrix import CompressionImplementation, EmbeddingMatrix, GradientEstimator, WeightMatrix

__all__ = [
    "DistillBatch",
    "DistillBatchMetrics",
    "DistillConfig",
    "DistillOptimizerState",
    "DistillTrainingState",
    "TraceDistillBatch",
    "apply_distill_gradients",
    "apply_donated_distill_gradients",
    "compute_distill_batch_metrics",
    "compute_distill_step_gradients",
    "compute_trace_distill_batch_metrics",
    "compute_trace_distill_step_gradients",
    "inject_lora_adapters",
    "initialize_distill_training_state",
    "make_trace_distill_batch",
]


# ---------------------------------------------------------------------------
#  Config and data
# ---------------------------------------------------------------------------


class DistillConfig(eqx.Module):
    master_dtype: DTypeLike = jnp.float32
    compute_dtype: DTypeLike = jnp.bfloat16
    gradient_estimator: GradientEstimator = GradientEstimator.DETERMINISTIC_ROUNDING
    teacher_argmax_ce_weight: float = 0.0
    hidden_state_mse_weight: float = 0.0
    hard_token_weight: float = 0.0


class DistillBatch(eqx.Module):
    token_ids: Int[Array, "batch tokens"]
    lengths_without_padding: Int[Array, " batch"]


class DistillBatchMetrics(eqx.Module):
    loss: Float[Array, ""]
    kl_divergence: Float[Array, ""]
    valid_tokens: Int[Array, ""]
    top1_matches: Int[Array, ""]


class TraceDistillBatch(eqx.Module):
    token_ids: Int[Array, "batch tokens"]
    prefix_lengths: Int[Array, " batch"]
    completion_lengths: Int[Array, " batch"]
    support_token_ids: Int[Array, "batch completion_tokens support"]
    support_logits: Float[Array, "batch completion_tokens support"]
    support_mask: Bool[Array, "batch completion_tokens support"]


class DistillTrainingState(eqx.Module):
    master_weights: eqx.Module
    frozen: eqx.Module
    muon_dimension_numbers: eqx.Module


class DistillOptimizerState(eqx.Module):
    training_state: DistillTrainingState
    optimizer_state: optax.OptState


# ---------------------------------------------------------------------------
#  Initialization
# ---------------------------------------------------------------------------


def _make_muon_dimension_numbers(module: eqx.Module) -> eqx.Module:
    muon_dims = optax.contrib.MuonDimensionNumbers(reduction_axis=1, output_axis=0)
    leaves_with_paths, treedef = jtu.tree_flatten_with_path(module)
    return treedef.unflatten(
        muon_dims
        if isinstance(leaf, Array)
        and leaf.ndim == 2
        and field_metadata_for_leaf(module, path).norm == ParameterNorm.SPECTRAL
        else None
        for path, leaf in leaves_with_paths
    )


def _cast_leaves(tree: eqx.Module, dtype: DTypeLike) -> eqx.Module:
    return jax.tree.map(
        lambda leaf: leaf.astype(dtype) if isinstance(leaf, Array) and jnp.issubdtype(leaf.dtype, jnp.floating) else leaf,
        tree,
        is_leaf=lambda leaf: leaf is None,
    )



def inject_lora_adapters(
    module: Decoder,
    lora_rank: int,
    *,
    key: jax.Array,
) -> Decoder:
    """Wrap each quantized WeightMatrix in a HybridMatrix with a low-rank adapter."""
    adapter_spec = LowRankSpec(rank=lora_rank)
    keys = jax.random.split(key, len(jax.tree.leaves(module)))
    key_iter = iter(keys)

    def add_adapter(wm: WeightMatrix) -> WeightMatrix:
        if wm.ndim < 2 or isinstance(wm, EmbeddingMatrix):
            return wm
        hybrid_spec = HybridSpec(
            quantization_spec=wm.spec,
            adapter_spec=adapter_spec,
            incoherence_block_size=None,
        )
        return hybrid_spec.compress(
            wm.decompress(),
            key=next(key_iter),
            implementation=CompressionImplementation.TRAINING,
        )

    return map_nodes_of_type(WeightMatrix, add_adapter, module)


def _lora_trainable_filter(module: eqx.Module, path: tuple[object, ...], leaf: object) -> bool:
    if not isinstance(leaf, Array):
        return False
    current = module
    for key in path:
        name = key.name if hasattr(key, "name") else str(key)
        child = getattr(current, name, None)
        if child is None:
            break
        if isinstance(child, (LowRankMatrix, Normalization)):
            return True
        current = child
    return False


def partition_lora_trainable(module: eqx.Module) -> tuple[eqx.Module, eqx.Module]:
    leaves_with_paths, treedef = jtu.tree_flatten_with_path(module)
    filter_spec = treedef.unflatten(
        _lora_trainable_filter(module, path, leaf) for path, leaf in leaves_with_paths
    )
    return eqx.partition(module, filter_spec)


def initialize_distill_training_state(
    module: eqx.Module,
    config: DistillConfig,
    *,
    lora: bool = False,
) -> DistillTrainingState:
    module = map_nodes_of_type(
        WeightMatrix,
        lambda wm: wm.switch_implementation(CompressionImplementation.TRAINING),
        module,
    )
    trainable, frozen = partition_lora_trainable(module) if lora else partition_trainable(module)
    return DistillTrainingState(
        master_weights=_cast_leaves(trainable, config.master_dtype),
        frozen=frozen,
        muon_dimension_numbers=_make_muon_dimension_numbers(module),
    )


def _materialize_student(
    student: Decoder,
    training_state: DistillTrainingState,
    compute_dtype: DTypeLike,
) -> Decoder:
    return eqx.combine(_cast_leaves(training_state.master_weights, compute_dtype), training_state.frozen)


# ---------------------------------------------------------------------------
#  Batch construction
# ---------------------------------------------------------------------------


def make_trace_distill_batch(
    traces: Sequence[LalamoCompletion],
    *,
    pad_token_id: int,
) -> TraceDistillBatch:
    prefix_lengths = jnp.array([len(t.prefix_token_ids) for t in traces], dtype=jnp.int32)
    completion_lengths = jnp.array([len(t.completion_token_ids) for t in traces], dtype=jnp.int32)

    max_seq = max(len(t.prefix_token_ids) + len(t.completion_token_ids) for t in traces)
    max_comp = max(len(t.completion_token_ids) for t in traces)
    max_support = max(len(tl) for t in traces for tl in t.completion_token_logits)

    token_ids = jnp.full((len(traces), max_seq), pad_token_id, dtype=jnp.int32)
    support_token_ids = jnp.zeros((len(traces), max_comp, max_support), dtype=jnp.int32)
    support_logits = jnp.full((len(traces), max_comp, max_support), -jnp.inf, dtype=jnp.float32)
    support_mask = jnp.zeros((len(traces), max_comp, max_support), dtype=bool)

    for i, trace in enumerate(traces):
        seq = jnp.array([*trace.prefix_token_ids, *trace.completion_token_ids], dtype=jnp.int32)
        token_ids = token_ids.at[i, : seq.shape[0]].set(seq)
        for j, logits in enumerate(trace.completion_token_logits):
            items = tuple(logits.items())
            ids = jnp.array([tok for tok, _ in items], dtype=jnp.int32)
            vals = jnp.array([v for _, v in items], dtype=jnp.float32)
            w = ids.shape[0]
            support_token_ids = support_token_ids.at[i, j, :w].set(ids)
            support_logits = support_logits.at[i, j, :w].set(vals)
            support_mask = support_mask.at[i, j, :w].set(True)

    return TraceDistillBatch(
        token_ids=token_ids,
        prefix_lengths=prefix_lengths,
        completion_lengths=completion_lengths,
        support_token_ids=support_token_ids,
        support_logits=support_logits,
        support_mask=support_mask,
    )


# ---------------------------------------------------------------------------
#  Loss computation
# ---------------------------------------------------------------------------


def _token_positions(token_ids: Int[Array, "batch tokens"]) -> Int[Array, "batch tokens"]:
    _, num_tokens = token_ids.shape
    return jnp.broadcast_to(jnp.arange(num_tokens, dtype=jnp.int32), token_ids.shape)


def _training_forward_pass_config(config: DistillConfig) -> DecoderForwardPassConfig:
    return DecoderForwardPassConfig.for_training(gradient_estimator=config.gradient_estimator)


@eqx.filter_jit
def compute_distill_batch_metrics(
    student: Decoder,
    teacher: Decoder,
    batch: DistillBatch,
    config: DistillConfig,
    *,
    keychain: Keychain,
) -> DistillBatchMetrics:
    positions = _token_positions(batch.token_ids)
    fwd_config = _training_forward_pass_config(config)
    return_trace = config.hidden_state_mse_weight > 0.0
    student_keychain, teacher_keychain = keychain.split(2)

    teacher_result = teacher(
        token_ids=batch.token_ids,
        token_positions=positions,
        return_activation_trace=return_trace,
        forward_pass_config=DecoderForwardPassConfig.for_inference(),
        keychain=teacher_keychain,
    )
    student_result = student(
        token_ids=batch.token_ids,
        token_positions=positions,
        return_activation_trace=return_trace,
        forward_pass_config=fwd_config,
        keychain=student_keychain,
    )

    teacher_logits = teacher_result.logits[:, :-1, :].astype(jnp.float32)
    student_logits = student_result.logits[:, :-1, :].astype(jnp.float32)
    teacher_log_probs = jax.nn.log_softmax(teacher_logits, axis=-1)
    student_log_probs = jax.nn.log_softmax(student_logits, axis=-1)
    teacher_probs = jnp.exp(teacher_log_probs)

    _, num_tokens = batch.token_ids.shape
    token_kl = jnp.sum(teacher_probs * (teacher_log_probs - student_log_probs), axis=-1)
    prediction_mask = (
        jnp.arange(num_tokens - 1, dtype=jnp.int32)[None, :] < (batch.lengths_without_padding[:, None] - 1)
    ).astype(token_kl.dtype)
    valid_tokens = prediction_mask.sum(dtype=jnp.int32)
    valid_tokens = eqx.error_if(valid_tokens, valid_tokens <= 0, "Batch requires at least one prediction token")

    kl_divergence = jnp.sum(token_kl * prediction_mask) / valid_tokens.astype(token_kl.dtype)
    loss = kl_divergence

    if config.hard_token_weight > 0.0:
        token_weights = 1.0 + config.hard_token_weight * jax.lax.stop_gradient(token_kl / kl_divergence)
        weighted_mask = prediction_mask * token_weights
        loss = jnp.sum(token_kl * weighted_mask) / jnp.sum(weighted_mask)

    student_top1 = jnp.argmax(student_logits, axis=-1)
    teacher_top1 = jnp.argmax(teacher_logits, axis=-1)

    if config.teacher_argmax_ce_weight > 0.0:
        ce = optax.softmax_cross_entropy_with_integer_labels(student_logits, teacher_top1)
        loss = loss + config.teacher_argmax_ce_weight * jnp.sum(ce * prediction_mask) / valid_tokens.astype(ce.dtype)

    if config.hidden_state_mse_weight > 0.0:
        assert teacher_result.activation_trace is not None
        assert student_result.activation_trace is not None
        mse = jnp.mean(
            jnp.square(
                student_result.activation_trace.output_norm[:, :-1, :].astype(jnp.float32)
                - teacher_result.activation_trace.output_norm[:, :-1, :].astype(jnp.float32)
            ),
            axis=-1,
        )
        loss = loss + config.hidden_state_mse_weight * jnp.sum(mse * prediction_mask) / valid_tokens.astype(mse.dtype)

    top1_matches = jnp.sum(
        jnp.logical_and(student_top1 == teacher_top1, prediction_mask.astype(bool)),
        dtype=jnp.int32,
    )
    return DistillBatchMetrics(loss=loss, kl_divergence=kl_divergence, valid_tokens=valid_tokens, top1_matches=top1_matches)


@eqx.filter_jit
def compute_trace_distill_batch_metrics(
    student: Decoder,
    batch: TraceDistillBatch,
    config: DistillConfig,
    *,
    keychain: Keychain,
) -> DistillBatchMetrics:
    positions = _token_positions(batch.token_ids)
    student_result = student(
        token_ids=batch.token_ids,
        token_positions=positions,
        forward_pass_config=_training_forward_pass_config(config),
        keychain=keychain,
    )
    student_logits = student_result.logits[:, :-1, :].astype(jnp.float32)

    _, completion_tokens, _ = batch.support_token_ids.shape
    completion_range = jnp.arange(completion_tokens, dtype=jnp.int32)[None, :]
    completion_mask = completion_range < batch.completion_lengths[:, None]

    prediction_indices = batch.prefix_lengths[:, None] - 1 + completion_range
    batch_indices = jnp.arange(batch.token_ids.shape[0], dtype=jnp.int32)[:, None]
    student_completion_logits = student_logits[batch_indices, prediction_indices, :]
    student_support_logits = jnp.take_along_axis(student_completion_logits, batch.support_token_ids, axis=-1)

    masked_teacher = jnp.where(batch.support_mask, batch.support_logits, -jnp.inf)
    masked_student = jnp.where(batch.support_mask, student_support_logits, -jnp.inf)
    safe_mask = completion_mask[:, :, None]

    teacher_log_probs = jax.nn.log_softmax(jnp.where(safe_mask, masked_teacher, 0.0), axis=-1)
    student_log_probs = jax.nn.log_softmax(jnp.where(safe_mask, masked_student, 0.0), axis=-1)
    teacher_probs = jnp.exp(teacher_log_probs)
    token_kl = jnp.sum(teacher_probs * (teacher_log_probs - student_log_probs), axis=-1)

    valid_tokens = completion_mask.sum(dtype=jnp.int32)
    valid_tokens = eqx.error_if(valid_tokens, valid_tokens <= 0, "Batch requires at least one completion token")
    loss = jnp.sum(token_kl * completion_mask.astype(token_kl.dtype)) / valid_tokens.astype(token_kl.dtype)

    student_top1 = jnp.argmax(masked_student, axis=-1)
    teacher_top1 = jnp.argmax(masked_teacher, axis=-1)
    top1_matches = jnp.sum(jnp.logical_and(student_top1 == teacher_top1, completion_mask), dtype=jnp.int32)

    return DistillBatchMetrics(loss=loss, kl_divergence=loss, valid_tokens=valid_tokens, top1_matches=top1_matches)


# ---------------------------------------------------------------------------
#  Gradient computation and optimizer
# ---------------------------------------------------------------------------


@eqx.filter_jit
def compute_distill_step_gradients(
    training_state: DistillTrainingState,
    student: Decoder,
    teacher: Decoder,
    batch: DistillBatch,
    config: DistillConfig,
    *,
    keychain: Keychain,
) -> tuple[eqx.Module, DistillBatchMetrics]:
    def loss_fn(master_weights: eqx.Module) -> tuple[Float[Array, ""], DistillBatchMetrics]:
        materialized = _materialize_student(student, replace(training_state, master_weights=master_weights), config.compute_dtype)
        metrics = compute_distill_batch_metrics(materialized, teacher, batch, config, keychain=keychain)
        return metrics.loss, metrics

    (_, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(training_state.master_weights)
    return grads, metrics


@eqx.filter_jit
def compute_trace_distill_step_gradients(
    training_state: DistillTrainingState,
    student: Decoder,
    batch: TraceDistillBatch,
    config: DistillConfig,
    *,
    keychain: Keychain,
) -> tuple[eqx.Module, DistillBatchMetrics]:
    def loss_fn(master_weights: eqx.Module) -> tuple[Float[Array, ""], DistillBatchMetrics]:
        materialized = _materialize_student(student, replace(training_state, master_weights=master_weights), config.compute_dtype)
        metrics = compute_trace_distill_batch_metrics(materialized, batch, config, keychain=keychain)
        return metrics.loss, metrics

    (_, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(training_state.master_weights)
    return grads, metrics


def apply_distill_gradients(
    optimizer_state: DistillOptimizerState,
    optimizer: optax.GradientTransformation,
    grads: eqx.Module,
) -> DistillOptimizerState:
    updates, new_opt_state = optimizer.update(
        grads,
        optimizer_state.optimizer_state,
        optimizer_state.training_state.master_weights,
    )
    updated_weights = optax.apply_updates(optimizer_state.training_state.master_weights, updates)
    return replace(
        optimizer_state,
        training_state=replace(optimizer_state.training_state, master_weights=updated_weights),
        optimizer_state=new_opt_state,
    )


@eqx.filter_jit(donate="all")
def apply_donated_distill_gradients(
    optimizer_state: DistillOptimizerState,
    optimizer: optax.GradientTransformation,
    grads: eqx.Module,
) -> DistillOptimizerState:
    return apply_distill_gradients(optimizer_state, optimizer, grads)
