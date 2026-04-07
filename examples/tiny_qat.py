import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PRNGKeyArray

from lalamo.arrays import FullPrecisionArray
from lalamo.arrays.awq import AWQQuantArray
from lalamo.arrays.base import GradientEstimator
from lalamo.arrays.mlx import MLXQuantArray
from lalamo.arrays.quantization_helpers import stochastic_quantize_to_grid
from lalamo.model_import import import_model
from lalamo.models import ForwardPassConfig

from utils import Batch, kl_divergence, load_lmsys_conversations, make_batch

is_quantized = lambda x: isinstance(x, (MLXQuantArray, AWQQuantArray))


def perturb_quantized_weights(model: eqx.Module, key: PRNGKeyArray) -> eqx.Module:
    leaves = jax.tree.leaves(model, is_leaf=is_quantized)
    keys = iter(jax.random.split(key, sum(1 for l in leaves if is_quantized(l))))

    def perturb_leaf(leaf: object) -> object:
        if not is_quantized(leaf):
            return leaf
        rounded = stochastic_quantize_to_grid(leaf.weights, leaf.bits, next(keys))
        ste = leaf.weights + jax.lax.stop_gradient(rounded - leaf.weights)
        return eqx.tree_at(lambda l: l.weights, leaf, ste)

    return jax.tree.map(perturb_leaf, model, is_leaf=is_quantized)


def quantize_model(model: eqx.Module, group_size: int = 64, bits: int = 4) -> eqx.Module:
    def convert(leaf: object) -> object:
        if isinstance(leaf, FullPrecisionArray):
            return MLXQuantArray.compress(leaf.weights, group_size=group_size, bits=bits)
        return leaf

    return jax.tree.map(convert, model, is_leaf=lambda x: isinstance(x, FullPrecisionArray))


def eval_kl(teacher: eqx.Module, student: eqx.Module, batch: Batch) -> float:
    ids, pos, mask = batch.token_ids[:, :-1], batch.positions[:, :-1], batch.loss_mask[:, 1:]
    teacher_logits = teacher(ids, pos).logits
    student_logits = student(ids, pos).logits
    return float(kl_divergence(teacher_logits, student_logits, mask))


def main() -> None:
    language_model = import_model("LiquidAI/LFM2-350M").model
    teacher = language_model.model
    student = quantize_model(teacher)

    conversations = load_lmsys_conversations(num_sequences=80)
    eval_batch = make_batch(conversations[:8], language_model.message_processor)

    print(f"KL before: {eval_kl(teacher, student, eval_batch):.4f}")

    num_train = 48
    batch_size = 2
    num_batches = num_train // batch_size
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(2e-5, weight_decay=0.01))
    opt_state = optimizer.init(eqx.filter(student, eqx.is_inexact_array))
    deterministic_fpc = ForwardPassConfig.init(gradient_estimator=GradientEstimator.DETERMINISTIC).decoder

    @eqx.filter_jit
    def step(
        student: eqx.Module,
        opt_state: optax.OptState,
        teacher_logits: Float[Array, "batch seq_len vocab"],
        batch: Batch,
        key: PRNGKeyArray,
    ) -> tuple[eqx.Module, optax.OptState, Float[Array, ""]]:
        mask = batch.loss_mask[:, 1:]

        def loss_fn(s: eqx.Module) -> Float[Array, ""]:
            perturbed = perturb_quantized_weights(s, key)
            logits = perturbed(
                batch.token_ids[:, :-1], batch.positions[:, :-1], forward_pass_config=deterministic_fpc
            ).logits
            return kl_divergence(teacher_logits, logits, mask)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(student)
        updates, new_opt_state = optimizer.update(grads, opt_state, student)
        return eqx.apply_updates(student, updates), new_opt_state, loss

    key = jax.random.key(42)
    for epoch in range(3):
        for batch_idx, batch_start in enumerate(range(8, 8 + num_train, batch_size)):
            batch = make_batch(conversations[batch_start : batch_start + batch_size], language_model.message_processor)
            teacher_logits = teacher(batch.token_ids[:, :-1], batch.positions[:, :-1]).logits
            key, subkey = jax.random.split(key)
            student, opt_state, loss = step(student, opt_state, teacher_logits, batch, subkey)
            print(f"epoch={epoch} batch={batch_idx}/{num_batches} kl={float(loss):.4f}")

    print(f"KL after: {eval_kl(teacher, student, eval_batch):.4f}")


if __name__ == "__main__":
    main()
