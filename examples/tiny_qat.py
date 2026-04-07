import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PRNGKeyArray

from lalamo.arrays import FullPrecisionArray
from lalamo.arrays.mlx import MLXQuantArray
from lalamo.model_import import import_model
from lalamo.models import ForwardPassConfig
from lalamo.arrays.base import GradientEstimator

from utils import Batch, kl_divergence, load_lmsys_conversations, make_batch


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
    batch_size = 4
    num_batches = num_train // batch_size
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(2e-5, weight_decay=0.01))
    opt_state = optimizer.init(eqx.filter(student, eqx.is_inexact_array))
    stochastic_fpc = ForwardPassConfig.init(gradient_estimator=GradientEstimator.STOCHASTIC).decoder

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
            logits = s(
                batch.token_ids[:, :-1], batch.positions[:, :-1], forward_pass_config=stochastic_fpc, key=key
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
