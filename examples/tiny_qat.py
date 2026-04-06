import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PRNGKeyArray

from lalamo.arrays import FullPrecisionArray
from lalamo.arrays.mlx import MLXQuantArray
from lalamo.model_import import import_model
from lalamo.models import ForwardPassConfig

from utils import Batch, kl_divergence, load_lmsys_conversations, make_batch


def quantize_model(model: eqx.Module, group_size: int = 64, bits: int = 4) -> eqx.Module:
    def convert(leaf: object) -> object:
        if isinstance(leaf, FullPrecisionArray):
            return MLXQuantArray.from_raw(leaf.raw, group_size=group_size, bits=bits)
        return leaf

    return jax.tree.map(convert, model, is_leaf=lambda x: isinstance(x, FullPrecisionArray))


def eval_kl(teacher_decoder: eqx.Module, student_decoder: eqx.Module, batch: Batch) -> float:
    input_ids, input_pos = batch.token_ids[:, :-1], batch.positions[:, :-1]
    mask = batch.loss_mask[:, 1:]
    teacher_logits = teacher_decoder(input_ids, input_pos).logits
    student_logits = student_decoder(input_ids, input_pos).logits
    return float(kl_divergence(teacher_logits, student_logits, mask))


def main() -> None:
    language_model = import_model("LiquidAI/LFM2-350M").model
    teacher_decoder = language_model.model
    student_decoder = quantize_model(language_model.model)

    conversations = load_lmsys_conversations(num_sequences=1020)
    eval_batch = make_batch(conversations[:20], language_model.message_processor)

    print(f"KL before: {eval_kl(teacher_decoder, student_decoder, eval_batch):.4f}")

    num_train = len(conversations) - 50
    num_batches = num_train // 8
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(2e-5, weight_decay=0.01),
    )
    opt_state = optimizer.init(eqx.filter(student_decoder, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(
        student_decoder: eqx.Module,
        opt_state: optax.OptState,
        teacher_logits: Float[Array, "batch seq_len vocab"],
        batch: Batch,
        key: PRNGKeyArray,
    ) -> tuple[eqx.Module, optax.OptState, Float[Array, ""]]:
        student_fwd = ForwardPassConfig.init(stochastic_quantize_key=key).decoder
        mask = batch.loss_mask[:, 1:]

        def loss_fn(current_student: eqx.Module) -> Float[Array, ""]:
            student_logits = current_student(
                batch.token_ids[:, :-1], batch.positions[:, :-1], forward_pass_config=student_fwd
            ).logits
            return kl_divergence(teacher_logits, student_logits, mask)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(student_decoder)
        updates, new_opt_state = optimizer.update(grads, opt_state, student_decoder)
        return eqx.apply_updates(student_decoder, updates), new_opt_state, loss

    key = jax.random.key(42)
    for epoch in range(20):
        for batch_idx, batch_start in enumerate(range(20, 20 + num_train, 8)):
            batch = make_batch(
                conversations[batch_start : batch_start + 8], language_model.message_processor
            )
            input_ids, input_pos = batch.token_ids[:, :-1], batch.positions[:, :-1]
            teacher_logits = teacher_decoder(input_ids, input_pos).logits
            key, subkey = jax.random.split(key)
            student_decoder, opt_state, loss = step(student_decoder, opt_state, teacher_logits, batch, subkey)
            if batch_idx % 25 == 0:
                print(f"epoch={epoch} batch={batch_idx}/{num_batches} kl={float(loss):.4f}")

    print(f"KL after: {eval_kl(teacher_decoder, student_decoder, eval_batch):.4f}")


if __name__ == "__main__":
    main()
