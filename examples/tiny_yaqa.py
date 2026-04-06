"""Tiny YAQA: Model-Preserving Adaptive Rounding (Tseng et al., 2025)

Sketch B computes Kronecker-factored Hessians per layer via weight gradients:
  HI = E[(nabla_W l)^T (nabla_W l)] / m,  HO = E[(nabla_W l)(nabla_W l)^T] / n

YAQA rounds with symmetric input/output feedback (Eq. 5):
  W = Q(W* + L'_O^T Delta L'_I + L'_O^T Delta + Delta L'_I)
"""

import gc
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, Int, PRNGKeyArray

from lalamo.arrays import FullPrecisionArray
from lalamo.model_import import import_model

from utils import eval_kl, load_lmsys_calibration_texts, tokenize_batch


class HessianPair(eqx.Module):
    input_hessian: Float[Array, "in_channels in_channels"]
    output_hessian: Float[Array, "out_channels out_channels"]


def is_quantizable_weight(leaf: object) -> bool:
    return isinstance(leaf, FullPrecisionArray) and leaf.raw.ndim == 2


is_weight_or_hessian = lambda x: isinstance(x, (FullPrecisionArray, HessianPair))


def ldl(hessian: Float[Array, "dim dim"]) -> tuple[Float[Array, "dim dim"], Float[Array, " dim"]]:
    size = hessian.shape[0]
    damping = jnp.maximum(jnp.diag(hessian).max() * 1e-2, 1e-4)
    cholesky = jnp.linalg.cholesky(hessian + damping * jnp.eye(size, dtype=hessian.dtype))
    diagonal = jnp.diag(cholesky)
    return cholesky / diagonal, diagonal**2


def round_to_grid(
    weights: Float[Array, "out_channels in_channels"], bits: int = 4
) -> Float[Array, "out_channels in_channels"]:
    qmax = 2**bits - 1
    weight_min = weights.min()
    scale = jnp.maximum((weights.max() - weight_min) / qmax, jnp.finfo(weights.dtype).eps)
    return jnp.clip(jnp.round((weights - weight_min) / scale), 0, qmax) * scale + weight_min


@partial(jax.jit, static_argnames=("bits", "num_iters"))
def yaqa_round(
    original_weights: Float[Array, "out_channels in_channels"],
    output_factor: Float[Array, "out_channels out_channels"],
    input_factor: Float[Array, "in_channels in_channels"],
    *,
    bits: int = 4,
    num_iters: int = 1,
) -> Float[Array, "out_channels in_channels"]:
    out_channels, in_channels = original_weights.shape
    output_feedback = output_factor - jnp.eye(out_channels, dtype=output_factor.dtype)
    input_feedback = input_factor - jnp.eye(in_channels, dtype=input_factor.dtype)

    qmax = 2**bits - 1
    weight_min = original_weights.min()
    scale = jnp.maximum((original_weights.max() - weight_min) / qmax, jnp.finfo(original_weights.dtype).eps)

    def quantize(weights: Float[Array, "out_channels in_channels"]) -> Float[Array, "out_channels in_channels"]:
        return jnp.clip(jnp.round((weights - weight_min) / scale), 0, qmax) * scale + weight_min

    def step(_: int, rounded: Float[Array, "out_channels in_channels"]) -> Float[Array, "out_channels in_channels"]:
        error = original_weights - rounded
        correction = output_feedback.T @ error @ input_feedback + output_feedback.T @ error + error @ input_feedback
        return quantize(original_weights + correction)

    return jax.lax.fori_loop(0, num_iters, step, quantize(original_weights))


def init_hessian(leaf: object) -> HessianPair | object:
    if not is_quantizable_weight(leaf):
        return leaf
    return HessianPair(
        input_hessian=jnp.zeros((leaf.raw.shape[1],) * 2, dtype=jnp.float32),
        output_hessian=jnp.zeros((leaf.raw.shape[0],) * 2, dtype=jnp.float32),
    )


def accumulate_hessian(hessian_leaf: HessianPair | object, grad_leaf: object) -> HessianPair | object:
    if not isinstance(hessian_leaf, HessianPair):
        return hessian_leaf
    weight_grad = grad_leaf.raw.astype(jnp.float32)
    return HessianPair(
        input_hessian=hessian_leaf.input_hessian + weight_grad.T @ weight_grad,
        output_hessian=hessian_leaf.output_hessian + weight_grad @ weight_grad.T,
    )


def main() -> None:
    language_model = import_model("LiquidAI/LFM2-350M").model
    original_decoder = language_model.model

    texts = load_lmsys_calibration_texts(num_sequences=300)
    eval_token_ids, eval_positions = tokenize_batch(texts[:8], language_model.message_processor, seq_len=128)
    calib_batches = [
        tokenize_batch([text], language_model.message_processor, seq_len=128)
        for text in texts[8:264]
    ]
    key = jax.random.key(0)

    @eqx.filter_jit
    def fisher_grads(
        decoder: eqx.Module,
        token_ids: Int[Array, "batch seq_len"],
        positions: Int[Array, "batch seq_len"],
        seq_key: PRNGKeyArray,
    ) -> eqx.Module:
        def loss_fn(current_decoder: eqx.Module) -> Float[Array, ""]:
            logits = current_decoder(token_ids, positions).logits
            sampled = jax.random.categorical(seq_key, logits)
            return optax.softmax_cross_entropy_with_integer_labels(logits, sampled).mean()

        return eqx.filter_grad(loss_fn)(decoder)

    num_batches = len(calib_batches)
    decoder = original_decoder

    for layer_idx in range(len(original_decoder.transformer.layers)):
        layer = original_decoder.transformer.layers[layer_idx]
        layer_hessians = jax.tree.map(init_hessian, layer, is_leaf=is_quantizable_weight)

        for batch_idx, (token_ids, positions) in enumerate(calib_batches):
            grads = fisher_grads(original_decoder, token_ids, positions, jax.random.fold_in(key, batch_idx))
            grad_layer = grads.transformer.layers[layer_idx]
            layer_hessians = jax.tree.map(
                accumulate_hessian, layer_hessians, grad_layer, is_leaf=is_weight_or_hessian
            )

        def yaqa_round_leaf(hessian_leaf: HessianPair | object, model_leaf: object) -> object:
            if not isinstance(hessian_leaf, HessianPair):
                return model_leaf
            original_weights = model_leaf.raw.astype(jnp.float32)
            out_channels, in_channels = original_weights.shape
            input_factor, _ = ldl(hessian_leaf.input_hessian / (num_batches * out_channels))
            output_factor, _ = ldl(hessian_leaf.output_hessian / (num_batches * in_channels))
            rounded = yaqa_round(original_weights, output_factor, input_factor)
            return FullPrecisionArray(raw=rounded.astype(model_leaf.raw.dtype))

        rounded_layer = jax.tree.map(yaqa_round_leaf, layer_hessians, layer, is_leaf=is_weight_or_hessian)
        decoder = eqx.tree_at(lambda d, i=layer_idx: d.transformer.layers[i], decoder, rounded_layer)

        del layer_hessians, rounded_layer
        gc.collect()
        print(f"  Layer {layer_idx + 1}/{len(original_decoder.transformer.layers)}")

    print(f"YAQA KL:  {eval_kl(original_decoder, decoder, eval_token_ids, eval_positions):.4f}")

    naive_decoder = jax.tree.map(
        lambda leaf: FullPrecisionArray(raw=round_to_grid(leaf.raw.astype(jnp.float32)).astype(leaf.raw.dtype))
        if is_quantizable_weight(leaf)
        else leaf,
        original_decoder,
        is_leaf=is_quantizable_weight,
    )
    print(f"Naive KL: {eval_kl(original_decoder, naive_decoder, eval_token_ids, eval_positions):.4f}")


if __name__ == "__main__":
    main()
