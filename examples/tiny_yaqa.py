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
    return isinstance(leaf, FullPrecisionArray) and leaf.weights.ndim == 2


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

    def quantize(w: Float[Array, "out_channels in_channels"]) -> Float[Array, "out_channels in_channels"]:
        return jnp.clip(jnp.round((w - weight_min) / scale), 0, qmax) * scale + weight_min

    def step(_: int, rounded: Float[Array, "out_channels in_channels"]) -> Float[Array, "out_channels in_channels"]:
        error = original_weights - rounded
        correction = output_feedback.T @ error @ input_feedback + output_feedback.T @ error + error @ input_feedback
        return quantize(original_weights + correction)

    return jax.lax.fori_loop(0, num_iters, step, quantize(original_weights))


def init_hessian(leaf: object) -> HessianPair | object:
    if not is_quantizable_weight(leaf):
        return leaf
    out_ch, in_ch = leaf.weights.shape
    return HessianPair(
        input_hessian=jnp.zeros((in_ch, in_ch), dtype=jnp.float32),
        output_hessian=jnp.zeros((out_ch, out_ch), dtype=jnp.float32),
    )


def accumulate_hessian(hessian_leaf: HessianPair | object, grad_leaf: object) -> HessianPair | object:
    if not isinstance(hessian_leaf, HessianPair):
        return hessian_leaf
    g = grad_leaf.weights.astype(jnp.float32)
    return HessianPair(
        input_hessian=hessian_leaf.input_hessian + g.T @ g,
        output_hessian=hessian_leaf.output_hessian + g @ g.T,
    )


def main() -> None:
    language_model = import_model("LiquidAI/LFM2-350M").model
    decoder = language_model.model

    texts = load_lmsys_calibration_texts(num_sequences=50)
    eval_ids, eval_pos = tokenize_batch(texts[:8], language_model.message_processor, seq_len=128)
    calib_batches = [tokenize_batch([t], language_model.message_processor, seq_len=128) for t in texts[8:40]]
    key = jax.random.key(0)

    @eqx.filter_jit
    def fisher_grads(
        d: eqx.Module, token_ids: Int[Array, "1 seq"], positions: Int[Array, "1 seq"], seq_key: PRNGKeyArray
    ) -> eqx.Module:
        def loss_fn(d: eqx.Module) -> Float[Array, ""]:
            logits = d(token_ids, positions).logits
            return optax.softmax_cross_entropy_with_integer_labels(
                logits, jax.random.categorical(seq_key, logits)
            ).mean()

        return eqx.filter_grad(loss_fn)(d)

    num_calib = len(calib_batches)
    rounded_decoder = decoder

    for layer_idx in range(len(decoder.transformer.layers)):
        layer = decoder.transformer.layers[layer_idx]
        hessians = jax.tree.map(init_hessian, layer, is_leaf=is_quantizable_weight)

        for i, (ids, pos) in enumerate(calib_batches):
            grads = fisher_grads(decoder, ids, pos, jax.random.fold_in(key, i))
            hessians = jax.tree.map(
                accumulate_hessian, hessians, grads.transformer.layers[layer_idx], is_leaf=is_weight_or_hessian
            )

        def apply_yaqa(h: HessianPair | object, w: object) -> object:
            if not isinstance(h, HessianPair):
                return w
            raw = w.weights.astype(jnp.float32)
            out_ch, in_ch = raw.shape
            out_factor, _ = ldl(h.output_hessian / (num_calib * in_ch))
            in_factor, _ = ldl(h.input_hessian / (num_calib * out_ch))
            return FullPrecisionArray(weights=yaqa_round(raw, out_factor, in_factor).astype(w.weights.dtype))

        rounded_layer = jax.tree.map(apply_yaqa, hessians, layer, is_leaf=is_weight_or_hessian)
        rounded_decoder = eqx.tree_at(lambda d, i=layer_idx: d.transformer.layers[i], rounded_decoder, rounded_layer)
        del hessians, rounded_layer
        gc.collect()
        print(f"  Layer {layer_idx + 1}/{len(decoder.transformer.layers)}")

    print(f"YAQA KL:  {eval_kl(decoder, rounded_decoder, eval_ids, eval_pos):.4f}")

    naive = jax.tree.map(
        lambda l: FullPrecisionArray(weights=round_to_grid(l.weights.astype(jnp.float32)).astype(l.weights.dtype))
        if is_quantizable_weight(l)
        else l,
        decoder,
        is_leaf=is_quantizable_weight,
    )
    print(f"Naive KL: {eval_kl(decoder, naive, eval_ids, eval_pos):.4f}")


if __name__ == "__main__":
    main()
