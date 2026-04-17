import dataclasses
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, ClassVar, Self, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from huggingface_hub import snapshot_download
from jaxtyping import Array, DTypeLike, Float, Int
from ml_dtypes import bfloat16
from safetensors.numpy import load_file

from lalamo.modules.activations import SiLU
from lalamo.modules.common import ForwardPassMode
from lalamo.modules.decoder import Decoder, DecoderResult
from lalamo.modules.linear import FullPrecisionLinear, FullPrecisionLinearConfig
from lalamo.modules.mlp import DenseMLP, DenseMLPConfig
from lalamo.modules.normalization import Normalization, NormalizationConfig, UpcastMode
from lalamo.modules.rope import PositionalEmbeddings, RoPE, UnscaledRoPEConfig
from lalamo.modules.token_mixers.state.common import State
from lalamo.modules.token_mixers.state.kv_cache import StaticKVCacheLayer
from lalamo.speculator.common import LMState, SamplerConfig, SpeculationStep, Speculator
from lalamo.speculator.utils import extract_activations

__all__ = [
    "BlockProposal",
    "DFlashConfig",
    "DFlashDraftModel",
    "DFlashSpeculator",
    "DraftKVLayer",
    "DraftKVState",
    "Qwen3DFlashAttention",
    "Qwen3DFlashDecoderLayer",
    "load_from_hf",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DFlashConfig:
    """Mirror of the relevant HF `config.json` fields for a DFlash checkpoint.

    See z-lab/Qwen3-{4B,8B}-DFlash-b16 for canonical values.
    """

    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    block_size: int
    num_target_layers: int
    target_layer_ids: tuple[int, ...]
    mask_token_id: int
    rope_theta: float
    rms_norm_eps: float
    max_position_embeddings: int
    vocab_size: int
    precision: DTypeLike

    @property
    def num_fused_target_layers(self) -> int:
        return len(self.target_layer_ids)

    @property
    def fused_dim(self) -> int:
        """Channel dim of the concatenated target context (input to ``fc``)."""
        return self.num_fused_target_layers * self.hidden_size

    @classmethod
    def from_hf_config(cls, hf: dict[str, Any], precision: DTypeLike = jnp.bfloat16) -> "DFlashConfig":
        dcfg = hf["dflash_config"]
        return cls(
            num_layers=int(hf["num_hidden_layers"]),
            hidden_size=int(hf["hidden_size"]),
            intermediate_size=int(hf["intermediate_size"]),
            num_heads=int(hf["num_attention_heads"]),
            num_kv_heads=int(hf["num_key_value_heads"]),
            head_dim=int(hf["head_dim"]),
            block_size=int(hf["block_size"]),
            num_target_layers=int(hf["num_target_layers"]),
            target_layer_ids=tuple(int(i) for i in dcfg["target_layer_ids"]),
            mask_token_id=int(dcfg["mask_token_id"]),
            rope_theta=float(hf["rope_theta"]),
            rms_norm_eps=float(hf["rms_norm_eps"]),
            max_position_embeddings=int(hf["max_position_embeddings"]),
            vocab_size=int(hf["vocab_size"]),
            precision=precision,
        )


def qwen3_normalization_config(eps: float, precision: DTypeLike) -> NormalizationConfig:
    return NormalizationConfig(
        scale_precision=precision,
        accumulation_precision=jnp.float32,
        epsilon=eps,
        scale_offset=None,
        upcast_mode=UpcastMode.ONLY_NORMALIZATION,
        subtract_mean=False,
        use_bias=False,
    )


def qwen3_mlp_config(precision: DTypeLike) -> DenseMLPConfig:
    return DenseMLPConfig(
        linear_config=FullPrecisionLinearConfig(precision=precision),
        activation=SiLU(),
        has_up_biases=False,
        has_down_biases=False,
        gate_clipping=None,
        up_clipping=None,
    )


# ---------------------------------------------------------------------------
# Draft KV cache
# ---------------------------------------------------------------------------


class DraftKVLayer(eqx.Module):
    """Static-capacity KV cache for one draft attention layer.

    ``length`` counts committed entries. Slots ``[length, capacity)`` may
    contain stale K/V from the previously written noise block and must be
    masked out by the attention call.
    """

    keys: Float[Array, "capacity num_kv_heads head_dim"]
    values: Float[Array, "capacity num_kv_heads head_dim"]
    length: Int[Array, ""]

    @staticmethod
    def empty(capacity: int, num_kv_heads: int, head_dim: int, dtype: DTypeLike) -> "DraftKVLayer":
        return DraftKVLayer(
            keys=jnp.zeros((capacity, num_kv_heads, head_dim), dtype=dtype),
            values=jnp.zeros((capacity, num_kv_heads, head_dim), dtype=dtype),
            length=jnp.int32(0),
        )

    @property
    def capacity(self) -> int:
        capacity, _, _ = self.keys.shape
        return capacity


type DraftKVState = tuple[DraftKVLayer, ...]


# ---------------------------------------------------------------------------
# Attention: Q from noise only, K/V from [ctx_new ; noise], non-causal window
# ---------------------------------------------------------------------------


class Qwen3DFlashAttention(eqx.Module):
    """One DFlash attention layer.

    Attention is non-causal across the full window (cached + new ctx + noise):
    every noise position attends to everything valid, but writing noise K/V
    into the cache is ephemeral — ``length`` is advanced only by ``ctx_new``.
    """

    q_proj: FullPrecisionLinear
    k_proj: FullPrecisionLinear
    v_proj: FullPrecisionLinear
    o_proj: FullPrecisionLinear
    q_norm: Normalization
    k_norm: Normalization

    num_heads: int = eqx.field(static=True)
    num_kv_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __call__(
        self,
        noise: Float[Array, "block_size hidden"],
        target_hidden_new: Float[Array, "ctx_new hidden"],
        rope_new: PositionalEmbeddings,
        kv: DraftKVLayer,
        valid_ctx_len: Int[Array, ""],
    ) -> tuple[Float[Array, "block_size hidden"], DraftKVLayer]:
        block_size, _ = noise.shape
        ctx_new_len, _ = target_hidden_new.shape
        cached = kv.length

        (q,) = jax.vmap(self.q_proj)(noise)
        q = rearrange(q, "t (h d) -> t h d", h=self.num_heads)
        q = jax.vmap(jax.vmap(self.q_norm))(q)
        q_rope = PositionalEmbeddings(
            cosines=rope_new.cosines[-block_size:],
            sines=rope_new.sines[-block_size:],
        )
        q = jax.vmap(q_rope.apply, in_axes=1, out_axes=1)(q)

        kv_input = jnp.concatenate([target_hidden_new, noise], axis=0)
        (k_new,) = jax.vmap(self.k_proj)(kv_input)
        (v_new,) = jax.vmap(self.v_proj)(kv_input)
        k_new = rearrange(k_new, "t (h d) -> t h d", h=self.num_kv_heads)
        v_new = rearrange(v_new, "t (h d) -> t h d", h=self.num_kv_heads)
        k_new = jax.vmap(jax.vmap(self.k_norm))(k_new)
        k_new = jax.vmap(rope_new.apply, in_axes=1, out_axes=1)(k_new)

        # Cached entries already had RoPE baked in at write time; never re-apply.
        keys = jax.lax.dynamic_update_slice(kv.keys, k_new, (cached, 0, 0))
        values = jax.lax.dynamic_update_slice(kv.values, v_new, (cached, 0, 0))

        # Three valid regions: committed cache, valid prefix of the fresh ctx,
        # and the full noise block. Padded ctx slots between valid_ctx_len and
        # ctx_new_len are masked out (and get overwritten by the next call's
        # write, since ``length`` below advances by valid_ctx_len only).
        capacity = kv.capacity
        indices = jnp.arange(capacity)
        in_cached = indices < cached
        in_ctx = (indices >= cached) & (indices < cached + valid_ctx_len)
        in_noise = (indices >= cached + ctx_new_len) & (indices < cached + ctx_new_len + block_size)
        valid = in_cached | in_ctx | in_noise
        attn_mask = jnp.broadcast_to(valid[None, :], (block_size, capacity))

        # jax.nn.dot_product_attention broadcasts when num_kv_heads < num_heads,
        # so we pass the compressed K/V directly instead of repeating.
        attn_out = jax.nn.dot_product_attention(
            query=q[None],
            key=keys[None],
            value=values[None],
            mask=attn_mask[None, None, :, :],
            scale=self.head_dim**-0.5,
        )[0]
        attn_out = rearrange(attn_out, "t h d -> t (h d)")
        (out,) = jax.vmap(self.o_proj)(attn_out)

        return out, DraftKVLayer(
            keys=keys,
            values=values,
            length=cached + valid_ctx_len,
        )


# ---------------------------------------------------------------------------
# Decoder layer (pre-norm attn + pre-norm MLP, residual on both)
# ---------------------------------------------------------------------------


class Qwen3DFlashDecoderLayer(eqx.Module):
    self_attn: Qwen3DFlashAttention
    mlp: DenseMLP
    input_layernorm: Normalization
    post_attention_layernorm: Normalization

    def __call__(
        self,
        hidden: Float[Array, "block_size hidden"],
        target_hidden_new: Float[Array, "ctx_new hidden"],
        rope_new: PositionalEmbeddings,
        kv: DraftKVLayer,
        valid_ctx_len: Int[Array, ""],
    ) -> tuple[Float[Array, "block_size hidden"], DraftKVLayer]:
        residual = hidden
        h = jax.vmap(self.input_layernorm)(hidden)
        h, kv = self.self_attn(h, target_hidden_new, rope_new, kv, valid_ctx_len)
        h = residual + h

        residual = h
        h = jax.vmap(self.post_attention_layernorm)(h)
        h = jax.vmap(self.mlp.call_unbatched)(h)
        return residual + h, kv


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class DFlashDraftModel(eqx.Module):
    """Qwen3-shaped block-diffusion drafter.

    Pure function of ``(noise_embedding, target_hidden_new, kv_state, position_ids)``.
    Does not own ``embed_tokens`` or ``lm_head`` — the caller supplies those
    from the target model.
    """

    fc: FullPrecisionLinear
    hidden_norm: Normalization
    layers: tuple[Qwen3DFlashDecoderLayer, ...]
    norm: Normalization
    rope: RoPE

    config: DFlashConfig = eqx.field(static=True)

    def init_kv_cache(self, capacity: int, dtype: DTypeLike | None = None) -> DraftKVState:
        dtype = dtype if dtype is not None else self.config.precision
        return tuple(
            DraftKVLayer.empty(capacity, self.config.num_kv_heads, self.config.head_dim, dtype)
            for _ in range(self.config.num_layers)
        )

    @eqx.filter_jit
    def __call__(
        self,
        noise_embedding: Float[Array, "block_size hidden"],
        target_hidden_new: Float[Array, "ctx_new fused_dim"],
        kv_state: DraftKVState,
        position_ids: Int[Array, " ctx_new_plus_block"],
        valid_ctx_len: Int[Array, ""],
    ) -> tuple[Float[Array, "block_size hidden"], DraftKVState]:
        (fc_out,) = jax.vmap(self.fc)(target_hidden_new)
        ctx = jax.vmap(self.hidden_norm)(fc_out)

        rope_new = self.rope(position_ids)

        hidden = noise_embedding
        new_kv: list[DraftKVLayer] = []
        for layer, layer_kv in zip(self.layers, kv_state, strict=True):
            hidden, updated_kv = layer(hidden, ctx, rope_new, layer_kv, valid_ctx_len)
            new_kv.append(updated_kv)

        hidden = jax.vmap(self.norm)(hidden)
        return hidden, tuple(new_kv)


# ---------------------------------------------------------------------------
# HuggingFace safetensors loader
# ---------------------------------------------------------------------------


def to_jax(arr: np.ndarray, target_dtype: DTypeLike) -> Array:
    """Convert a safetensors numpy array to JAX, handling uint16-stored bf16."""
    if arr.dtype == np.dtype("uint16") and target_dtype == jnp.bfloat16:
        arr = arr.view(bfloat16)
    out = jnp.asarray(arr)
    if out.dtype != target_dtype:
        out = out.astype(target_dtype)
    return out


def read_hf_state_dict(directory: Path) -> dict[str, np.ndarray]:
    shards = sorted(directory.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors in {directory}")
    state: dict[str, np.ndarray] = {}
    for shard in shards:
        state.update(load_file(shard))
    return state


def build_linear(
    sd: dict[str, np.ndarray],
    key: str,
    input_dim: int,
    output_dim: int,
    precision: DTypeLike,
) -> FullPrecisionLinear:
    empty = FullPrecisionLinearConfig(precision=precision).empty(
        input_dim=input_dim,
        output_dims=(output_dim,),
        has_biases=False,
    )
    return replace(empty, weights=to_jax(sd[key], precision), biases=None)


def build_normalization(
    sd: dict[str, np.ndarray],
    key: str,
    input_dim: int,
    eps: float,
    precision: DTypeLike,
) -> Normalization:
    empty = qwen3_normalization_config(eps, precision).empty(input_dim=input_dim)
    return replace(empty, scales=to_jax(sd[key], precision))


def build_mlp(
    sd: dict[str, np.ndarray],
    prefix: str,
    model_dim: int,
    hidden_dim: int,
    precision: DTypeLike,
) -> DenseMLP:
    # lalamo convention (see model_import/loaders/huggingface.py load_mlp):
    # up_projection is a single fused linear whose output is split into
    # (up, gate). Concatenate along the output axis in that order.
    up = to_jax(sd[f"{prefix}.up_proj.weight"], precision)
    gate = to_jax(sd[f"{prefix}.gate_proj.weight"], precision)
    down = to_jax(sd[f"{prefix}.down_proj.weight"], precision)
    up_gate = jnp.concatenate([up, gate], axis=0)

    empty = qwen3_mlp_config(precision).empty(model_dim=model_dim, hidden_dim=hidden_dim)
    up_proj = replace(empty.up_projection, weights=up_gate, biases=None)
    down_proj = replace(empty.down_projection, weights=down, biases=None)
    return replace(empty, up_projection=up_proj, down_projection=down_proj)


def build_layer(sd: dict[str, np.ndarray], layer_idx: int, config: DFlashConfig) -> Qwen3DFlashDecoderLayer:
    prefix = f"layers.{layer_idx}"
    q_dim = config.num_heads * config.head_dim
    kv_dim = config.num_kv_heads * config.head_dim
    attn = Qwen3DFlashAttention(
        q_proj=build_linear(sd, f"{prefix}.self_attn.q_proj.weight", config.hidden_size, q_dim, config.precision),
        k_proj=build_linear(sd, f"{prefix}.self_attn.k_proj.weight", config.hidden_size, kv_dim, config.precision),
        v_proj=build_linear(sd, f"{prefix}.self_attn.v_proj.weight", config.hidden_size, kv_dim, config.precision),
        o_proj=build_linear(sd, f"{prefix}.self_attn.o_proj.weight", q_dim, config.hidden_size, config.precision),
        q_norm=build_normalization(
            sd,
            f"{prefix}.self_attn.q_norm.weight",
            config.head_dim,
            config.rms_norm_eps,
            config.precision,
        ),
        k_norm=build_normalization(
            sd,
            f"{prefix}.self_attn.k_norm.weight",
            config.head_dim,
            config.rms_norm_eps,
            config.precision,
        ),
        num_heads=config.num_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
    )
    return Qwen3DFlashDecoderLayer(
        self_attn=attn,
        mlp=build_mlp(sd, f"{prefix}.mlp", config.hidden_size, config.intermediate_size, config.precision),
        input_layernorm=build_normalization(
            sd,
            f"{prefix}.input_layernorm.weight",
            config.hidden_size,
            config.rms_norm_eps,
            config.precision,
        ),
        post_attention_layernorm=build_normalization(
            sd,
            f"{prefix}.post_attention_layernorm.weight",
            config.hidden_size,
            config.rms_norm_eps,
            config.precision,
        ),
    )


def load_from_hf(
    repo_or_path: str | Path,
    dtype: DTypeLike = jnp.bfloat16,
) -> tuple[DFlashConfig, DFlashDraftModel]:
    """Load a DFlash draft model from a local directory or a HuggingFace repo id.

    Examples:
        cfg, model = load_from_hf("z-lab/Qwen3-4B-DFlash-b16")
        cfg, model = load_from_hf("/path/to/Qwen3-8B-DFlash-b16")
    """
    path = Path(repo_or_path)
    if not path.is_dir():
        path = Path(snapshot_download(repo_id=str(repo_or_path)))

    config = DFlashConfig.from_hf_config(json.loads((path / "config.json").read_text()), precision=dtype)
    sd = read_hf_state_dict(path)

    rope = UnscaledRoPEConfig(
        precision=dtype,
        base=config.rope_theta,
        max_sequence_length=config.max_position_embeddings,
        head_dim=config.head_dim,
    ).init()

    model = DFlashDraftModel(
        fc=build_linear(sd, "fc.weight", config.fused_dim, config.hidden_size, config.precision),
        hidden_norm=build_normalization(
            sd,
            "hidden_norm.weight",
            config.hidden_size,
            config.rms_norm_eps,
            config.precision,
        ),
        layers=tuple(build_layer(sd, i, config) for i in range(config.num_layers)),
        norm=build_normalization(sd, "norm.weight", config.hidden_size, config.rms_norm_eps, config.precision),
        rope=rope,
        config=config,
    )
    return config, model


# ---------------------------------------------------------------------------
# Speculator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockProposal:
    """One block-diffusion draft: the anchor bonus plus ``block_size - 1`` drafts,
    arranged as a flat chain. No tree — DFlash verifies linearly and accepts the
    longest prefix on which the draft argmax matches the target's sample.
    """

    tokens: Int[np.ndarray, " block_size"]


@dataclass(frozen=True, kw_only=True)
class DFlashSpeculator(Speculator[BlockProposal]):
    """DFlash block-diffusion speculator, directly on top of :class:`Speculator`.

    This is a chain speculator — it does *not* inherit ``TreeSpeculator`` because
    DFlash uses neither tree attention nor the Gumbel-max shared-seed coupling.
    The acceptance rule follows the paper (and ``run_reference.py``):

    * draft tokens: ``argmax(draft_logits)`` — greedy, no noise;
    * target tokens: ``argmax`` at T=0, multinomial on ``softmax(logits / T)`` at T>0;
    * accept the longest prefix of ``draft[1:] == target_sample[:-1]``;
    * ``next_bonus = target_sample[num_accepted]``.
    """

    name: ClassVar[str] = "dflash"

    model: DFlashDraftModel
    draft_kv_state: DraftKVState
    temperature: float
    rng_key: Int[Array, "2"]

    @classmethod
    def create(
        cls,
        *,
        decoder: Decoder,
        model: DFlashDraftModel,
        config: SamplerConfig,
        eos_set: frozenset[int],
        temperature: float = 0.0,
    ) -> Self:
        """Construct a speculator with an empty drafter KV cache — sized
        against the concrete prompt length by :meth:`prefill`.
        """
        return cls(
            decoder=decoder,
            config=config,
            eos_set=eos_set,
            model=model,
            draft_kv_state=(),
            temperature=temperature,
            rng_key=jax.random.PRNGKey(config.seed),
            trace_layer_outputs=model.config.target_layer_ids,
            trace_output_norm=False,
            prefill_hidden_range=None,
        )

    @classmethod
    def deserialize_impl(cls, data: bytes, **kwargs: object) -> Self:
        """Treat ``data`` as a HuggingFace repo id or local model directory,
        fetch weights via :func:`load_from_hf` (uses the HF cache on repeat
        runs), and build through :meth:`create`. Weights are never serialised
        locally — the .bin file is just a pointer. Write one with, e.g.,
        ``echo -n 'z-lab/Qwen3-4B-DFlash-b16' > dflash_4b.bin``.

        Required kwargs: ``decoder``, ``config``, ``eos_set``. Optional:
        ``temperature`` (default 0.0). Extra kwargs forwarded by the generic
        ``lalamo speculator eval`` CLI (``width``, ``depth``) are ignored.
        """
        decoder = kwargs["decoder"]
        config = kwargs["config"]
        eos_set = kwargs["eos_set"]
        temperature = kwargs.get("temperature", 0.0)
        assert isinstance(decoder, Decoder)
        assert isinstance(config, SamplerConfig)
        assert isinstance(eos_set, frozenset)
        assert isinstance(temperature, (int, float))

        _, model = load_from_hf(data.decode().strip())
        return cls.create(
            decoder=decoder,
            model=model,
            config=config,
            eos_set=cast("frozenset[int]", eos_set),
            temperature=float(temperature),
        )

    @property
    def generation_capacity(self) -> int:
        return self.config.max_tokens + self.model.config.block_size + 16

    def prefill(self, prompt_ids: list[int]) -> tuple[Self, LMState]:
        state = self.decoder.init_static_state(1, self.generation_capacity + len(prompt_ids))
        prefix = jnp.array([prompt_ids], dtype=jnp.int32)
        fwd = self.decoder(
            prefix,
            jnp.arange(len(prompt_ids))[None, :],
            state,
            return_updated_state=True,
            return_activation_trace=True,
        )
        logits = fwd.logits[0, -1]
        assert fwd.activation_trace is not None
        assert fwd.updated_state is not None
        layer_outputs, output_norm = extract_activations(
            fwd.activation_trace,
            sample_index=0,
            positions=slice(None),
            trace_layer_outputs=self.trace_layer_outputs,
            trace_output_norm=self.trace_output_norm,
        )
        rng_key, sub_key = jax.random.split(self.rng_key)
        bonus = self.sample_token(logits, sub_key)
        draft_kv = self.model.init_kv_cache(capacity=self.generation_capacity + len(prompt_ids))
        new_self = dataclasses.replace(self, rng_key=rng_key, draft_kv_state=draft_kv)
        lm = LMState(
            kv_cache=fwd.updated_state,
            layer_outputs=layer_outputs,
            output_norm=output_norm,
            logits=logits,
            position=len(prompt_ids),
            bonus=bonus,
        )
        return new_self, lm

    def draft(self, lm: LMState) -> BlockProposal:
        proposal, _ = self.generate_block(lm)
        return proposal

    def step(self, lm: LMState) -> tuple[Self, LMState, SpeculationStep]:
        proposal, new_draft_kv = self.generate_block(lm)
        fwd = self.chain_forward(lm, proposal)
        rng_key, verify_key = jax.random.split(self.rng_key)
        target_samples = self.sample_block(fwd.logits[0], verify_key)
        accepted, next_bonus = self.accept_chain(proposal, target_samples)
        new_lm = self.build_next_state(lm, fwd, accepted, next_bonus)
        final_self = dataclasses.replace(self, rng_key=rng_key, draft_kv_state=new_draft_kv)
        return (
            final_self,
            new_lm,
            SpeculationStep(accepted=accepted, bonus=next_bonus),
        )

    # --- sampling --------------------------------------------------------------

    def sample_token(self, logits: Float[Array, " vocab"], key: Int[Array, "2"]) -> int:
        """One-shot sampler used by prefill for the first bonus."""
        if self.temperature < 1e-5:
            return int(jnp.argmax(logits))
        return int(jax.random.categorical(key, logits.astype(jnp.float32) / self.temperature))

    def sample_block(
        self,
        logits: Float[Array, "block_size vocab"],
        key: Int[Array, "2"],
    ) -> Int[np.ndarray, " block_size"]:
        """Per-position target sample used during verification.

        Paper convention: greedy at T=0, multinomial at T>0. Emphatically NOT
        Gumbel-max coupled with the drafter — the drafter is argmax-only.
        """
        if self.temperature < 1e-5:
            return np.asarray(jnp.argmax(logits, axis=-1))
        block_size, _ = logits.shape
        keys = jax.random.split(key, block_size)
        scaled = logits.astype(jnp.float32) / self.temperature
        sampled = jax.vmap(lambda k, row: jax.random.categorical(k, row))(keys, scaled)
        return np.asarray(sampled)

    # --- verification ----------------------------------------------------------

    def chain_forward(self, lm: LMState, proposal: BlockProposal) -> DecoderResult:
        """Plain causal forward — no ``attention_parent_indices`` → standard
        causal mask. The target decoder writes K/V for ``block_size`` new
        positions starting at ``cache_len``.
        """
        first_layer = lm.kv_cache[0]
        assert isinstance(first_layer, StaticKVCacheLayer)
        cache_len = first_layer.current_length[0]  # traced scalar — no host sync
        block_size = int(proposal.tokens.shape[0])
        tok_ids = jnp.array(proposal.tokens[None, :], dtype=jnp.int32)
        positions = (cache_len + jnp.arange(block_size, dtype=jnp.int32))[None, :]
        return self.decoder(
            tok_ids,
            positions,
            lm.kv_cache,
            return_updated_state=True,
            return_activation_trace=True,
            forward_pass_mode=ForwardPassMode.SINGLE_TOKEN,
        )

    @staticmethod
    def accept_chain(
        proposal: BlockProposal,
        target_samples: Int[np.ndarray, " block_size"],
    ) -> tuple[list[int], int]:
        """Exact-match accept: longest prefix where ``draft[i+1] == target[i]``.

        Returns ``(accepted_drafts, next_bonus)``. ``next_bonus`` is the target
        sample at the first rejected position (or at the end of the block if
        everything was accepted).
        """
        matches = (proposal.tokens[1:] == target_samples[:-1]).astype(np.int32)
        num_accepted = int(np.sum(np.cumprod(matches)))
        accepted = proposal.tokens[1 : 1 + num_accepted].tolist()
        next_bonus = int(target_samples[num_accepted])
        return accepted, next_bonus

    def build_next_state(
        self,
        lm: LMState,
        fwd: DecoderResult,
        accepted: list[int],
        next_bonus: int,
    ) -> LMState:
        """Commit the accepted prefix into the target KV cache and package
        the new :class:`LMState`.
        """
        first_layer = lm.kv_cache[0]
        assert isinstance(first_layer, StaticKVCacheLayer)
        cache_len = first_layer.current_length[0]  # traced scalar
        total_kept = 1 + len(accepted)
        block_size = self.model.config.block_size

        assert fwd.updated_state is not None
        kept_indices = jnp.arange(total_kept, dtype=jnp.int32)
        kept_padded = jnp.concatenate(
            [kept_indices, jnp.zeros(block_size - total_kept, dtype=jnp.int32)],
        )
        new_layers: list[StaticKVCacheLayer] = []
        for layer in fwd.updated_state:
            if not isinstance(layer, StaticKVCacheLayer):
                raise TypeError(
                    f"DFlash requires StaticKVCacheLayer, got {type(layer).__name__}",
                )
            new_layers.append(
                layer.compact(cache_len, kept_padded, jnp.int32(total_kept), block_size),
            )
        new_kv = State(new_layers)

        assert fwd.activation_trace is not None
        layer_outputs, output_norm = extract_activations(
            fwd.activation_trace,
            sample_index=0,
            positions=jnp.arange(total_kept, dtype=jnp.int32),
            trace_layer_outputs=self.trace_layer_outputs,
            trace_output_norm=self.trace_output_norm,
        )
        return LMState(
            kv_cache=new_kv,
            layer_outputs=layer_outputs,
            output_norm=output_norm,
            logits=jnp.array(fwd.logits[0, len(accepted)]),
            position=int(cache_len) + total_kept,
            bonus=next_bonus,
        )

    # --- drafter forward -------------------------------------------------------

    def generate_block(self, lm: LMState) -> tuple[BlockProposal, DraftKVState]:
        cfg = self.model.config
        block_size = cfg.block_size

        if not lm.layer_outputs or len(lm.layer_outputs) != cfg.num_fused_target_layers:
            raise ValueError(
                f"DFlash expects {cfg.num_fused_target_layers} traced target-layer outputs on "
                f"LMState, got {len(lm.layer_outputs)}.",
            )
        target_hidden_new = jnp.concatenate(lm.layer_outputs, axis=-1)
        valid_ctx_len, _ = target_hidden_new.shape

        # Pad the fresh target context to ``block_size`` (the maximum over
        # post-prefill steps) so the drafter forward sees a single shape for
        # every speculation iteration. The prefill step can exceed block_size
        # (prompt_len rows); leave it untouched so it triggers its own
        # one-shot compile.
        if valid_ctx_len < block_size:
            pad = block_size - valid_ctx_len
            target_hidden_new = jnp.concatenate(
                [
                    target_hidden_new,
                    jnp.zeros((pad, cfg.fused_dim), dtype=target_hidden_new.dtype),
                ],
                axis=0,
            )
        padded_ctx_len, _ = target_hidden_new.shape

        noise_tokens = jnp.array(
            [lm.bonus, *([cfg.mask_token_id] * (block_size - 1))],
            dtype=jnp.int32,
        )
        noise_embedding = self.decoder.embedding.embed(noise_tokens)

        # Keep ``cached`` as a traced scalar so the drafter's position_ids
        # construction stays on-device — no host sync every iteration.
        #
        # Positions are laid out [real_ctx; padded_ctx (masked); noise] but the
        # noise must receive absolute positions ``cached + valid_ctx_len + i``
        # — not ``cached + padded_ctx_len + i`` — otherwise RoPE puts the
        # noise anchor past the real context and token predictions fall out
        # of lockstep with the target, collapsing acceptance after step 1.
        # The padded-ctx slots get a dummy position (0) and are masked out
        # inside the attention call.
        cached = self.draft_kv_state[0].length
        ctx_idx = jnp.arange(padded_ctx_len, dtype=jnp.int32)
        ctx_positions = cached + jnp.where(ctx_idx < jnp.int32(valid_ctx_len), ctx_idx, jnp.int32(0))
        noise_positions = cached + jnp.int32(valid_ctx_len) + jnp.arange(block_size, dtype=jnp.int32)
        position_ids = jnp.concatenate([ctx_positions, noise_positions])

        hidden, new_draft_kv = self.model(
            noise_embedding,
            target_hidden_new.astype(cfg.precision),
            self.draft_kv_state,
            position_ids,
            jnp.int32(valid_ctx_len),
        )

        # hidden[0] corresponds to the anchor (known bonus); hidden[1:] predicts
        # the next ``block_size - 1`` tokens. Draft = argmax (greedy), per paper.
        draft_logits = jax.vmap(self.decoder.embedding.readout)(hidden[1:])
        draft_tokens = np.asarray(jnp.argmax(draft_logits, axis=-1))

        tokens = np.concatenate([np.array([lm.bonus], dtype=np.int32), draft_tokens.astype(np.int32)])
        return BlockProposal(tokens=tokens), new_draft_kv
