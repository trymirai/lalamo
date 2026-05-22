# ruff: noqa: TC002, E501, F821, UP037, I001, RUF100
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, ClassVar, NamedTuple, Self

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from einops import rearrange
from huggingface_hub import snapshot_download
from jaxtyping import Array, Bool, DTypeLike, Float, Int, Key
from ml_dtypes import bfloat16

from lalamo.data.completion_features import FeatureRequest
from lalamo.initializer import EmptyInitializer
from lalamo.module import Keychain
from lalamo.modules import (
    Decoder,
    DenseMLP,
    DenseMLPConfig,
    EmbeddingBase,
    Linear,
    LinearConfig,
    Normalization,
    NormalizationConfig,
    PositionalEmbeddings,
    RoPE,
    SiLU,
    UnscaledRoPEConfig,
    UpcastMode,
)
from lalamo.sampling import SamplingPolicy
from lalamo.modules.token_mixer import State
from lalamo.modules.utils import call_vmapped
from lalamo.speculator.common import (
    LMState,
    PrefillResults,
    Speculator,
    SpeculatorBackend,
    write_speculator_artifact,
)
from lalamo.speculator.proposal import AcceptedProposal, ChainProposal
from lalamo.weight_matrix import FullPrecisionSpec, Layout

__all__ = [
    "DFlashBackend",
    "DFlashBackendConfig",
    "DFlashConfig",
    "DFlashDraftModel",
    "DFlashLMState",
    "DFlashSpeculator",
    "load_from_hf",
    "write_dflash_artifact",
]

DEFAULT_CONTEXT_CAPACITY = 4096


@dataclass(frozen=True)
class DFlashConfig:
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
        return self.num_fused_target_layers * self.hidden_size

    @classmethod
    def from_hf_config(
        cls,
        hf_config: dict[str, Any],
        precision: DTypeLike = jnp.bfloat16,
    ) -> DFlashConfig:
        dflash_config = hf_config["dflash_config"]
        return cls(
            num_layers=int(hf_config["num_hidden_layers"]),
            hidden_size=int(hf_config["hidden_size"]),
            intermediate_size=int(hf_config["intermediate_size"]),
            num_heads=int(hf_config["num_attention_heads"]),
            num_kv_heads=int(hf_config["num_key_value_heads"]),
            head_dim=int(hf_config["head_dim"]),
            block_size=int(hf_config["block_size"]),
            num_target_layers=int(hf_config["num_target_layers"]),
            target_layer_ids=tuple(int(i) for i in dflash_config["target_layer_ids"]),
            mask_token_id=int(dflash_config["mask_token_id"]),
            rope_theta=float(hf_config["rope_theta"]),
            rms_norm_eps=float(hf_config["rms_norm_eps"]),
            max_position_embeddings=int(hf_config["max_position_embeddings"]),
            vocab_size=int(hf_config["vocab_size"]),
            precision=precision,
        )


def qwen3_normalization_config(epsilon: float) -> NormalizationConfig:
    return NormalizationConfig(
        epsilon=epsilon,
        scale_offset=None,
        upcast_mode=UpcastMode.ONLY_NORMALIZATION,
        subtract_mean=False,
        has_biases=False,
    )


def qwen3_mlp_config() -> DenseMLPConfig:
    return DenseMLPConfig(
        linear_config=LinearConfig(),
        activation=SiLU(),
        has_up_biases=False,
        has_down_biases=False,
        gate_clipping=None,
        up_clipping=None,
    )


class DraftKVLayer(eqx.Module):
    keys: Float[Array, "capacity num_kv_heads head_dim"]
    values: Float[Array, "capacity num_kv_heads head_dim"]
    length: Int[Array, ""]

    @staticmethod
    def empty(
        capacity: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: DTypeLike,
    ) -> DraftKVLayer:
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


SAFETENSORS_DTYPES: dict[str, np.dtype[Any]] = {
    "F64": np.dtype(np.float64),
    "F32": np.dtype(np.float32),
    "F16": np.dtype(np.float16),
    "BF16": np.dtype(np.uint16),
    "I64": np.dtype(np.int64),
    "I32": np.dtype(np.int32),
    "I16": np.dtype(np.int16),
    "I8": np.dtype(np.int8),
    "U64": np.dtype(np.uint64),
    "U32": np.dtype(np.uint32),
    "U16": np.dtype(np.uint16),
    "U8": np.dtype(np.uint8),
    "BOOL": np.dtype(np.bool_),
}


class Qwen3DFlashAttention(eqx.Module):
    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    o_proj: Linear
    q_norm: Normalization
    k_norm: Normalization

    num_heads: int = eqx.field(static=True)
    num_kv_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def append_context(
        self,
        target_hidden_new: Float[Array, "ctx hidden"],
        rope_new: PositionalEmbeddings,
        kv: DraftKVLayer,
        valid_ctx_len: Int[Array, ""],
        *,
        keychain: Keychain,
    ) -> DraftKVLayer:
        k_keychain, v_keychain = keychain.split(2)
        cached = kv.length
        (k_new,) = call_vmapped(self.k_proj, target_hidden_new, keychain=k_keychain)
        (v_new,) = call_vmapped(self.v_proj, target_hidden_new, keychain=v_keychain)
        k_new = rearrange(k_new, "t (h d) -> t h d", h=self.num_kv_heads)
        v_new = rearrange(v_new, "t (h d) -> t h d", h=self.num_kv_heads)
        k_new = jax.vmap(jax.vmap(self.k_norm))(k_new)
        k_new = jax.vmap(rope_new.apply, in_axes=1, out_axes=1)(k_new)
        keys = jax.lax.dynamic_update_slice(kv.keys, k_new, (cached, 0, 0))
        values = jax.lax.dynamic_update_slice(kv.values, v_new, (cached, 0, 0))
        return DraftKVLayer(
            keys=keys,
            values=values,
            length=cached + valid_ctx_len,
        )

    def __call__(
        self,
        noise: Float[Array, "block hidden"],
        target_hidden_new: Float[Array, "ctx hidden"],
        rope_new: PositionalEmbeddings,
        kv: DraftKVLayer,
        valid_ctx_len: Int[Array, ""],
        *,
        keychain: Keychain,
    ) -> tuple[Float[Array, "block hidden"], DraftKVLayer]:
        block_size, _ = noise.shape
        ctx_new_len, _ = target_hidden_new.shape
        cached = kv.length
        q_keychain, k_keychain, v_keychain, o_keychain = keychain.split(4)

        (q,) = call_vmapped(self.q_proj, noise, keychain=q_keychain)
        q = rearrange(q, "t (h d) -> t h d", h=self.num_heads)
        q = jax.vmap(jax.vmap(self.q_norm))(q)
        q_rope = PositionalEmbeddings(
            cosines=rope_new.cosines[-block_size:],
            sines=rope_new.sines[-block_size:],
        )
        q = jax.vmap(q_rope.apply, in_axes=1, out_axes=1)(q)

        kv_input = jnp.concatenate([target_hidden_new, noise], axis=0)
        (k_new,) = call_vmapped(self.k_proj, kv_input, keychain=k_keychain)
        (v_new,) = call_vmapped(self.v_proj, kv_input, keychain=v_keychain)
        k_new = rearrange(k_new, "t (h d) -> t h d", h=self.num_kv_heads)
        v_new = rearrange(v_new, "t (h d) -> t h d", h=self.num_kv_heads)
        k_new = jax.vmap(jax.vmap(self.k_norm))(k_new)
        k_new = jax.vmap(rope_new.apply, in_axes=1, out_axes=1)(k_new)

        keys = jax.lax.dynamic_update_slice(kv.keys, k_new, (cached, 0, 0))
        values = jax.lax.dynamic_update_slice(kv.values, v_new, (cached, 0, 0))

        indices = jnp.arange(kv.capacity)
        in_cached = indices < cached
        in_ctx = (indices >= cached) & (indices < cached + valid_ctx_len)
        in_noise = (indices >= cached + ctx_new_len) & (indices < cached + ctx_new_len + block_size)
        valid = in_cached | in_ctx | in_noise
        attn_mask = jnp.broadcast_to(valid[None, :], (block_size, kv.capacity))

        attn_out = jax.nn.dot_product_attention(
            query=q[None],
            key=keys[None],
            value=values[None],
            mask=attn_mask[None, None, :, :],
            scale=self.head_dim**-0.5,
        )[0]
        attn_out = rearrange(attn_out, "t h d -> t (h d)")
        (out,) = call_vmapped(self.o_proj, attn_out, keychain=o_keychain)
        return out, DraftKVLayer(
            keys=keys,
            values=values,
            length=cached + valid_ctx_len,
        )


class Qwen3DFlashDecoderLayer(eqx.Module):
    self_attn: Qwen3DFlashAttention
    mlp: DenseMLP
    input_layernorm: Normalization
    post_attention_layernorm: Normalization

    def __call__(
        self,
        hidden: Float[Array, "block hidden"],
        target_hidden_new: Float[Array, "ctx hidden"],
        rope_new: PositionalEmbeddings,
        kv: DraftKVLayer,
        valid_ctx_len: Int[Array, ""],
        *,
        keychain: Keychain,
    ) -> tuple[Float[Array, "block hidden"], DraftKVLayer]:
        attn_keychain, mlp_keychain = keychain.split(2)

        residual = hidden
        h = jax.vmap(self.input_layernorm)(hidden)
        h, kv = self.self_attn(
            h,
            target_hidden_new,
            rope_new,
            kv,
            valid_ctx_len,
            keychain=attn_keychain,
        )
        h = residual + h

        residual = h
        h = jax.vmap(self.post_attention_layernorm)(h)
        h = call_vmapped(self.mlp.call_unbatched, h, keychain=mlp_keychain)
        return residual + h, kv


class DFlashDraftModel(eqx.Module):
    fc: Linear
    hidden_norm: Normalization
    layers: tuple[Qwen3DFlashDecoderLayer, ...]
    norm: Normalization
    rope: RoPE

    config: DFlashConfig = eqx.field(static=True)

    def init_kv_cache(
        self,
        capacity: int,
        dtype: DTypeLike | None = None,
    ) -> DraftKVState:
        dtype = dtype if dtype is not None else self.config.precision
        return tuple(
            DraftKVLayer.empty(
                capacity,
                self.config.num_kv_heads,
                self.config.head_dim,
                dtype,
            )
            for _ in range(self.config.num_layers)
        )

    def encode_context(
        self,
        target_hidden_new: Float[Array, "ctx fused_dim"],
        *,
        keychain: Keychain,
    ) -> Float[Array, "ctx hidden"]:
        (fc_out,) = call_vmapped(self.fc, target_hidden_new, keychain=keychain)
        return jax.vmap(self.hidden_norm)(fc_out)

    def append_context(
        self,
        target_hidden_new: Float[Array, "ctx fused_dim"],
        kv_state: DraftKVState,
        valid_ctx_len: Int[Array, ""],
        *,
        keychain: Keychain,
    ) -> DraftKVState:
        fc_keychain, layer_keychain = keychain.split(2)
        ctx = self.encode_context(target_hidden_new.astype(self.config.precision), keychain=fc_keychain)
        cached = kv_state[0].length
        ctx_idx = jnp.arange(target_hidden_new.shape[0], dtype=jnp.int32)
        position_ids = cached + jnp.where(ctx_idx < valid_ctx_len, ctx_idx, jnp.int32(0))
        rope_new = self.rope(position_ids)
        new_kv = []
        for layer, layer_kv, step_keychain in zip(
            self.layers,
            kv_state,
            layer_keychain.split(len(self.layers)),
            strict=True,
        ):
            new_kv.append(
                layer.self_attn.append_context(
                    ctx,
                    rope_new,
                    layer_kv,
                    valid_ctx_len,
                    keychain=step_keychain,
                ),
            )
        return tuple(new_kv)

    @eqx.filter_jit
    def __call__(
        self,
        noise_embedding: Float[Array, "block hidden"],
        target_hidden_new: Float[Array, "ctx fused_dim"],
        kv_state: DraftKVState,
        position_ids: Int[Array, " ctx_plus_block"],
        valid_ctx_len: Int[Array, ""],
        *,
        keychain: Keychain,
    ) -> tuple[Float[Array, "block hidden"], DraftKVState]:
        fc_keychain, layer_keychain = keychain.split(2)
        ctx = self.encode_context(target_hidden_new.astype(self.config.precision), keychain=fc_keychain)
        rope_new = self.rope(position_ids)

        hidden = noise_embedding
        new_kv = []
        for layer, layer_kv, step_keychain in zip(
            self.layers,
            kv_state,
            layer_keychain.split(len(self.layers)),
            strict=True,
        ):
            hidden, updated_kv = layer(
                hidden,
                ctx,
                rope_new,
                layer_kv,
                valid_ctx_len,
                keychain=step_keychain,
            )
            new_kv.append(updated_kv)

        hidden = jax.vmap(self.norm)(hidden)
        return hidden, tuple(new_kv)


class BlockDraft(NamedTuple):
    draft_logits: Float[Array, "draft vocab"]
    draft_kv_state: DraftKVState


@dataclass(frozen=True)
class DFlashBackendConfig:
    context_capacity: int = DEFAULT_CONTEXT_CAPACITY
    tree_budget: int | None = None
    keychain_seed: int = 0


class DFlashLMState(LMState):
    model: DFlashDraftModel
    context_capacity: int = eqx.field(static=True)
    keychain_seed: int = eqx.field(static=True)
    draft_kv_state: DraftKVState
    draft_context_start: Int[Array, " batch"]

    def update_status(
        self,
        kv_cache: State,
        processed_tree_logits: Float[Array, "batch nodes vocabulary"],
        accepted: AcceptedProposal,
        accepted_token_ids: Int[Array, "batch max_slots"],
        accepted_output_features: Float[Array, "batch max_slots channels"] | None,
        accepted_layer_outputs: tuple[Float[Array, "batch max_slots channels"], ...],
        stop_flags: Bool[Array, " batch"],
    ) -> DFlashLMState:
        next_state = super().update_status(
            kv_cache,
            processed_tree_logits,
            accepted,
            accepted_token_ids,
            accepted_output_features,
            accepted_layer_outputs,
            stop_flags,
        )
        if not accepted_layer_outputs:
            raise ValueError("DFlash requires accepted target layer outputs")
        accepted_hidden = jnp.concatenate(
            [layer_output.astype(jnp.float32) for layer_output in accepted_layer_outputs],
            axis=-1,
        )
        keychain = Keychain.init(self.keychain_seed, (accepted_token_ids.shape[0],))
        draft_kv_state = append_context_batched(
            self.model,
            accepted_hidden,
            self.draft_kv_state,
            accepted.num_compact_indices,
            keychain=keychain,
        )
        return eqx.tree_at(
            lambda lm_state: (lm_state.draft_kv_state, lm_state.draft_context_start),
            next_state,
            (draft_kv_state, self.draft_context_start),
        )


def dflash_target_hidden_from_prefill(
    model: DFlashDraftModel,
    prefill_results: PrefillResults,
    context_capacity: int,
) -> tuple[Float[Array, "batch ctx fused_dim"], Bool[Array, "batch ctx"]]:
    activation_trace = prefill_results.activation_trace
    if activation_trace is None:
        raise ValueError("DFlash requires target layer outputs")
    retained_length = jnp.minimum(prefill_results.input_lengths, context_capacity)
    slots = jnp.arange(context_capacity, dtype=jnp.int32)[None, :]
    positions = prefill_results.input_lengths[:, None] - retained_length[:, None] + slots
    layer_mask = slots < retained_length[:, None]
    positions = jnp.clip(positions, 0, prefill_results.input_token_ids.shape[1] - 1)
    batch_indices = jnp.arange(prefill_results.input_token_ids.shape[0], dtype=jnp.int32)[:, None]

    layer_outputs = [
        activation_trace.layer_results[layer_index].outputs[batch_indices, positions].astype(jnp.float32)
        for layer_index in model.config.target_layer_ids
    ]
    return jnp.concatenate(layer_outputs, axis=-1), layer_mask


def to_jax(array: np.ndarray, target_dtype: DTypeLike) -> Array:
    if array.dtype == np.dtype("uint16") and jnp.dtype(target_dtype) == jnp.dtype(jnp.bfloat16):
        array = array.view(bfloat16)
    result = jnp.asarray(array)
    if result.dtype != target_dtype:
        result = result.astype(target_dtype)
    return result


def load_safetensors_numpy(path: Path) -> dict[str, np.ndarray]:
    data = path.read_bytes()
    header_size = int.from_bytes(data[:8], "little")
    header_start = 8
    header_end = header_start + header_size
    header = json.loads(data[header_start:header_end])
    tensors: dict[str, np.ndarray] = {}
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        dtype_name = metadata["dtype"]
        dtype = SAFETENSORS_DTYPES[dtype_name]
        shape = tuple(int(dim) for dim in metadata["shape"])
        start, end = (int(offset) for offset in metadata["data_offsets"])
        buffer = memoryview(data)[header_end + start : header_end + end]
        tensors[name] = np.frombuffer(buffer, dtype=dtype).reshape(shape).copy()
    return tensors


def read_hf_state_dict(directory: Path) -> dict[str, np.ndarray]:
    shards = sorted(directory.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors in {directory}")
    state: dict[str, np.ndarray] = {}
    for shard in shards:
        state.update(load_safetensors_numpy(shard))
    return state


def build_linear_from_array(
    weights: Array,
    input_dim: int,
    output_dims: tuple[int, ...],
) -> Linear:
    output_dim = sum(output_dims)
    if weights.shape != (output_dim, input_dim):
        raise ValueError(f"linear weights have shape {weights.shape}, expected {(output_dim, input_dim)}")
    return Linear(
        config=LinearConfig(),
        weights=FullPrecisionSpec(Layout.OUTPUT_INPUT).compress(weights, is_sharded=False),
        biases=None,
        output_dims=output_dims,
    )


def build_linear(
    state_dict: dict[str, np.ndarray],
    key: str,
    input_dim: int,
    output_dim: int,
    precision: DTypeLike,
) -> Linear:
    return build_linear_from_array(
        to_jax(state_dict[key], precision),
        input_dim,
        (output_dim,),
    )


def build_normalization(
    state_dict: dict[str, np.ndarray],
    key: str,
    input_dim: int,
    epsilon: float,
    precision: DTypeLike,
) -> Normalization:
    scales = to_jax(state_dict[key], precision)
    if scales.shape != (input_dim,):
        raise ValueError(f"normalization scales have shape {scales.shape}, expected {(input_dim,)}")
    return Normalization(
        config=qwen3_normalization_config(epsilon),
        scales=scales,
        biases=None,
    )


def build_mlp(
    state_dict: dict[str, np.ndarray],
    prefix: str,
    model_dim: int,
    hidden_dim: int,
    precision: DTypeLike,
) -> DenseMLP:
    up = to_jax(state_dict[f"{prefix}.up_proj.weight"], precision)
    gate = to_jax(state_dict[f"{prefix}.gate_proj.weight"], precision)
    down = to_jax(state_dict[f"{prefix}.down_proj.weight"], precision)
    up_gate = jnp.concatenate([up, gate], axis=0)
    return DenseMLP(
        config=qwen3_mlp_config(),
        up_projection=build_linear_from_array(up_gate, model_dim, (hidden_dim, hidden_dim)),
        down_projection=build_linear_from_array(down, hidden_dim, (model_dim,)),
    )


def build_layer(
    state_dict: dict[str, np.ndarray],
    layer_idx: int,
    config: DFlashConfig,
) -> Qwen3DFlashDecoderLayer:
    prefix = f"layers.{layer_idx}"
    q_dim = config.num_heads * config.head_dim
    kv_dim = config.num_kv_heads * config.head_dim
    attn = Qwen3DFlashAttention(
        q_proj=build_linear(state_dict, f"{prefix}.self_attn.q_proj.weight", config.hidden_size, q_dim, config.precision),
        k_proj=build_linear(state_dict, f"{prefix}.self_attn.k_proj.weight", config.hidden_size, kv_dim, config.precision),
        v_proj=build_linear(state_dict, f"{prefix}.self_attn.v_proj.weight", config.hidden_size, kv_dim, config.precision),
        o_proj=build_linear(state_dict, f"{prefix}.self_attn.o_proj.weight", q_dim, config.hidden_size, config.precision),
        q_norm=build_normalization(state_dict, f"{prefix}.self_attn.q_norm.weight", config.head_dim, config.rms_norm_eps, config.precision),
        k_norm=build_normalization(state_dict, f"{prefix}.self_attn.k_norm.weight", config.head_dim, config.rms_norm_eps, config.precision),
        num_heads=config.num_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
    )
    return Qwen3DFlashDecoderLayer(
        self_attn=attn,
        mlp=build_mlp(state_dict, f"{prefix}.mlp", config.hidden_size, config.intermediate_size, config.precision),
        input_layernorm=build_normalization(state_dict, f"{prefix}.input_layernorm.weight", config.hidden_size, config.rms_norm_eps, config.precision),
        post_attention_layernorm=build_normalization(state_dict, f"{prefix}.post_attention_layernorm.weight", config.hidden_size, config.rms_norm_eps, config.precision),
    )


def load_from_hf(
    repo_or_path: str | Path,
    dtype: DTypeLike = jnp.bfloat16,
) -> tuple[DFlashConfig, DFlashDraftModel]:
    path = Path(repo_or_path)
    if not path.is_dir():
        path = Path(snapshot_download(repo_id=str(repo_or_path)))

    config = DFlashConfig.from_hf_config(
        json.loads((path / "config.json").read_text()),
        precision=dtype,
    )
    state_dict = read_hf_state_dict(path)
    initializer = EmptyInitializer(dtype)
    rope = UnscaledRoPEConfig(
        base=config.rope_theta,
        max_sequence_length=config.max_position_embeddings,
        head_dim=config.head_dim,
    ).init(initializer)

    model = DFlashDraftModel(
        fc=build_linear(state_dict, "fc.weight", config.fused_dim, config.hidden_size, config.precision),
        hidden_norm=build_normalization(state_dict, "hidden_norm.weight", config.hidden_size, config.rms_norm_eps, config.precision),
        layers=tuple(build_layer(state_dict, i, config) for i in range(config.num_layers)),
        norm=build_normalization(state_dict, "norm.weight", config.hidden_size, config.rms_norm_eps, config.precision),
        rope=rope,
        config=config,
    )
    return config, model


@eqx.filter_jit
def draft_block(
    model: DFlashDraftModel,
    embedding: EmbeddingBase[Any],
    noise_template: Int[Array, " block"],
    target_hidden_new: Float[Array, "ctx fused_dim"],
    draft_kv_state: DraftKVState,
    valid_ctx_len: Int[Array, ""],
    bonus: Int[Array, ""],
    *,
    keychain: Keychain,
) -> BlockDraft:
    cfg = model.config
    block_size = cfg.block_size
    embed_keychain, model_keychain, readout_keychain = keychain.split(3)

    if target_hidden_new.shape[0] < block_size:
        pad = block_size - target_hidden_new.shape[0]
        target_hidden_new = jnp.concatenate(
            [
                target_hidden_new,
                jnp.zeros((pad, cfg.fused_dim), dtype=target_hidden_new.dtype),
            ],
            axis=0,
        )
    padded_ctx_len, _ = target_hidden_new.shape

    noise_tokens = noise_template.at[0].set(bonus)
    noise_embedding = call_vmapped(
        embedding.embed,
        noise_tokens,
        keychain=embed_keychain,
    ).astype(cfg.precision)

    cached = draft_kv_state[0].length
    ctx_idx = jnp.arange(padded_ctx_len, dtype=jnp.int32)
    ctx_positions = cached + jnp.where(ctx_idx < valid_ctx_len, ctx_idx, jnp.int32(0))
    noise_positions = cached + valid_ctx_len + jnp.arange(block_size, dtype=jnp.int32)
    position_ids = jnp.concatenate([ctx_positions, noise_positions])

    hidden, new_draft_kv = model(
        noise_embedding,
        target_hidden_new.astype(cfg.precision),
        draft_kv_state,
        position_ids,
        valid_ctx_len,
        keychain=model_keychain,
    )
    draft_logits = call_vmapped(
        embedding.readout,
        hidden[1:],
        keychain=readout_keychain,
    )
    return BlockDraft(
        draft_logits=draft_logits,
        draft_kv_state=new_draft_kv,
    )


def empty_batched_draft_kv(
    batch_size: int,
    capacity: int,
    config: DFlashConfig,
) -> DraftKVState:
    return tuple(
        DraftKVLayer(
            keys=jnp.zeros(
                (batch_size, capacity, config.num_kv_heads, config.head_dim),
                dtype=config.precision,
            ),
            values=jnp.zeros(
                (batch_size, capacity, config.num_kv_heads, config.head_dim),
                dtype=config.precision,
            ),
            length=jnp.zeros((batch_size,), dtype=jnp.int32),
        )
        for _ in range(config.num_layers)
    )


@eqx.filter_jit
def append_context_batched(
    model: DFlashDraftModel,
    target_hidden_new: Float[Array, "batch ctx fused_dim"],
    draft_kv_state: DraftKVState,
    valid_ctx_len: Int[Array, " batch"],
    *,
    keychain: Keychain,
) -> DraftKVState:
    def append_row(
        row_hidden: Float[Array, "ctx fused_dim"],
        row_kv: DraftKVState,
        row_valid_ctx_len: Int[Array, ""],
        row_key: Key[Array, ""],
    ) -> DraftKVState:
        return model.append_context(
            row_hidden,
            row_kv,
            row_valid_ctx_len,
            keychain=Keychain(vmapped_keys=row_key, batch_key=keychain.batch_key),
        )

    return jax.vmap(append_row)(
        target_hidden_new,
        draft_kv_state,
        valid_ctx_len,
        keychain.vmapped_keys,
    )


@eqx.filter_jit
def rebuild_context_batched(
    model: DFlashDraftModel,
    target_hidden: Float[Array, "batch ctx fused_dim"],
    valid_ctx_len: Int[Array, " batch"],
    context_capacity: int,
    *,
    keychain: Keychain,
) -> DraftKVState:
    batch_size, _, _ = target_hidden.shape
    draft_kv_state = empty_batched_draft_kv(
        batch_size,
        context_capacity + model.config.block_size,
        model.config,
    )
    return append_context_batched(
        model,
        target_hidden,
        draft_kv_state,
        valid_ctx_len,
        keychain=keychain,
    )


@eqx.filter_jit
def draft_block_from_cache_batched(
    model: DFlashDraftModel,
    embedding: EmbeddingBase[Any],
    noise_template: Int[Array, " block"],
    draft_kv_state: DraftKVState,
    bonus_ids: Int[Array, " batch"],
    *,
    keychain: Keychain,
) -> BlockDraft:
    batch_size = bonus_ids.shape[0]
    empty_context = jnp.zeros((batch_size, 0, model.config.fused_dim), dtype=model.config.precision)
    valid_ctx_len = jnp.zeros((batch_size,), dtype=jnp.int32)

    def draft_row(
        row_hidden: Float[Array, "ctx fused_dim"],
        row_kv: DraftKVState,
        row_valid_ctx_len: Int[Array, ""],
        row_bonus_id: Int[Array, ""],
        row_key: Key[Array, ""],
    ) -> BlockDraft:
        return draft_block(
            model,
            embedding,
            noise_template,
            row_hidden,
            row_kv,
            row_valid_ctx_len,
            row_bonus_id,
            keychain=Keychain(vmapped_keys=row_key, batch_key=keychain.batch_key),
        )

    return jax.vmap(draft_row)(
        empty_context,
        draft_kv_state,
        valid_ctx_len,
        bonus_ids,
        keychain.vmapped_keys,
    )


@partial(
    jtu.register_dataclass,
    data_fields=["model", "embedding", "noise_template"],
    meta_fields=["context_capacity", "tree_budget", "keychain_seed"],
)
@dataclass(frozen=True, kw_only=True)
class DFlashSpeculator(Speculator):
    model: DFlashDraftModel
    embedding: EmbeddingBase[Any]
    noise_template: Int[Array, " block"]
    context_capacity: int = DEFAULT_CONTEXT_CAPACITY
    tree_budget: int = 16
    keychain_seed: int = 0

    @classmethod
    def create(
        cls,
        model: DFlashDraftModel,
        target_model: Decoder,
        context_capacity: int = DEFAULT_CONTEXT_CAPACITY,
        tree_budget: int | None = None,
        keychain_seed: int = 0,
    ) -> Self:
        if context_capacity < 1:
            raise ValueError("context_capacity must be positive")
        if target_model.embedding.model_dim != model.config.hidden_size:
            raise ValueError("target model hidden size does not match DFlash")
        if target_model.embedding.vocab_size != model.config.vocab_size:
            raise ValueError("target model vocab size does not match DFlash")
        resolved_tree_budget = model.config.block_size if tree_budget is None else int(tree_budget)
        if resolved_tree_budget < 1:
            raise ValueError("tree_budget must be positive")
        return cls(
            model=model,
            embedding=target_model.embedding,
            noise_template=jnp.full(
                (model.config.block_size,),
                model.config.mask_token_id,
                dtype=jnp.int32,
            ),
            context_capacity=int(context_capacity),
            tree_budget=resolved_tree_budget,
            keychain_seed=int(keychain_seed),
        )

    @property
    def feature_request(self) -> FeatureRequest:
        return FeatureRequest(completions=(), layer_indices=self.model.config.target_layer_ids)

    @property
    def max_step_tokens(self) -> int:
        return min(self.tree_budget, self.model.config.block_size)

    def init_state(
        self,
        prefill_results: PrefillResults,
        next_token_position: Int[Array, " batch"],
        sampling_policy: SamplingPolicy,
        gumbel_keys: Key[Array, " batch"],
    ) -> DFlashLMState:
        state = LMState.from_prefill(
            prefill_results,
            next_token_position,
            sampling_policy,
            gumbel_keys,
        )
        target_hidden, mask = dflash_target_hidden_from_prefill(
            self.model,
            prefill_results,
            self.context_capacity,
        )
        valid_ctx_len = jnp.sum(mask, axis=1).astype(jnp.int32)
        draft_kv_state = rebuild_context_batched(
            self.model,
            target_hidden,
            valid_ctx_len,
            self.context_capacity,
            keychain=Keychain.init(self.keychain_seed, (state.root_bonus_id.shape[0],)),
        )
        return DFlashLMState(
            kv_cache=state.kv_cache,
            next_token_position=state.next_token_position,
            root_bonus_id=state.root_bonus_id,
            root_sample_logits=state.root_sample_logits,
            sampling_policy=state.sampling_policy,
            gumbel_keys=state.gumbel_keys,
            output_lengths=state.output_lengths,
            stop_flags=state.stop_flags,
            model=self.model,
            context_capacity=self.context_capacity,
            keychain_seed=self.keychain_seed,
            draft_kv_state=draft_kv_state,
            draft_context_start=state.next_token_position - valid_ctx_len,
        )

    def draft(self, state: LMState) -> ChainProposal:
        if not isinstance(state, DFlashLMState):
            raise TypeError(f"DFlash requires DFlashLMState, got {type(state).__name__}")
        batch_size = state.root_bonus_id.shape[0]
        draft = draft_block_from_cache_batched(
            self.model,
            self.embedding,
            self.noise_template,
            state.draft_kv_state,
            state.root_bonus_id,
            keychain=Keychain.init(self.keychain_seed, (batch_size,)),
        )

        proposal_budget = min(self.tree_budget, self.model.config.block_size)
        if proposal_budget == 1:
            return state.create_chain_proposal()

        chain_depth = proposal_budget - 1
        return state.create_chain_proposal(draft.draft_logits[:, :chain_depth].astype(jnp.float32))


class DFlashBackend(SpeculatorBackend[DFlashBackendConfig]):
    name: ClassVar[str] = "dflash"
    config_type: ClassVar[type[Any]] = DFlashBackendConfig

    @classmethod
    def create_trainer(
        cls,
        config: DFlashBackendConfig,
        artifact_path: Path,
        target_model: Decoder,
    ) -> None:
        del config, artifact_path, target_model
        raise RuntimeError("DFlash uses pretrained HuggingFace draft checkpoints.")

    @classmethod
    def deserialize(
        cls,
        fields: tuple[Any, ...],
        target_model: Decoder,
    ) -> DFlashSpeculator:
        if len(fields) not in (1, 2, 3, 4):
            raise ValueError(
                "dflash artifact must contain repo_or_path, optional context_capacity, optional tree_budget, and optional keychain_seed",
            )
        repo_or_path = fields[0]
        if isinstance(repo_or_path, bytes):
            repo_or_path = repo_or_path.decode()
        if not isinstance(repo_or_path, str):
            raise TypeError("dflash repo_or_path must be a string")

        context_capacity = DEFAULT_CONTEXT_CAPACITY
        tree_budget = None
        keychain_seed = 0
        if len(fields) >= 2:
            context_capacity = int(fields[1])
        if len(fields) >= 3:
            tree_budget = None if fields[2] is None else int(fields[2])
        if len(fields) >= 4:
            keychain_seed = int(fields[3])

        _, model = load_from_hf(repo_or_path)
        return DFlashSpeculator.create(
            model,
            target_model,
            context_capacity=context_capacity,
            tree_budget=tree_budget,
            keychain_seed=keychain_seed,
        )


def write_dflash_artifact(
    path: Path | str,
    repo_or_path: str | Path,
    context_capacity: int = DEFAULT_CONTEXT_CAPACITY,
    tree_budget: int | None = None,
    keychain_seed: int = 0,
) -> None:
    write_speculator_artifact(
        path,
        DFlashBackend,
        str(repo_or_path),
        int(context_capacity),
        tree_budget,
        int(keychain_seed),
    )


BACKEND = DFlashBackend
