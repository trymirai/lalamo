from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree, dummy_array, require_array, require_tree
from lalamo.modules.common import PositionalEmbeddingSelector
from lalamo.modules.linear import LinearBase, LinearConfig
from lalamo.modules.normalization import Normalization, NormalizationConfig
from lalamo.modules.rope import PositionalEmbeddings

from .common import TokenMixerBase, TokenMixerConfigBase, TokenMixerResult
from .mamba import SeparableCausalConv, SeparableCausalConvConfig
from .state import DeltaNetStateLayer

__all__ = [
    "DeltaNetAttention",
    "DeltaNetAttentionConfig",
    "DeltaNetAttentionResult",
]


DeltaNetAttentionResult = TokenMixerResult[DeltaNetStateLayer]


@dataclass(frozen=True)
class DeltaNetAttentionConfig(TokenMixerConfigBase):
    in_proj_qkvz_config: LinearConfig
    in_proj_ba_config: LinearConfig
    conv_config: SeparableCausalConvConfig
    out_proj_config: LinearConfig
    norm_config: NormalizationConfig

    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    kernel_size: int

    @property
    def rope_dim(self) -> None:
        return None

    @property
    def key_dim(self) -> int:
        return self.num_k_heads * self.head_k_dim

    @property
    def value_dim(self) -> int:
        return self.num_v_heads * self.head_v_dim

    @property
    def conv_dim(self) -> int:
        return self.key_dim * 2 + self.value_dim

    def random_init(
        self,
        model_dim: int,
        *,
        key: PRNGKeyArray,
    ) -> "DeltaNetAttention":
        qkvz_key, ba_key, conv_key, out_key = jax.random.split(key, 4)
        in_proj_qkvz = self.in_proj_qkvz_config.random_init(
            input_dim=model_dim,
            output_dims=(self.key_dim * 2 + self.value_dim * 2,),
            has_biases=False,
            key=qkvz_key,
        )
        in_proj_ba = self.in_proj_ba_config.random_init(
            input_dim=model_dim,
            output_dims=(self.num_v_heads * 2,),
            has_biases=False,
            key=ba_key,
        )
        conv = self.conv_config.random_init(self.conv_dim, self.kernel_size, key=conv_key)
        out_proj = self.out_proj_config.random_init(
            input_dim=self.value_dim,
            output_dims=(model_dim,),
            has_biases=False,
            key=out_key,
        )
        norm = self.norm_config.init(self.head_v_dim)
        dt_bias = jnp.zeros((self.num_v_heads,), dtype=in_proj_qkvz.activation_precision)
        a_log = jnp.zeros((self.num_v_heads,), dtype=in_proj_qkvz.activation_precision)
        return DeltaNetAttention(
            self,
            in_proj_qkvz=in_proj_qkvz,
            in_proj_ba=in_proj_ba,
            conv=conv,
            out_proj=out_proj,
            norm=norm,
            dt_bias=dt_bias,
            a_log=a_log,
        )

    def empty(
        self,
        model_dim: int,
    ) -> "DeltaNetAttention":
        in_proj_qkvz = self.in_proj_qkvz_config.empty(
            input_dim=model_dim,
            output_dims=(self.key_dim * 2 + self.value_dim * 2,),
            has_biases=False,
        )
        in_proj_ba = self.in_proj_ba_config.empty(
            input_dim=model_dim,
            output_dims=(self.num_v_heads * 2,),
            has_biases=False,
        )
        conv = self.conv_config.empty(self.conv_dim, self.kernel_size)
        out_proj = self.out_proj_config.empty(
            input_dim=self.value_dim,
            output_dims=(model_dim,),
            has_biases=False,
        )
        norm = self.norm_config.empty(self.head_v_dim)
        dt_bias = dummy_array((self.num_v_heads,), dtype=in_proj_qkvz.activation_precision)
        a_log = dummy_array((self.num_v_heads,), dtype=in_proj_qkvz.activation_precision)
        return DeltaNetAttention(
            self,
            in_proj_qkvz=in_proj_qkvz,
            in_proj_ba=in_proj_ba,
            conv=conv,
            out_proj=out_proj,
            norm=norm,
            dt_bias=dt_bias,
            a_log=a_log,
        )


class DeltaNetAttention(TokenMixerBase[DeltaNetAttentionConfig, DeltaNetStateLayer]):
    in_proj_qkvz: LinearBase
    in_proj_ba: LinearBase
    conv: SeparableCausalConv
    out_proj: LinearBase
    norm: Normalization
    dt_bias: Float[Array, " heads"]
    a_log: Float[Array, " heads"]

    @property
    def activation_precision(self) -> DTypeLike:
        return self.in_proj_qkvz.activation_precision

    @property
    def model_dim(self) -> int:
        return self.in_proj_qkvz.input_dim

    @property
    def positional_embedding_selector(self) -> PositionalEmbeddingSelector:
        return PositionalEmbeddingSelector.NONE

    @eqx.filter_jit
    def __call__(
        self,
        inputs: Float[Array, "suffix_tokens channels"],
        positional_embeddings: PositionalEmbeddings | None,
        state: DeltaNetStateLayer | None = None,
        return_updated_state: bool = False,
        length_without_padding: Int[Array, ""] | int | None = None,  # noqa: ARG002
    ) -> DeltaNetAttentionResult:
        if positional_embeddings is not None:
            raise ValueError("Positional embeddings are not supported for DeltaNetAttention.")

        outputs = jnp.zeros((inputs.shape[0], self.model_dim), dtype=inputs.dtype)

        if return_updated_state:
            if state is not None:
                updated_state = state
            else:
                updated_state = DeltaNetStateLayer.init(
                    self.config.kernel_size,
                    self.config.conv_dim,
                    self.config.num_v_heads,
                    self.config.head_k_dim,
                    self.config.head_v_dim,
                    self.activation_precision,
                )
        else:
            updated_state = None

        return TokenMixerResult(outputs, updated_state)

    def init_static_state(self, capacity: int) -> DeltaNetStateLayer:  # noqa: ARG002
        return DeltaNetStateLayer.init(
            self.config.kernel_size,
            self.config.conv_dim,
            self.config.num_v_heads,
            self.config.head_k_dim,
            self.config.head_v_dim,
            self.activation_precision,
        )

    def export_weights(self) -> ParameterTree:
        return {
            "in_proj_qkvz": self.in_proj_qkvz.export_weights(),
            "in_proj_ba": self.in_proj_ba.export_weights(),
            "conv": self.conv.export_weights(),
            "out_proj": self.out_proj.export_weights(),
            "norm": self.norm.export_weights(),
            "dt_bias": self.dt_bias,
            "a_log": self.a_log,
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            in_proj_qkvz=self.in_proj_qkvz.import_weights(require_tree(weights["in_proj_qkvz"])),
            in_proj_ba=self.in_proj_ba.import_weights(require_tree(weights["in_proj_ba"])),
            conv=self.conv.import_weights(require_tree(weights["conv"])),
            out_proj=self.out_proj.import_weights(require_tree(weights["out_proj"])),
            norm=self.norm.import_weights(require_tree(weights["norm"])),
            dt_bias=require_array(weights["dt_bias"]),
            a_log=require_array(weights["a_log"]),
        )
