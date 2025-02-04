from dataclasses import dataclass
from typing import NamedTuple

import jax
from jax import vmap
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from fartsovka.common import ParameterDict

from .attention import Attention, AttentionConfig
from .common import FartsovkaModule
from .kv_cache import KVCacheLayerSlice
from .mlp import MLP, MLPConfig
from .normalization import RMSNorm, RMSNormConfig
from .rope import PositionalEmbeddings

__all__ = [
    "DecoderLayer",
    "DecoderLayerConfig",
    "DecoderLayerOutput",
]


class DecoderLayerOutput(NamedTuple):
    output: Float[Array, "suffix_tokens channels"]
    kv_cache: KVCacheLayerSlice | None


@dataclass
class DecoderLayerConfig:
    pre_attention_norm_config: RMSNormConfig
    attention_config: AttentionConfig
    post_attention_norm_config: RMSNormConfig | None
    pre_mlp_norm_config: RMSNormConfig
    mlp_config: MLPConfig
    post_mlp_norm_config: RMSNormConfig | None

    def random_init(
        self,
        model_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_groups: int,
        head_dim: int,
        attention_scale: float | None,
        sliding_window_size: int | None,
        *,
        key: PRNGKeyArray,
    ) -> "DecoderLayer":
        attention_key, mlp_key = jax.random.split(key)
        pre_attention_norm = self.pre_attention_norm_config.init(model_dim)
        attention = self.attention_config.random_init(
            model_dim=model_dim,
            num_heads=num_heads,
            num_groups=num_groups,
            head_dim=head_dim,
            scale=attention_scale,
            sliding_window_size=sliding_window_size,
            key=attention_key,
        )
        if self.post_attention_norm_config is not None:
            post_attention_norm = self.post_attention_norm_config.init(model_dim)
        else:
            post_attention_norm = None
        pre_mlp_norm = self.pre_mlp_norm_config.init(model_dim)
        mlp = self.mlp_config.random_init(model_dim, hidden_dim, key=mlp_key)
        if self.post_mlp_norm_config is not None:
            post_mlp_norm = self.post_mlp_norm_config.init(model_dim)
        else:
            post_mlp_norm = None
        return DecoderLayer(
            config=self,
            pre_attention_norm=pre_attention_norm,
            attention=attention,
            post_attention_norm=post_attention_norm,
            pre_mlp_norm=pre_mlp_norm,
            mlp=mlp,
            post_mlp_norm=post_mlp_norm,
        )


class DecoderLayer(FartsovkaModule[DecoderLayerConfig]):
    pre_attention_norm: RMSNorm
    attention: Attention
    post_attention_norm: RMSNorm | None
    pre_mlp_norm: RMSNorm
    mlp: MLP
    post_mlp_norm: RMSNorm | None

    def __post_init__(self) -> None:
        model_dim = self.pre_attention_norm.input_dim
        if self.attention.model_dim != model_dim:
            raise ValueError(
                f"Attention model dim {self.attention.model_dim} does not match"
                f" the first normalization layer dim {model_dim}",
            )
        if self.post_attention_norm is not None and self.post_attention_norm.input_dim != model_dim:
            raise ValueError(
                f"Post attention normalization dim {self.post_attention_norm.input_dim} does not match"
                f" the first normalization layer dim {model_dim}",
            )
        if self.pre_mlp_norm.input_dim != model_dim:
            raise ValueError(
                f"Pre MLP normalization dim {self.pre_mlp_norm.input_dim} does not match"
                f" the first normalization layer dim {model_dim}",
            )
        if self.mlp.model_dim != model_dim:
            raise ValueError(
                f"MLP up projection dim {self.mlp.up_projection.input_dim} does not match"
                f" the first normalization layer dim {model_dim}",
            )
        if self.mlp.hidden_dim != self.mlp.down_projection.input_dim:
            raise ValueError(
                f"MLP down projection dim {self.mlp.down_projection.input_dim} does not match"
                f" the up projection dim {self.mlp.hidden_dim}",
            )

    def __call__(
        self,
        x: Float[Array, "suffix_tokens channels"],
        positional_embeddings: PositionalEmbeddings,
        kv_cache: KVCacheLayerSlice | None = None,
        mask: Bool[Array, "suffix_tokens total_tokens"] | None = None,
        return_updated_kv_cache: bool = False,
    ) -> DecoderLayerOutput:
        residual = x
        x = vmap(self.pre_attention_norm, in_axes=0)(x)
        x, kv_cache = self.attention(x, positional_embeddings, kv_cache, mask, return_updated_kv_cache)
        if self.post_attention_norm is not None:
            x = vmap(self.post_attention_norm, in_axes=0)(x)
        x = residual + x

        residual = x
        x = vmap(self.pre_mlp_norm, in_axes=0)(x)
        x = vmap(self.mlp, in_axes=0)(x)
        if self.post_mlp_norm is not None:
            x = vmap(self.post_mlp_norm, in_axes=0)(x)
        x = residual + x

        return DecoderLayerOutput(output=x, kv_cache=kv_cache)

    def export_weights(self) -> ParameterDict:
        result = ParameterDict(
            pre_attention_norm=self.pre_attention_norm.export_weights(),
            attention=self.attention.export_weights(),
            pre_mlp_norm=self.pre_mlp_norm.export_weights(),
            mlp=self.mlp.export_weights(),
        )
        if self.post_attention_norm is not None:
            result["post_attention_norm"] = self.post_attention_norm.export_weights()
        if self.post_mlp_norm is not None:
            result["post_mlp_norm"] = self.post_mlp_norm.export_weights()
        return result
