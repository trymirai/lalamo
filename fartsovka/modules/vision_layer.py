from dataclasses import dataclass

import jax
from jax import vmap
from jaxtyping import Array, Float, Int, PRNGKeyArray

from fartsovka.common import ParameterDict

from .common import FartsovkaModule
from .mlp import MLP, MLPConfig
from .normalization import RMSNorm, RMSNormConfig
from .rope import PositionalEmbeddings
from .vision_attention import VisionAttention, VisionAttentionConfig

__all__ = [
    "VisionLayer",
    "VisionLayerConfig",
]

@dataclass
class VisionLayerConfig:
    norm_config: RMSNormConfig
    attention_config: VisionAttentionConfig
    mlp_config: MLPConfig

    def random_init(
        self,
        model_dim: int,
        hidden_dim: int,
        num_heads: int,
        *,
        key: PRNGKeyArray,
    ) -> "VisionLayer":
        pre_attention_norm_key, attn_key, pre_mlp_norm_key, mlp_key = jax.random.split(key, 4)

        pre_attention_norm = self.norm_config.init(model_dim)
        pre_mlp_norm = self.norm_config.init(model_dim)

        attention = self.attention_config.random_init(
            model_dim=model_dim,
            num_heads=num_heads,
            key=attn_key,
        )

        mlp = self.mlp_config.random_init(
            model_dim=model_dim,
            hidden_dim=hidden_dim,
            key=mlp_key,
        )

        return VisionLayer(
            config=self,
            pre_attention_norm=pre_attention_norm,
            attention=attention,
            pre_mlp_norm=pre_mlp_norm,
            mlp=mlp,
        )


class VisionLayer(FartsovkaModule[VisionLayerConfig]):
    pre_attention_norm: RMSNorm
    attention: VisionAttention
    pre_mlp_norm: RMSNorm
    mlp: MLP


    def __call__(
        self,
        hidden_states: Float[Array, "seq_len hidden_size"],
        position_embeddings: PositionalEmbeddings | None = None,
        cumulative_seqlens: Int[Array, "n_plus_1"] | None = None,
    ) -> Float[Array, "seq_len hidden_size"]:
        residual = hidden_states
        normed_hidden_states = vmap(self.pre_attention_norm, in_axes=0)(hidden_states)
        attention_output = self.attention(
            hidden_states=normed_hidden_states,
            cumulative_seqlens=cumulative_seqlens,
            position_embeddings=position_embeddings,
        )
        hidden_states_after_attn = residual + attention_output

        residual_mlp = hidden_states_after_attn
        normed_hidden_states_mlp = vmap(self.pre_mlp_norm, in_axes=0)(hidden_states_after_attn)
        mlp_output = vmap(self.mlp, in_axes=0)(normed_hidden_states_mlp)

        hidden_states = residual_mlp + mlp_output
        return hidden_states

    def export_weights(self) -> ParameterDict:
        return ParameterDict(
            pre_attention_norm=self.pre_attention_norm.export_weights(),
            attention=self.attention.export_weights(),
            pre_mlp_norm=self.pre_mlp_norm.export_weights(),
            mlp=self.mlp.export_weights(),
        )

