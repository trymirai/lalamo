
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, Float, Int, PRNGKeyArray

from fartsovka.common import DType, ParameterDict

from .common import FartsovkaModule
from .mlp import MLP, MLPConfig
from .vision_attention import VisionAttention, VisionAttentionConfig
from .normalization import RMSNorm, RMSNormConfig


__all__ = [
    "VisionLayerConfig",
    "VisionLayer",
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
        norm1_key, attn_key, norm2_key, mlp_key = jax.random.split(key, 4)
        
        norm1 = self.norm_config.init(model_dim)
        norm2 = self.norm_config.init(model_dim)
        
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
            norm1=norm1,
            attention=attention,
            norm2=norm2,
            mlp=mlp,
        )


class VisionLayer(FartsovkaModule[VisionLayerConfig]):
    norm1: RMSNorm
    attention: VisionAttention
    norm2: RMSNorm
    mlp: MLP
    

    def __call__(
        self,
        hidden_states: Float[Array, "seq_len hidden_size"],
        position_embeddings_tuple: tuple[Float[Array, "seq_len head_dim"], Float[Array, "seq_len head_dim"]],
        cu_seqlens: Int[Array, "n_plus_1"] | None = None,
    ) -> Float[Array, "seq_len hidden_size"]:
        residual = hidden_states
        normed_hidden_states = vmap(self.norm1, in_axes=0)(hidden_states)
        attention_output = self.attention(
            hidden_states=normed_hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings_tuple,
        )
        hidden_states_after_attn = residual + attention_output

        residual_mlp = hidden_states_after_attn
        normed_hidden_states_mlp = vmap(self.norm2, in_axes=0)(hidden_states_after_attn)
        mlp_output = vmap(self.mlp, in_axes=0)(normed_hidden_states_mlp)

        hidden_states = residual_mlp + mlp_output
        return hidden_states
    
    def export_weights(self) -> ParameterDict:
        return ParameterDict(
            norm1=self.norm1.export_weights(),
            attention=self.attention.export_weights(),
            norm2=self.norm2.export_weights(),
            mlp=self.mlp.export_weights(),
        )

