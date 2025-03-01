from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union

from jaxtyping import Array, Float, PRNGKeyArray
import jax
import jax.numpy as jnp
import equinox as eqx

from fartsovka.common import ParameterDict, DType

from .activations import Activation
from .common import FartsovkaModule
from .resblock import ResBlock, ResBlockConfig
from .linear import LinearBase, LinearConfig, FullPrecisionLinearConfig

__all__ = ["Medusa", "MedusaConfig", "create_medusa_model"]


@dataclass
class MedusaConfig:

    num_heads: int = 3
    num_layers: int = 1
    
    resblock_config: ResBlockConfig = field(
        default_factory=lambda: ResBlockConfig(
            linear_config=FullPrecisionLinearConfig(precision=jnp.float32),
            activation=Activation.SILU
        )
    )
    linear_config: LinearConfig = field(
        default_factory=lambda: FullPrecisionLinearConfig(precision=jnp.float32)
    )
    
    def random_init(
        self, 
        hidden_size: int, 
        vocab_size: int, 
        *, 
        key: PRNGKeyArray,
        num_heads: Optional[int] = None
    ) -> "Medusa":
        
        actual_num_heads = num_heads if num_heads is not None else self.num_heads
        
        keys = jax.random.split(key, 2 * actual_num_heads)
        
        medusa_heads = []
        medusa_projections = []
        
        for i in range(actual_num_heads):

            head_blocks = []
            blocks_key = keys[i * 2]
            
            if self.num_layers > 1:
                block_keys = jax.random.split(blocks_key, self.num_layers)
            else:
                block_keys = [blocks_key] 
                
            for j in range(self.num_layers):
                block_key = block_keys[j]
                head_blocks.append(
                    self.resblock_config.random_init(
                        hidden_size=hidden_size,
                        key=block_key,
                    )
                )
            
            proj_key = keys[i * 2 + 1]
            head_projection = self.linear_config.random_init(
                hidden_size, 
                (vocab_size,),
                has_biases=True,
                key=proj_key,
            )
            
            medusa_heads.append(head_blocks)
            medusa_projections.append(head_projection)
            
        return Medusa(
            config=self,
            num_heads=actual_num_heads,
            medusa_heads=medusa_heads,
            medusa_projections=medusa_projections,
        )


class Medusa(FartsovkaModule):

    num_heads: int = eqx.field(static=True)
    
    medusa_heads: List[List[ResBlock]]
    medusa_projections: List[LinearBase]
    
    def __init__(
        self,
        config: MedusaConfig,
        num_heads: int,
        medusa_heads: List[List[ResBlock]],
        medusa_projections: List[LinearBase],
    ):
        
        super().__init__(config)
        self.num_heads = num_heads
        self.medusa_heads = medusa_heads
        self.medusa_projections = medusa_projections
        
        if len(medusa_heads) != num_heads or len(medusa_projections) != num_heads:
            raise ValueError(
                f"Number of heads ({num_heads}) doesn't match number of provided "
                f"heads ({len(medusa_heads)}) or projections ({len(medusa_projections)})"
            )
    
    @property
    def num_layers(self) -> int:
        return self.config.num_layers
    
    def __call__(
        self, 
        hidden_states: Float[Array, "batch seq hidden"]
    ) -> Float[Array, "num_heads batch seq vocab"]:
        
        medusa_logits = []
        
        for i in range(self.num_heads):

            head_hidden = hidden_states
            for resblock in self.medusa_heads[i]:
                head_hidden = resblock(head_hidden)

            (head_logits,) = self.medusa_projections[i](head_hidden)
            medusa_logits.append(head_logits)
        
        return jnp.stack(medusa_logits, axis=0)
    
    def export_weights(self) -> ParameterDict:

        heads_weights = []
        projections_weights = []
        
        for i in range(self.num_heads):
            
            head_blocks_weights = []
            for j in range(len(self.medusa_heads[i])):
                head_blocks_weights.append(
                    self.medusa_heads[i][j].export_weights()
                )
            
            heads_weights.append(head_blocks_weights)
            projections_weights.append(self.medusa_projections[i].export_weights())
        
        return ParameterDict(
            medusa_heads=heads_weights,
            medusa_projections=projections_weights,
        )


def create_medusa_model(
    hidden_size: int,
    vocab_size: int,
    num_heads: int = 3, 
    num_layers: int = 1,
    *,
    key: PRNGKeyArray,
    resblock_config: Optional[ResBlockConfig] = None,
    linear_config: Optional[LinearConfig] = None,
) -> Medusa:

    config = MedusaConfig(num_heads=num_heads, num_layers=num_layers)
    
    if resblock_config is not None:
        config.resblock_config = resblock_config
    
    if linear_config is not None:
        config.linear_config = linear_config
        
    return config.random_init(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        key=key,
    ) 