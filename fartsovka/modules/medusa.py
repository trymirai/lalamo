import jax
import equinox as eqx
import jax.numpy as jnp
from typing import List, Optional
from dataclasses import dataclass, field
from jaxtyping import Array, Float, PRNGKeyArray

from fartsovka.common import ParameterDict

from .activations import Activation
from .common import FartsovkaModule
from .resblock import ResBlock, ResBlockConfig
from .linear import FullPrecisionLinearConfig

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
    
    def random_init(
        self, 
        hidden_size: int, 
        *, 
        key: PRNGKeyArray,
        num_heads: Optional[int] = None
    ) -> "Medusa":
        
        actual_num_heads = num_heads if num_heads is not None else self.num_heads
        
        keys = jax.random.split(key, 2 * actual_num_heads)
        
        medusa_heads = []
        
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
                        
            medusa_heads.append(head_blocks)
            
        return Medusa(
            config=self,
            num_heads=actual_num_heads,
            medusa_heads=medusa_heads,
        )


class Medusa(FartsovkaModule):

    num_heads: int = eqx.field(static=True)
    
    medusa_heads: List[List[ResBlock]]
    
    def __init__(
        self,
        config: MedusaConfig,
        num_heads: int,
        medusa_heads: List[List[ResBlock]],
    ):
        
        super().__init__(config)
        self.num_heads = num_heads
        self.medusa_heads = medusa_heads
        
        if len(medusa_heads) != num_heads:
            raise ValueError(
                f"Number of heads ({num_heads}) doesn't match."
            )
    
    @property
    def num_layers(self) -> int:
        return self.config.num_layers
    
    def __call__(
        self, 
        hidden_states: Float[Array, "hidde hidden"]
    ) -> Float[Array, "num_heads hidden hidden"]:
        
        medusa_states = []
        
        for i in range(self.num_heads):

            head_hidden = hidden_states
            for resblock in self.medusa_heads[i]:
                head_hidden = resblock(head_hidden)

            medusa_states.append(head_hidden)
        
        return jnp.stack(medusa_states, axis=0)
    
    def export_weights(self) -> ParameterDict:

        heads_weights = []
        
        for i in range(self.num_heads):
            
            head_blocks_weights = []
            for j in range(len(self.medusa_heads[i])):
                head_blocks_weights.append(
                    self.medusa_heads[i][j].export_weights()
                )
            
            heads_weights.append(head_blocks_weights)
        
        return ParameterDict(
            medusa_heads=heads_weights
        )


def create_medusa_model(
    hidden_size: int,
    num_heads: int = 3, 
    num_layers: int = 1,
    *,
    key: PRNGKeyArray,
    resblock_config: Optional[ResBlockConfig] = None
) -> Medusa:

    config = MedusaConfig(num_heads=num_heads, num_layers=num_layers)
    
    if resblock_config is not None:
        config.resblock_config = resblock_config
        
    return config.random_init(
        hidden_size=hidden_size,
        key=key,
    ) 