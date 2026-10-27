from collections.abc import Sequence
from dataclasses import dataclass

import equinox as eqx
from jaxtyping import Array, Float

from lalamo.exportable import Exportable
from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoConfig, LalamoModule
from lalamo.modules.normalization import Normalization, NormalizationConfig
from lalamo.modules.rope import PositionalEmbeddings
from lalamo.modules.utils import call_vmapped

from .forward_pass_config import VisionForwardPassConfig
from .image_grid import ImageGrid
from .vision_attention import VisionAttention, VisionAttentionConfig
from .vision_mlp import VisionMLP, VisionMLPConfig

__all__ = [
    "VisionLayer",
    "VisionLayerActivationTrace",
    "VisionLayerConfig",
    "VisionLayerResult",
]


class VisionLayerActivationTrace(Exportable, eqx.Module):
    inputs: Float[Array, "tokens channels"]
    pre_attention_norm: Float[Array, "tokens channels"]
    attention: Float[Array, "tokens channels"]
    mlp_inputs: Float[Array, "tokens channels"]
    pre_mlp_norm: Float[Array, "tokens channels"]
    mlp: Float[Array, "tokens channels"]


class VisionLayerResult(Exportable, eqx.Module):
    outputs: Float[Array, "tokens channels"]
    activation_trace: VisionLayerActivationTrace | None


@dataclass(frozen=True)
class VisionLayerConfig(LalamoConfig):
    """Pre-norm transformer block: ``x + attn(norm(x))``, then ``x + mlp(norm(x))``."""

    pre_attention_norm_config: NormalizationConfig
    attention_config: VisionAttentionConfig
    pre_mlp_norm_config: NormalizationConfig
    mlp_config: VisionMLPConfig

    def init(self, initializer: Initializer, model_dim: int, hidden_dim: int) -> "VisionLayer":
        return VisionLayer(
            config=self,
            sharding_config=initializer.sharding_config,
            pre_attention_norm=self.pre_attention_norm_config.init(initializer, model_dim),
            attention=self.attention_config.init(initializer, model_dim),
            pre_mlp_norm=self.pre_mlp_norm_config.init(initializer, model_dim),
            mlp=self.mlp_config.init(initializer, model_dim, hidden_dim, model_dim),
        )


class VisionLayer(LalamoModule[VisionLayerConfig]):
    pre_attention_norm: Normalization
    attention: VisionAttention
    pre_mlp_norm: Normalization
    mlp: VisionMLP

    @eqx.filter_jit
    def __call__(
        self,
        inputs: Float[Array, "tokens channels"],
        positional_embeddings: PositionalEmbeddings,
        grids: Sequence[ImageGrid],
        forward_pass_config: VisionForwardPassConfig = VisionForwardPassConfig(),
        *,
        return_activation_trace: bool = False,
        keychain: Keychain,
    ) -> VisionLayerResult:
        attention_keychain, mlp_keychain = keychain.split()

        pre_attention_norm = call_vmapped(self.pre_attention_norm, inputs)
        attention = self.attention(
            pre_attention_norm,
            positional_embeddings,
            grids,
            forward_pass_config,
            keychain=attention_keychain,
        )
        mlp_inputs = inputs + attention

        pre_mlp_norm = call_vmapped(self.pre_mlp_norm, mlp_inputs)
        mlp = self.mlp(pre_mlp_norm, forward_pass_config.matmul_config, keychain=mlp_keychain)
        outputs = mlp_inputs + mlp

        if return_activation_trace:
            activation_trace = VisionLayerActivationTrace(
                inputs=inputs,
                pre_attention_norm=pre_attention_norm,
                attention=attention,
                mlp_inputs=mlp_inputs,
                pre_mlp_norm=pre_mlp_norm,
                mlp=mlp,
            )
        else:
            activation_trace = None
        return VisionLayerResult(outputs=outputs, activation_trace=activation_trace)
