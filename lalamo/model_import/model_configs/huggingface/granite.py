from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from jaxtyping import Array

from lalamo.model import Model
from lalamo.model_import.loaders.binding import Binder
from lalamo.modules.activations import SiLU
from lalamo.modules.decoder import DecoderConfig
from lalamo.modules.embedding import TiedEmbeddingConfig
from lalamo.modules.linear import LinearConfig
from lalamo.modules.mlp import DenseMLPConfig
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.rope import UnscaledRoPEConfig
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.utils.lazy_collections import MappedValues
from lalamo.weight_matrix import CompressionImplementation

from .common import HuggingFaceLMConfig, residual_decoder

__all__ = ["HFGraniteConfig"]


@dataclass(frozen=True)
class HFGraniteConfig(HuggingFaceLMConfig):
    torch_dtype: Literal["bfloat16", "float16", "float32"]
    attention_bias: bool
    eos_token_id: int | list[int]
    hidden_act: Literal["silu"]
    hidden_size: int
    intermediate_size: int
    max_position_embeddings: int
    mlp_bias: bool
    model_type: Literal["granite"]
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_scaling: dict | None
    rope_theta: float
    tie_word_embeddings: bool
    vocab_size: int

    embedding_multiplier: float
    residual_multiplier: float
    attention_multiplier: float
    logits_scaling: float

    head_dim: int | None = None

    def build_decoder(self, at: Binder) -> DecoderConfig:
        assert self.rope_scaling is None, "Granite with rope scaling is not supported"
        head_dim = self.head_dim if self.head_dim is not None else self.hidden_size // self.num_attention_heads

        assert self.tie_word_embeddings, "Granite always ties word embeddings"
        embedding_config = TiedEmbeddingConfig(
            input_scale=self.embedding_multiplier,
            logit_soft_cap=None,
        )

        rope_config = UnscaledRoPEConfig(
            base=self.rope_theta,
            max_sequence_length=self.max_position_embeddings,
            head_dim=head_dim,
        )

        rmsnorm_config = NormalizationConfig(
            epsilon=self.rms_norm_eps,
            scale_offset=None,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=False,
        )
        linear_config = LinearConfig()
        attention_config = AttentionConfig(
            qkvg_projection_config=linear_config,
            out_projection_config=linear_config,
            query_norm_config=None,
            key_norm_config=None,
            logit_soft_cap=None,
            has_sinks=False,
            has_qkvg_biases=self.attention_bias,
            has_out_biases=False,
            num_heads=self.num_attention_heads,
            num_groups=self.num_key_value_heads,
            head_dim=head_dim,
            is_causal=True,
            scale=self.attention_multiplier,
            sliding_window_size=None,
            rope_config=rope_config,
        )
        mlp_config = DenseMLPConfig(
            linear_config=linear_config,
            activation=SiLU(),
            has_up_biases=self.mlp_bias,
            has_down_biases=self.mlp_bias,
            up_clipping=None,
            gate_clipping=None,
            hidden_dim=self.intermediate_size,
        )
        return residual_decoder(
            at,
            embedding=at("model.embed_tokens", embedding_config),
            layers=[
                (at(f"model.layers.{index}.self_attn", attention_config), at(f"model.layers.{index}.mlp", mlp_config))
                for index in range(self.num_hidden_layers)
            ],
            norm=rmsnorm_config,
            model_dim=self.hidden_size,
            vocab_size=self.vocab_size,
        )

    def _scaled_weights(self, weights_dict: Mapping[str, Array]) -> MappedValues[str, Array, Array]:
        final_norm_suffixes = ("model.norm.weight", "language_model.norm.weight")
        residual_scaled_suffixes = tuple(
            f"layers.{i}.{projection}.{parameter}"
            for i in range(self.num_hidden_layers)
            for projection in ("self_attn.o_proj", "mlp.down_proj")
            for parameter in ("weight", "bias")
        )

        def scale_weight(key: str, weight: Array) -> Array:
            if key.endswith(final_norm_suffixes):
                return weight / self.logits_scaling

            if key.endswith(residual_scaled_suffixes):
                return weight * self.residual_multiplier

            return weight

        return MappedValues(weights_dict, scale_weight)

    def _load_weights(
        self,
        model: Model,
        weights_dict: Mapping[str, Array],
        *,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
    ) -> Model:
        return super()._load_weights(
            model,
            self._scaled_weights(weights_dict),
            implementation=implementation,
        )
