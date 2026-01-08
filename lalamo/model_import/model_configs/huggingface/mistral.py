from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from jaxtyping import DTypeLike

from lalamo.modules import (
    AttentionConfig,
    DecoderConfig,
    DenseMLPConfig,
    FullPrecisionLinearConfig,
    NormalizationConfig,
    TiedEmbeddingConfig,
    TransformerConfig,
    TransformerLayerConfig,
    UnscaledRoPEConfig,
    UntiedEmbeddingConfig,
)
from lalamo.modules.activations import SiLU
from lalamo.modules.normalization import UpcastMode

from .common import HuggingFaceLMConfig

__all__ = ["HFMistralConfig"]


@dataclass(frozen=True)
class HFMistralConfig(HuggingFaceLMConfig):
    architectures: list[Literal["MistralForCausalLM"]]
    attention_dropout: float
    bos_token_id: int
    eos_token_id: int
    hidden_act: Literal["silu"]
    hidden_size: int
    initializer_range: float
    intermediate_size: int
    max_position_embeddings: int
    model_type: Literal["mistral"]
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    sliding_window: int | None
    tie_word_embeddings: bool
    transformers_version: str
    use_cache: bool
    vocab_size: int
    head_dim: int | None = None

    def to_decoder_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
        metadata_dict: Mapping[str, str],  # noqa: ARG002
    ) -> DecoderConfig:
        if self.tie_word_embeddings:
            embedding_config = TiedEmbeddingConfig(
                input_scale=None,
                logit_soft_cap=None,
                precision=activation_precision,
            )
        else:
            embedding_config = UntiedEmbeddingConfig(
                input_scale=None,
                logit_soft_cap=None,
                precision=activation_precision,
            )

        rope_config = UnscaledRoPEConfig(
            precision=activation_precision,
            base=self.rope_theta,
            max_sequence_length=context_length or self.max_position_embeddings,
        )

        rmsnorm_config = NormalizationConfig(
            scale_precision=activation_precision,
            accumulation_precision=accumulation_precision,
            epsilon=self.rms_norm_eps,
            scale_offset=None,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=False,
        )

        linear_config = FullPrecisionLinearConfig(
            precision=activation_precision,
        )

        head_dim = self.head_dim or self.hidden_size // self.num_attention_heads

        mlp_config = DenseMLPConfig(
            linear_config=linear_config,
            activation=SiLU(),
            has_up_biases=False,
            has_down_biases=False,
            up_clipping=None,
            gate_clipping=None,
        )

        layer_configs = []
        for _ in range(self.num_hidden_layers):
            attention_config = AttentionConfig(
                qkv_projection_config=linear_config,
                out_projection_config=linear_config,
                query_norm_config=None,
                key_norm_config=None,
                logit_soft_cap=None,
                has_sinks=False,
                has_qkv_biases=False,
                has_out_biases=False,
                num_heads=self.num_attention_heads,
                num_groups=self.num_key_value_heads,
                head_dim=head_dim,
                is_causal=True,
                scale=None,
                sliding_window_size=self.sliding_window,
            )

            transformer_layer_config = TransformerLayerConfig(
                pre_mixer_norm_config=rmsnorm_config,
                mixer_config=attention_config,
                post_mixer_norm_config=None,
                pre_mlp_norm_config=rmsnorm_config,
                mlp_config=mlp_config,
                post_mlp_norm_config=None,
            )
            layer_configs.append(transformer_layer_config)

        transformer_config = TransformerConfig(
            global_rope_config=rope_config,
            local_rope_config=None,
            layer_configs=tuple(layer_configs),
            output_norm_config=rmsnorm_config,
            model_dim=self.hidden_size,
            hidden_dim=self.intermediate_size,
            context_length=context_length or self.max_position_embeddings,
        )

        return DecoderConfig(
            embedding_config=embedding_config,
            transformer_config=transformer_config,
            vocab_size=self.vocab_size,
        )
