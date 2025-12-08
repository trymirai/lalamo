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
    SeparableCausalConvConfig,
    ShortConvConfig,
    SiLU,
    TiedEmbeddingConfig,
    TransformerConfig,
    TransformerLayerConfig,
    UnscaledRoPEConfig,
    UntiedEmbeddingConfig,
    UpcastMode,
)

from .common import HuggingFaceLMConfig


@dataclass(frozen=True)
class HFLFM2Config(HuggingFaceLMConfig):
    architectures: list[Literal["Lfm2ForCausalLM"]]
    block_auto_adjust_ff_dim: Literal[False]
    block_dim: int
    block_ff_dim: int
    block_ffn_dim_multiplier: float
    block_mlp_init_scale: float
    block_multiple_of: int
    block_norm_eps: float
    block_out_init_scale: float
    block_use_swiglu: bool
    block_use_xavier_init: bool
    bos_token_id: int
    conv_L_cache: int # noqa: N815
    conv_bias: int
    conv_dim: int
    conv_dim_out: int
    conv_use_xavier_init: bool
    eos_token_id: int
    hidden_size: int
    initializer_range: float
    intermediate_size: int
    layer_types: list[Literal["conv", "full_attention"]]
    max_position_embeddings: int
    model_type: Literal["lfm2"]
    norm_eps: float
    num_attention_heads: int
    num_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    pad_token_id: int
    rope_theta: float
    theta: float
    tie_embedding: bool
    torch_dtype: Literal["bfloat16"]
    transformers_version: str
    use_cache: bool
    use_pos_enc: bool
    vocab_size: int

    def to_decoder_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
        metadata_dict: Mapping[str, str],  # noqa: ARG002
    ) -> DecoderConfig:
        assert self.num_attention_heads == self.num_heads

        if self.tie_embedding:
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

        linear_config = FullPrecisionLinearConfig(activation_precision)

        block_norm_config = NormalizationConfig(
            scale_precision=activation_precision,
            accumulation_precision=accumulation_precision,
            epsilon=self.block_norm_eps,
            scale_offset=None,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=False,
        )

        attention_config = AttentionConfig(
            qkv_projection_config=linear_config,
            out_projection_config=linear_config,
            query_norm_config=block_norm_config,
            key_norm_config=block_norm_config,
            num_heads=self.num_attention_heads,
            num_groups=self.num_key_value_heads,
            head_dim=self.hidden_size // self.num_heads,
            is_causal=True,
            scale=None,
            sliding_window_size=None,
            logit_soft_cap=None,
            has_sinks=False,
            has_qkv_biases=False,
            has_out_biases=False,
        )

        short_conv_config = ShortConvConfig(
            in_projection_config=linear_config,
            conv_config=SeparableCausalConvConfig(activation_precision, has_biases=False),
            out_projection_config=linear_config,
            kernel_size=self.conv_L_cache,
        )

        mlp_config = DenseMLPConfig(
            linear_config=linear_config,
            activation=SiLU(),
            has_up_biases=False,
            has_down_biases=False,
            up_clipping=None,
            gate_clipping=None,
        )

        layer_configs = [
            TransformerLayerConfig(
                pre_mixer_norm_config=block_norm_config,
                mixer_config={"conv": short_conv_config, "full_attention": attention_config}[layer_type],
                post_mixer_norm_config=None,
                pre_mlp_norm_config=block_norm_config,
                mlp_config=mlp_config,
                post_mlp_norm_config=None,
            ) for layer_type in self.layer_types
        ]

        output_norm_config = NormalizationConfig(
            scale_precision=activation_precision,
            accumulation_precision=accumulation_precision,
            epsilon=self.norm_eps,
            scale_offset=None,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=False,
        )

        transformer_config = TransformerConfig(
            global_rope_config=rope_config,
            local_rope_config=None,
            layer_configs=tuple(layer_configs),
            output_norm_config=output_norm_config,
            model_dim=self.hidden_size,
            hidden_dim=self.intermediate_size,
            context_length=context_length or self.max_position_embeddings,
        )

        return DecoderConfig(
            embedding_config=embedding_config,
            transformer_config=transformer_config,
            vocab_size=self.vocab_size,
        )
