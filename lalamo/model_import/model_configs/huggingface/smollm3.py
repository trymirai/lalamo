from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from jaxtyping import DTypeLike

from lalamo.modules import (
    AttentionConfig,
    DecoderConfig,
    DenseMLPConfig,
    FullPrecisionLinearConfig,
    GroupQuantizedLinearConfig,
    MLXQuantizedLinearConfig,
    MLXQuantizedTiedEmbeddingConfig,
    MLXQuantizedUntiedEmbeddingConfig,
    NormalizationConfig,
    SiLU,
    TiedEmbeddingConfig,
    TransformerConfig,
    TransformerLayerConfig,
    UnscaledRoPEConfig,
    UntiedEmbeddingConfig,
    UpcastMode,
)
from lalamo.quantization import QuantizationMode

from .common import HuggingFaceLMConfig, MLXQuantizationConfig, QuantizationConfigType

__all__ = ["HFSmolLM3Config"]


@dataclass(frozen=True)
class HFSmolLM3Config(HuggingFaceLMConfig):
    torch_dtype: Literal["bfloat16", "float16", "float32"]
    architectures: list[Literal["SmolLM3ForCausalLM"]]
    attention_bias: bool
    attention_dropout: float
    bos_token_id: int | list[int]
    eos_token_id: int | list[int]
    hidden_act: Literal["silu_glu", "silu"]
    hidden_size: int
    initializer_range: float
    intermediate_size: int
    max_position_embeddings: int
    mlp_bias: bool
    model_type: Literal["smollm3"]
    no_rope_layers: list[int]
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    pretraining_tp: int
    rms_norm_eps: float
    rope_theta: float
    transformers_version: str
    use_cache: bool
    vocab_size: int
    tie_word_embeddings: bool = True

    quantization: QuantizationConfigType = None
    quantization_config: QuantizationConfigType = None

    def to_decoder_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
        metadata_dict: Mapping[str, str],  # noqa: ARG002
    ) -> DecoderConfig:
        quantization = self.quantization or self.quantization_config
        if isinstance(quantization, MLXQuantizationConfig):
            if self.tie_word_embeddings:
                embedding_config = MLXQuantizedTiedEmbeddingConfig(
                    input_scale=None,
                    logit_soft_cap=None,
                    group_size=quantization.group_size,
                    embedding_quantization_mode=QuantizationMode.from_num_bits(quantization.bits),
                    activation_quantization_mode=None,
                    activation_precision=activation_precision,
                )
            else:
                embedding_config = MLXQuantizedUntiedEmbeddingConfig(
                    input_scale=None,
                    logit_soft_cap=None,
                    group_size=quantization.group_size,
                    embedding_quantization_mode=QuantizationMode.from_num_bits(quantization.bits),
                    activation_quantization_mode=None,
                    activation_precision=activation_precision,
                )
        else:  # noqa: PLR5501
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

        if quantization is None:
            linear_config = FullPrecisionLinearConfig(
                precision=activation_precision,
            )
        elif isinstance(quantization, MLXQuantizationConfig):
            linear_config = MLXQuantizedLinearConfig(
                group_size=quantization.group_size,
                weight_quantization_mode=QuantizationMode.from_num_bits(quantization.bits),
                activation_quantization_mode=None,
                activation_precision=activation_precision,
            )
        else:
            linear_config = GroupQuantizedLinearConfig(
                group_size=quantization.group_size,
                weight_quantization_mode=QuantizationMode.from_num_bits(quantization.bits),
                activation_quantization_mode=None,
                activation_precision=activation_precision,
            )

        layer_head_dim = self.hidden_size // self.num_attention_heads
        if len(self.no_rope_layers) < self.num_hidden_layers:
            raise ValueError(
                "SmolLM3 requires no_rope_layers to be a per-layer mask with at least num_hidden_layers entries, "
                f"got {len(self.no_rope_layers)} entries for {self.num_hidden_layers} layers.",
            )

        layer_configs = []
        for layer_idx in range(self.num_hidden_layers):
            use_rope = bool(self.no_rope_layers[layer_idx])

            attention_config = AttentionConfig(
                qkv_projection_config=linear_config,
                out_projection_config=linear_config,
                query_norm_config=None,
                key_norm_config=None,
                logit_soft_cap=None,
                has_sinks=False,
                has_qkv_biases=self.attention_bias,
                has_out_biases=self.attention_bias,
                num_heads=self.num_attention_heads,
                num_groups=self.num_key_value_heads,
                head_dim=layer_head_dim,
                is_causal=True,
                scale=None,
                sliding_window_size=None,
                use_rope=use_rope,
            )
            mlp_config = DenseMLPConfig(
                linear_config=linear_config,
                activation=SiLU(),
                has_up_biases=self.mlp_bias,
                has_down_biases=self.mlp_bias,
                up_clipping=None,
                gate_clipping=None,
            )
            layer_configs.append(
                TransformerLayerConfig(
                    pre_mixer_norm_config=rmsnorm_config,
                    mixer_config=attention_config,
                    post_mixer_norm_config=None,
                    pre_mlp_norm_config=rmsnorm_config,
                    mlp_config=mlp_config,
                    post_mlp_norm_config=None,
                ),
            )

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
