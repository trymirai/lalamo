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
    NormalizationConfig,
    TiedEmbeddingConfig,
    TransformerConfig,
    TransformerLayerConfig,
    UnscaledRoPEConfig,
    UntiedEmbeddingConfig,
    UpcastMode,
)
from lalamo.modules.activations import SiLU
from lalamo.quantization import QuantizationMode

from .common import AWQQuantizationConfig, GPTQQuantizationConfig, HuggingFaceLMConfig

__all__ = ["HFQwen2Config"]


@dataclass(frozen=True)
class HFQwen2Config(HuggingFaceLMConfig):
    torch_dtype: Literal["bfloat16", "float16", "float32"]
    architectures: list[Literal["Qwen2ForCausalLM"]]
    attention_dropout: float
    bos_token_id: int | list[int]
    eos_token_id: int | list[int]
    hidden_act: Literal["silu"]
    hidden_size: int
    initializer_range: float
    intermediate_size: int
    max_position_embeddings: int
    max_window_layers: int
    model_type: Literal["qwen2"]
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    sliding_window: int
    tie_word_embeddings: bool
    transformers_version: str
    use_cache: bool
    use_sliding_window: bool
    vocab_size: int

    quantization_config: AWQQuantizationConfig | GPTQQuantizationConfig | None = None

    def _get_sliding_window_sizes(self) -> list[int | None]:
        if not self.use_sliding_window:
            return [None] * self.num_hidden_layers

        sliding_window_sizes = []
        for i in range(self.num_hidden_layers):
            if i < self.max_window_layers:
                sliding_window_sizes.append(self.sliding_window)
            else:
                sliding_window_sizes.append(None)
        return sliding_window_sizes

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
        if self.quantization_config is None:
            linear_config = FullPrecisionLinearConfig(
                precision=activation_precision,
            )
        else:
            linear_config = GroupQuantizedLinearConfig(
                group_size=self.quantization_config.group_size,
                weight_quantization_mode=QuantizationMode.from_num_bits(self.quantization_config.bits),
                activation_quantization_mode=None,
                activation_precision=activation_precision,
            )
        head_dim = self.hidden_size // self.num_attention_heads
        mlp_config = DenseMLPConfig(
            linear_config=linear_config,
            activation=SiLU(),
            has_up_biases=False,
            has_down_biases=False,
            up_clipping=None,
            gate_clipping=None,
        )

        sliding_window_sizes = self._get_sliding_window_sizes()
        layer_configs = []
        for sliding_window_size in sliding_window_sizes:
            attention_config = AttentionConfig(
                qkv_projection_config=linear_config,
                out_projection_config=linear_config,
                query_norm_config=None,
                key_norm_config=None,
                logit_soft_cap=None,
                has_sinks=False,
                has_qkv_biases=True,
                has_out_biases=False,
                num_heads=self.num_attention_heads,
                num_groups=self.num_key_value_heads,
                head_dim=head_dim,
                is_causal=True,
                scale=None,
                sliding_window_size=sliding_window_size,
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
