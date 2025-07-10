from dataclasses import dataclass
from typing import Literal

from jaxtyping import DTypeLike

from lalamo.modules import (
    Activation,
    AttentionConfig,
    DecoderConfig,
    DecoderLayerConfig,
    FullPrecisionLinearConfig,
    GroupQuantizedLinearConfig,
    MLPConfig,
    RMSNormConfig,
    TiedEmbeddingConfig,
    UnscaledRoPEConfig,
    UntiedEmbeddingConfig,
    UpcastMode,
)
from lalamo.quantization import QuantizationMode

from .common import AWQQuantizationConfig, GPTQQuantizationConfig, HuggingFaceConfig

__all__ = ["HFQwen2Config"]


@dataclass(frozen=True)
class HFQwen2Config(HuggingFaceConfig):
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
    ) -> DecoderConfig:
        if self.tie_word_embeddings:
            embedding_config = TiedEmbeddingConfig(
                input_scale=None,
                logits_soft_cap=None,
                precision=activation_precision,
            )
        else:
            embedding_config = UntiedEmbeddingConfig(
                input_scale=None,
                logits_soft_cap=None,
                precision=activation_precision,
            )
        rope_config = UnscaledRoPEConfig(
            precision=activation_precision,
            base=self.rope_theta,
            max_sequence_length=self.max_position_embeddings,
        )
        rmsnorm_config = RMSNormConfig(
            scale_precision=activation_precision,
            accumulation_precision=accumulation_precision,
            epsilon=self.rms_norm_eps,
            scale_offset=None,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
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
        attention_config = AttentionConfig(
            qkv_projection_config=linear_config,
            out_projection_config=linear_config,
            query_norm_config=None,
            key_norm_config=None,
            logit_soft_cap=None,
            has_qkv_biases=True,
            has_out_biases=False,
        )
        mlp_config = MLPConfig(
            linear_config=linear_config,
            activation=Activation.SILU,
        )
        decoder_layer_config = DecoderLayerConfig(
            pre_attention_norm_config=rmsnorm_config,
            attention_config=attention_config,
            post_attention_norm_config=None,
            pre_mlp_norm_config=rmsnorm_config,
            mlp_config=mlp_config,
            post_mlp_norm_config=None,
        )
        return DecoderConfig(
            embedding_config=embedding_config,
            global_rope_config=rope_config,
            local_rope_config=None,
            layer_config=decoder_layer_config,
            output_norm_config=rmsnorm_config,
            vocab_size=self.vocab_size,
            model_dim=self.hidden_size,
            hidden_dim=self.intermediate_size,
            num_heads=self.num_attention_heads,
            num_groups=self.num_key_value_heads,
            head_dim=self.hidden_size // self.num_attention_heads,
            attention_scale=None,
            num_layers=self.num_hidden_layers,
            sliding_window_sizes=tuple(self._get_sliding_window_sizes()),
            context_length=context_length or self.max_position_embeddings,
        )
