from dataclasses import dataclass
from typing import Literal

from fartsovka.common import DType
from fartsovka.modules import (
    Activation,
    AttentionConfig,
    DecoderConfig,
    DecoderLayerConfig,
    FullPrecisionLinearConfig,
    LlamaRoPEConfig,
    MLPConfig,
    RMSNormConfig,
    TiedEmbeddingConfig,
    UnscaledRoPEConfig,
)

from .common import ForeignConfig

__all__ = ["HFLlamaConfig", "HFQwen2Config", "HFGemma2Config"]


@dataclass
class HFRopeScalingConfig:
    factor: float
    high_freq_factor: float
    low_freq_factor: float
    original_max_position_embeddings: int
    rope_type: Literal["llama3"]


@dataclass
class HFLlamaConfig(ForeignConfig):
    architectures: list[Literal["LlamaForCausalLM"]]
    attention_bias: bool
    attention_dropout: float
    bos_token_id: int | list[int]
    eos_token_id: int | list[int]
    head_dim: int
    hidden_act: Literal["silu"]
    hidden_size: int
    initializer_range: float
    intermediate_size: int
    max_position_embeddings: int
    mlp_bias: bool
    model_type: Literal["llama"]
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    pretraining_tp: int
    rms_norm_eps: float
    rope_scaling: HFRopeScalingConfig
    rope_theta: float
    tie_word_embeddings: bool
    torch_dtype: Literal["bfloat16", "float16", "float32"]
    transformers_version: str
    use_cache: bool
    vocab_size: int

    def to_decoder_config(
        self,
        context_length: int,
        activation_precision: DType,
        accumulation_precision: DType,
    ) -> DecoderConfig:
        embedding_config = TiedEmbeddingConfig(
            input_scale=None,
            logits_soft_cap=None,
            precision=activation_precision,
        )
        rope_config = LlamaRoPEConfig(
            precision=activation_precision,
            base=self.rope_theta,
            max_sequence_length=self.max_position_embeddings,
            scaling_factor=self.rope_scaling.factor,
            original_context_length=self.rope_scaling.original_max_position_embeddings,
            low_frequency_factor=self.rope_scaling.low_freq_factor,
            high_frequency_factor=self.rope_scaling.high_freq_factor,
        )
        rmsnorm_config = RMSNormConfig(
            scale_precision=activation_precision,
            accumulation_precision=accumulation_precision,
            epsilon=self.rms_norm_eps,
        )
        linear_config = FullPrecisionLinearConfig(
            precision=activation_precision,
        )
        attention_config = AttentionConfig(
            qkv_projection_config=linear_config,
            out_projection_config=linear_config,
            logit_soft_cap=None,
            use_qkv_bias=self.attention_bias,
            use_out_bias=False,
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
            rope_config=rope_config,
            layer_config=decoder_layer_config,
            output_norm_config=rmsnorm_config,
            vocab_dim=self.vocab_size,
            model_dim=self.hidden_size,
            hidden_dim=self.intermediate_size,
            num_heads=self.num_attention_heads,
            num_groups=self.num_key_value_heads,
            head_dim=self.head_dim,
            attention_scale=None,
            num_layers=self.num_hidden_layers,
            sliding_window_sizes=None,
            context_length=context_length,
        )


@dataclass
class HFQwen2Config(ForeignConfig):
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
    torch_dtype: Literal["bfloat16", "float16", "float32"]
    transformers_version: str
    use_cache: bool
    use_sliding_window: bool
    vocab_size: int

    def _get_sliding_window_sizes(self) -> list[int | None]:
        sliding_window_sizes = []
        for i in range(self.num_hidden_layers):
            if i < self.max_window_layers:
                sliding_window_sizes.append(self.sliding_window)
            else:
                sliding_window_sizes.append(None)
        return sliding_window_sizes

    def to_decoder_config(
        self,
        context_length: int,
        activation_precision: DType,
        accumulation_precision: DType,
    ) -> DecoderConfig:
        embedding_config = TiedEmbeddingConfig(
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
        )
        linear_config = FullPrecisionLinearConfig(
            precision=activation_precision,
        )
        attention_config = AttentionConfig(
            qkv_projection_config=linear_config,
            out_projection_config=linear_config,
            logit_soft_cap=None,
            use_qkv_bias=True,
            use_out_bias=False,
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
            rope_config=rope_config,
            layer_config=decoder_layer_config,
            output_norm_config=rmsnorm_config,
            vocab_dim=self.vocab_size,
            model_dim=self.hidden_size,
            hidden_dim=self.intermediate_size,
            num_heads=self.num_attention_heads,
            num_groups=self.num_key_value_heads,
            head_dim=self.hidden_size // self.num_attention_heads,
            attention_scale=None,
            num_layers=self.num_hidden_layers,
            sliding_window_sizes=tuple(self._get_sliding_window_sizes()),
            context_length=context_length,
        )


@dataclass
class HFGemma2Config(ForeignConfig):
    architectures: list[Literal["Gemma2ForCausalLM"]]
    attention_bias: bool
    attention_dropout: float
    attn_logit_softcapping: float
    bos_token_id: int | list[int]
    cache_implementation: Literal["hybrid"]
    eos_token_id: int | list[int]
    final_logit_softcapping: float
    head_dim: int
    hidden_act: Literal["gelu_pytorch_tanh"]
    hidden_activation: Literal["gelu_pytorch_tanh"]
    hidden_size: int
    initializer_range: float
    intermediate_size: int
    max_position_embeddings: int
    model_type: Literal["gemma2"]
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    pad_token_id: int
    query_pre_attn_scalar: float
    rms_norm_eps: float
    rope_theta: float
    sliding_window: int
    torch_dtype: Literal["bfloat16", "float16", "float32"]
    transformers_version: str
    use_cache: bool
    vocab_size: int

    def to_decoder_config(
        self,
        context_length: int,
        activation_precision: DType,
        accumulation_precision: DType,
    ) -> DecoderConfig:
        sliding_window_sizes = tuple(
            self.sliding_window if not bool(i % 2) else None for i in range(self.num_hidden_layers)
        )
        embedding_input_scale = self.hidden_size**0.5
        attention_scale = self.query_pre_attn_scalar**-0.5
        embedding_config = TiedEmbeddingConfig(
            input_scale=embedding_input_scale,
            logits_soft_cap=self.final_logit_softcapping,
            precision=activation_precision,
        )
        rope_config = UnscaledRoPEConfig(
            precision=activation_precision,
            base=self.rope_theta,
            max_sequence_length=self.max_position_embeddings,
        )
        rmsnorm_config = RMSNormConfig(
            scale_precision=accumulation_precision,
            accumulation_precision=accumulation_precision,
            epsilon=self.rms_norm_eps,
        )
        linear_config = FullPrecisionLinearConfig(
            precision=activation_precision,
        )
        attention_config = AttentionConfig(
            qkv_projection_config=linear_config,
            out_projection_config=linear_config,
            logit_soft_cap=self.attn_logit_softcapping,
            use_qkv_bias=self.attention_bias,
            use_out_bias=False,
        )
        mlp_config = MLPConfig(
            linear_config=linear_config,
            activation=Activation.GELU,
        )
        decoder_layer_config = DecoderLayerConfig(
            pre_attention_norm_config=rmsnorm_config,
            attention_config=attention_config,
            post_attention_norm_config=rmsnorm_config,
            pre_mlp_norm_config=rmsnorm_config,
            mlp_config=mlp_config,
            post_mlp_norm_config=rmsnorm_config,
        )
        return DecoderConfig(
            embedding_config=embedding_config,
            rope_config=rope_config,
            layer_config=decoder_layer_config,
            output_norm_config=rmsnorm_config,
            vocab_dim=self.vocab_size,
            model_dim=self.hidden_size,
            hidden_dim=self.intermediate_size,
            num_heads=self.num_attention_heads,
            num_groups=self.num_key_value_heads,
            head_dim=self.head_dim,
            attention_scale=attention_scale,
            num_layers=self.num_hidden_layers,
            sliding_window_sizes=sliding_window_sizes,
            context_length=context_length,
        )
