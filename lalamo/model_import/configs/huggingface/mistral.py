from dataclasses import dataclass
from typing import Literal

from jaxtyping import DTypeLike

from lalamo.modules import (
    Activation,
    AttentionConfig,
    DecoderConfig,
    DecoderLayerConfig,
    FullPrecisionLinearConfig,
    MLPConfig,
    RMSNormConfig,
    TiedEmbeddingConfig,
    UnscaledRoPEConfig,
    UntiedEmbeddingConfig,
)
from lalamo.modules.normalization import UpcastMode

from .common import HuggingFaceConfig

__all__ = ["HFMistralConfig"]


@dataclass(frozen=True)
class HFMistralConfig(HuggingFaceConfig):
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
    torch_dtype: Literal["bfloat16", "float16", "float32"]
    transformers_version: str
    use_cache: bool
    vocab_size: int
    head_dim: int | None = None

    def to_decoder_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
    ) -> DecoderConfig:
        # Choose embedding config based on tie_word_embeddings flag
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

        linear_config = FullPrecisionLinearConfig(
            precision=activation_precision,
        )

        attention_config = AttentionConfig(
            qkv_projection_config=linear_config,
            out_projection_config=linear_config,
            query_norm_config=None,
            key_norm_config=None,
            logit_soft_cap=None,
            has_qkv_biases=False,
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

        head_dim = self.head_dim or self.hidden_size // self.num_attention_heads

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
            head_dim=head_dim,
            attention_scale=None,
            num_layers=self.num_hidden_layers,
            sliding_window_sizes=tuple([self.sliding_window] * self.num_hidden_layers)
            if self.sliding_window is not None
            else None,
            context_length=context_length or self.max_position_embeddings,
        )
