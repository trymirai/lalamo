from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, DTypeLike

from lalamo.model_import.loaders.executorch import load_executorch
from lalamo.modules import (
    Activation,
    AttentionConfig,
    Decoder,
    DecoderConfig,
    DecoderLayerConfig,
    LlamaRoPEConfig,
    MLPConfig,
    QLoRALinearConfig,
    QuantizedTiedEmbeddingConfig,
    RMSNormConfig,
    UpcastMode,
)
from lalamo.quantization import QuantizationMode

from .common import ForeignConfig

__all__ = ["ETLlamaConfig"]


# These parameters are not present in the config file, and are extracted from the executorch implementation
LOW_FREQ_FACTOR = 1.0
HIGH_FREQ_FACTOR = 4.0
OLD_CONTEXT_LENGTH = 8192
MAX_SEQUENCE_LENGTH = 8192 * 32

ROPE_SCALING_FACTOR = 32.0

EMBEDDING_QUANTIZATION_MODE = QuantizationMode.INT8
ACTIVATION_QUANTIZATION_MODE = QuantizationMode.INT8
WEIGHT_QUANTIZATION_MODE = QuantizationMode.UINT4


@dataclass(frozen=True)
class QuantizationConfig:
    group_size: int


@dataclass(frozen=True)
class LoraConfig:
    rank: int
    scale: float


@dataclass(frozen=True)
class ExecutorchConfig(ForeignConfig):
    @property
    def default_precision(self) -> DTypeLike:
        return jnp.bfloat16

    @classmethod
    def _load_weights(
        cls,
        model: Decoder,
        weights_dict: dict[str, Array],
    ) -> Decoder:
        return load_executorch(model, weights_dict)


@dataclass(frozen=True)
class ETLlamaConfig(ExecutorchConfig):
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    ffn_dim_multiplier: float
    multiple_of: int
    norm_eps: float
    rope_theta: float
    use_scaled_rope: bool
    quantization_args: QuantizationConfig | None = None
    lora_args: LoraConfig | None = None

    def _find_hidden_size(self) -> int:
        # Magic formula from executorch
        size_candidate = int(8 / 3 * self.dim * self.ffn_dim_multiplier)
        return size_candidate // self.multiple_of * self.multiple_of

    def to_decoder_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
    ) -> DecoderConfig:
        if self.lora_args is None:
            raise ValueError("We only support QLoRA models for now.")

        if self.quantization_args is None:
            raise ValueError("Quantization arguments are required for QLoRA models.")

        embedding_config = QuantizedTiedEmbeddingConfig(
            input_scale=None,
            logits_soft_cap=None,
            embedding_quantization_mode=EMBEDDING_QUANTIZATION_MODE,
            activation_quantization_mode=ACTIVATION_QUANTIZATION_MODE,
            activation_precision=activation_precision,
        )
        rope_config = LlamaRoPEConfig(
            precision=activation_precision,
            base=self.rope_theta,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            scaling_factor=ROPE_SCALING_FACTOR,
            original_context_length=OLD_CONTEXT_LENGTH,
            low_frequency_factor=LOW_FREQ_FACTOR,
            high_frequency_factor=HIGH_FREQ_FACTOR,
        )
        rmsnorm_config = RMSNormConfig(
            scale_precision=activation_precision,
            accumulation_precision=accumulation_precision,
            epsilon=self.norm_eps,
            scale_offset=None,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
        )
        linear_config = QLoRALinearConfig(
            group_size=self.quantization_args.group_size,
            weight_quantization_mode=WEIGHT_QUANTIZATION_MODE,
            activation_quantization_mode=ACTIVATION_QUANTIZATION_MODE,
            activation_precision=activation_precision,
            lora_rank=self.lora_args.rank,
            lora_scale=self.lora_args.scale,
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
        return DecoderConfig(
            embedding_config=embedding_config,
            global_rope_config=rope_config,
            local_rope_config=None,
            layer_config=decoder_layer_config,
            output_norm_config=rmsnorm_config,
            vocab_size=self.vocab_size,
            model_dim=self.dim,
            hidden_dim=self._find_hidden_size(),
            num_heads=self.n_heads,
            num_groups=self.n_kv_heads,
            head_dim=self.dim // self.n_heads,
            attention_scale=None,
            num_layers=self.n_layers,
            sliding_window_sizes=None,
            context_length=context_length or MAX_SEQUENCE_LENGTH,
        )
