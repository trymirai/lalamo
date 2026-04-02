import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Self

import jax.numpy as jnp
from jaxtyping import Array, DTypeLike

from lalamo.common import ParameterPath
from lalamo.model_import.loaders.huggingface import (
    load_huggingface_decoder,
    load_linear,
    load_rmsnorm,
)
from lalamo.modules import (
    DecoderConfig,
    PLELayerConfig,
    PLEModelConfig,
    TiedEmbeddingConfig,
    TransformerConfig,
)
from lalamo.modules.activations import GELU
from lalamo.modules.common import LalamoModule
from lalamo.modules.decoder import Decoder
from lalamo.modules.linear import FullPrecisionLinearConfig
from lalamo.modules.mlp import DenseMLPConfig
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.rope import UnscaledRoPEConfig
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.modules.transformer_layer import TransformerLayerConfig

from .common import HuggingFaceLMConfig, QuantizationConfigType

__all__ = ["HFGemma4Config"]


def _round_to_bfloat16(x: float) -> float:
    return jnp.asarray(x).astype(jnp.bfloat16).item()


@dataclass(frozen=True)
class Gemma4RoPEParameters:
    rope_theta: float
    rope_type: Literal["default", "proportional"]
    partial_rotary_factor: float | None = None


@dataclass(frozen=True)
class HFGemma4TextConfig:
    hidden_size: int
    intermediate_size: int
    model_type: Literal["gemma4_text"]
    num_hidden_layers: int
    sliding_window: int
    rms_norm_eps: float = 1e-06
    attention_bias: bool = False
    num_attention_heads: int = 8
    num_key_value_heads: int = 1
    num_global_key_value_heads: int | None = None
    head_dim: int = 256
    global_head_dim: int = 512
    max_position_embeddings: int = 131072
    rope_parameters: dict[str, Any] = field(default_factory=dict)
    final_logit_softcapping: float | None = 30.0
    vocab_size: int = 262144
    layer_types: list[Literal["sliding_attention", "full_attention"]] = field(default_factory=list)
    hidden_activation: str = "gelu_pytorch_tanh"
    hidden_size_per_layer_input: int = 0
    vocab_size_per_layer_input: int = 262144
    num_kv_shared_layers: int = 0
    use_double_wide_mlp: bool = False
    tie_word_embeddings: bool = True

    @property
    def _sliding_window_sizes(self) -> list[int | None]:
        return [self.sliding_window if lt == "sliding_attention" else None for lt in self.layer_types]

    def _head_dim_for_layer(self, layer_idx: int) -> int:
        if self.layer_types[layer_idx] == "full_attention":
            return self.global_head_dim
        return self.head_dim

    def _kv_source_layer(self, layer_idx: int) -> int | None:
        first_kv_shared = self.num_hidden_layers - self.num_kv_shared_layers
        if layer_idx < first_kv_shared or first_kv_shared <= 0:
            return None
        prev_types = self.layer_types[:first_kv_shared]
        target_type = self.layer_types[layer_idx]
        # Find the last non-shared layer of the same attention type
        for j in range(len(prev_types) - 1, -1, -1):
            if prev_types[j] == target_type:
                return j
        return None

    def _intermediate_size_for_layer(self, layer_idx: int) -> int:
        first_kv_shared = self.num_hidden_layers - self.num_kv_shared_layers
        is_shared = layer_idx >= first_kv_shared > 0
        if self.use_double_wide_mlp and is_shared:
            return self.intermediate_size * 2
        return self.intermediate_size

    def _rope_config_for_type(self, layer_type: str) -> Gemma4RoPEParameters:
        params = self.rope_parameters.get(layer_type, {})
        return Gemma4RoPEParameters(
            rope_theta=params.get("rope_theta", 10000.0),
            rope_type=params.get("rope_type", "default"),
            partial_rotary_factor=params.get("partial_rotary_factor"),
        )

    def to_decoder_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
        metadata_dict: Mapping[str, str],  # noqa: ARG002
    ) -> DecoderConfig:
        input_scale = _round_to_bfloat16(self.hidden_size**0.5)

        embedding_config = TiedEmbeddingConfig(
            input_scale=input_scale,
            logit_soft_cap=self.final_logit_softcapping,
            precision=activation_precision,
        )

        rms_norm_config = NormalizationConfig(
            scale_precision=activation_precision,
            accumulation_precision=accumulation_precision,
            epsilon=self.rms_norm_eps,
            scale_offset=None,
            upcast_mode=UpcastMode.FULL_LAYER,
            subtract_mean=False,
        )

        linear_config = FullPrecisionLinearConfig(precision=activation_precision)

        mlp_config = DenseMLPConfig(
            linear_config=linear_config,
            activation=GELU(),
            has_up_biases=False,
            has_down_biases=False,
            up_clipping=None,
            gate_clipping=None,
        )

        # Global RoPE (for full attention layers)
        full_rope_params = self._rope_config_for_type("full_attention")
        global_rope_config = UnscaledRoPEConfig(
            precision=activation_precision,
            base=full_rope_params.rope_theta,
            max_sequence_length=context_length or self.max_position_embeddings,
        )

        # Local RoPE (for sliding attention layers)
        sliding_rope_params = self._rope_config_for_type("sliding_attention")
        local_rope_config = UnscaledRoPEConfig(
            precision=activation_precision,
            base=sliding_rope_params.rope_theta,
            max_sequence_length=context_length or self.max_position_embeddings,
        )

        # PLE layer config
        ple_layer_config = None
        if self.hidden_size_per_layer_input > 0:
            ple_layer_config = PLELayerConfig(
                linear_config=linear_config,
                norm_config=rms_norm_config,
                ple_dim=self.hidden_size_per_layer_input,
                activation=GELU(),
            )

        # Build per-layer configs
        layer_configs = []
        for i, sliding_window_size in enumerate(self._sliding_window_sizes):
            layer_head_dim = self._head_dim_for_layer(i)
            layer_intermediate = self._intermediate_size_for_layer(i)

            num_kv_heads = self.num_key_value_heads
            if sliding_window_size is None and self.num_global_key_value_heads is not None:
                num_kv_heads = self.num_global_key_value_heads

            attention_config = AttentionConfig(
                qkv_projection_config=linear_config,
                out_projection_config=linear_config,
                query_norm_config=rms_norm_config,
                key_norm_config=rms_norm_config,
                logit_soft_cap=None,
                has_sinks=False,
                has_qkv_biases=self.attention_bias,
                has_out_biases=self.attention_bias,
                num_heads=self.num_attention_heads,
                num_groups=num_kv_heads,
                head_dim=layer_head_dim,
                is_causal=True,
                scale=1.0,
                sliding_window_size=sliding_window_size,
                normalize_values=True,
            )

            transformer_layer_config = TransformerLayerConfig(
                pre_mixer_norm_config=rms_norm_config,
                mixer_config=attention_config,
                post_mixer_norm_config=rms_norm_config,
                pre_mlp_norm_config=rms_norm_config,
                mlp_config=mlp_config,
                post_mlp_norm_config=rms_norm_config,
                hidden_dim=layer_intermediate,
                ple_config=ple_layer_config,
                has_layer_scalar=True,
                kv_source_layer=self._kv_source_layer(i),
            )
            layer_configs.append(transformer_layer_config)

        transformer_config = TransformerConfig(
            global_rope_config=global_rope_config,
            local_rope_config=local_rope_config,
            layer_configs=tuple(layer_configs),
            output_norm_config=rms_norm_config,
            model_dim=self.hidden_size,
            hidden_dim=self.intermediate_size,
            context_length=context_length or self.max_position_embeddings,
        )

        ple_model_config = None
        if self.hidden_size_per_layer_input > 0:
            ple_model_config = PLEModelConfig(
                ple_dim=self.hidden_size_per_layer_input,
                num_layers=self.num_hidden_layers,
                ple_vocab_size=self.vocab_size_per_layer_input,
                ple_embed_scale=self.hidden_size_per_layer_input**0.5,
                model_projection_scale=self.hidden_size**-0.5,
                input_scale=2.0**-0.5,
                linear_config=linear_config,
                norm_config=NormalizationConfig(
                    scale_precision=activation_precision,
                    accumulation_precision=accumulation_precision,
                    epsilon=self.rms_norm_eps,
                    scale_offset=None,
                    upcast_mode=UpcastMode.FULL_LAYER,
                    subtract_mean=False,
                ),
            )

        return DecoderConfig(
            embedding_config=embedding_config,
            transformer_config=transformer_config,
            vocab_size=self.vocab_size,
            ple_model_config=ple_model_config,
        )


@dataclass(frozen=True)
class HFGemma4Config(HuggingFaceLMConfig):
    text_config: HFGemma4TextConfig
    torch_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    architectures: list[str] = field(default_factory=list)
    model_type: Literal["gemma4"] = "gemma4"
    tie_word_embeddings: bool = True
    eos_token_id: int | list[int] = field(default_factory=lambda: [1])

    # Multimodal fields (ignored for text-only)
    vision_config: dict[str, Any] | None = field(default_factory=dict)
    audio_config: dict[str, Any] | None = field(default_factory=dict)
    image_token_id: int | None = None
    audio_token_id: int | None = None
    boi_token_id: int | None = None
    eoi_token_id: int | None = None
    video_token_id: int | None = None
    eoa_token_id: int | None = None
    eoa_token_index: int | None = None
    boa_token_id: int | None = None
    vision_soft_tokens_per_image: int | None = None
    initializer_range: float = 0.02
    transformers_version: str = ""

    quantization: QuantizationConfigType = None
    quantization_config: QuantizationConfigType = None

    @classmethod
    def from_json(cls, json_path: Path | str) -> Self:
        json_path = Path(json_path)
        with open(json_path) as f:
            config = json.load(f)
        return cls._converter.structure(config, cls)

    def to_decoder_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
        metadata_dict: Mapping[str, str],
    ) -> DecoderConfig:
        return self.text_config.to_decoder_config(
            context_length=context_length,
            activation_precision=activation_precision,
            accumulation_precision=accumulation_precision,
            metadata_dict=metadata_dict,
        )

    def _load_weights(
        self,
        model: LalamoModule,
        weights_dict: Mapping[str, Array],
    ) -> LalamoModule:
        assert isinstance(model, Decoder)
        model = load_huggingface_decoder(model, weights_dict)
        return self._load_ple_weights(model, weights_dict)

    def _load_ple_weights(
        self,
        model: Decoder,
        weights_dict: Mapping[str, Array],
    ) -> Decoder:
        if model.config.ple_model_config is None:
            return model

        # Determine the base path for language model weights
        if any(key.startswith("model.language_model.") for key in weights_dict):
            base = ParameterPath("model") / "language_model"
        elif any(key.startswith("language_model.") for key in weights_dict):
            base = ParameterPath("language_model")
        else:
            base = ParameterPath("model")

        # Load PLE token embedding
        ple_token_embedding = weights_dict[base / "embed_tokens_per_layer" / "weight"]

        # Load PLE model projection
        assert model.ple_model_projection is not None
        ple_model_projection = load_linear(
            model.ple_model_projection,
            weights_dict,
            base / "per_layer_model_projection",
        )

        # Load PLE projection norm
        assert model.ple_projection_norm is not None
        ple_projection_norm = load_rmsnorm(
            model.ple_projection_norm,
            weights_dict,
            base / "per_layer_projection_norm",
        )

        # Load per-layer PLE weights (gate, projection, norm) and layer_scalar
        layers_base = base / "layers"
        new_layers = []
        for i, layer in enumerate(model.transformer.layers):
            layer_path = layers_base / i

            updates = {}

            if layer.ple_gate is not None:
                updates["ple_gate"] = load_linear(
                    layer.ple_gate,
                    weights_dict,
                    layer_path / "per_layer_input_gate",
                )
            if layer.ple_projection is not None:
                updates["ple_projection"] = load_linear(
                    layer.ple_projection,
                    weights_dict,
                    layer_path / "per_layer_projection",
                )
            if layer.ple_norm is not None:
                updates["ple_norm"] = load_rmsnorm(
                    layer.ple_norm,
                    weights_dict,
                    layer_path / "post_per_layer_input_norm",
                )
            if layer.layer_scalar is not None:
                updates["layer_scalar"] = weights_dict[layer_path / "layer_scalar"]

            new_layers.append(replace(layer, **updates))

        new_transformer = replace(model.transformer, layers=tuple(new_layers))

        return replace(
            model,
            ple_token_embedding=ple_token_embedding,
            ple_model_projection=ple_model_projection,
            ple_projection_norm=ple_projection_norm,
            transformer=new_transformer,
        )
