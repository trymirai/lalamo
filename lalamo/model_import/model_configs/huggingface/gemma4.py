import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Self

from jax import numpy as jnp
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


@dataclass(frozen=True)
class HFGemma4TextConfig:
    hidden_size: int
    intermediate_size: int
    model_type: Literal["gemma4_text"]
    num_hidden_layers: int
    sliding_window: int
    rms_norm_eps: float
    attention_bias: bool
    num_attention_heads: int
    num_key_value_heads: int
    num_global_key_value_heads: int | None  # None for E2B/E4B, 4 for 31B
    head_dim: int
    global_head_dim: int
    max_position_embeddings: int
    rope_parameters: dict[str, Any]
    final_logit_softcapping: float | None
    vocab_size: int
    layer_types: list[Literal["sliding_attention", "full_attention"]]
    hidden_activation: str
    hidden_size_per_layer_input: int
    vocab_size_per_layer_input: int
    num_kv_shared_layers: int
    use_double_wide_mlp: bool
    tie_word_embeddings: bool

    def to_decoder_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
        metadata_dict: Mapping[str, str],  # noqa: ARG002
    ) -> DecoderConfig:
        embedding_config = TiedEmbeddingConfig(
            input_scale=self.hidden_size**0.5,
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

        max_seq_len = context_length or self.max_position_embeddings
        full_attention_params = self.rope_parameters.get("full_attention", {})
        partial_rotary_factor = full_attention_params.get("partial_rotary_factor", 1.0)
        global_rope_config = UnscaledRoPEConfig(
            precision=activation_precision,
            base=full_attention_params.get("rope_theta", 10000.0),
            max_sequence_length=max_seq_len,
            head_dim=self.global_head_dim,
            partial_rotary_dim=int(partial_rotary_factor * self.global_head_dim),
        )
        local_rope_config = UnscaledRoPEConfig(
            precision=activation_precision,
            base=self.rope_parameters.get("sliding_attention", {}).get("rope_theta", 10000.0),
            max_sequence_length=max_seq_len,
            head_dim=self.head_dim,
        )

        ple_embed_scale = self.hidden_size_per_layer_input**0.5
        inverse_hidden_scale = self.hidden_size**-0.5
        two_input_mixing_scale = 2.0**-0.5
        ple_layer_config = PLELayerConfig(
            linear_config=linear_config,
            norm_config=rms_norm_config,
            ple_dim=self.hidden_size_per_layer_input,
            activation=GELU(),
            has_layer_scalar=True,
        )
        ple_model_config = PLEModelConfig(
            ple_dim=self.hidden_size_per_layer_input,
            num_layers=self.num_hidden_layers,
            ple_vocab_size=self.vocab_size_per_layer_input,
            ple_embed_scale=ple_embed_scale,
            model_projection_scale=inverse_hidden_scale,
            input_scale=two_input_mixing_scale,
            linear_config=linear_config,
            norm_config=rms_norm_config,
        )

        first_kv_shared = self.num_hidden_layers - self.num_kv_shared_layers
        last_of_type: dict[str, int] = {}
        kv_source_per_layer: dict[int, int | None] = {}
        for i, lt in enumerate(self.layer_types):
            if i < first_kv_shared or first_kv_shared <= 0:
                kv_source_per_layer[i] = None
                last_of_type[lt] = i
            else:
                kv_source_per_layer[i] = last_of_type.get(lt)

        layer_configs = []
        for i, layer_type in enumerate(self.layer_types):
            is_global = layer_type == "full_attention"
            sliding_window_size = None if is_global else self.sliding_window
            layer_head_dim = self.global_head_dim if is_global else self.head_dim
            layer_rope_config = global_rope_config if is_global else local_rope_config

            kv_source = kv_source_per_layer[i]

            if self.use_double_wide_mlp and kv_source is not None:
                layer_intermediate = self.intermediate_size * 2
            else:
                layer_intermediate = self.intermediate_size

            num_kv_heads = self.num_key_value_heads
            if is_global and self.num_global_key_value_heads is not None:
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
                kv_source_layer=kv_source,
                rope_config=layer_rope_config,
            )
            layer_configs.append(transformer_layer_config)

        transformer_config = TransformerConfig(
            layer_configs=tuple(layer_configs),
            output_norm_config=rms_norm_config,
            model_dim=self.hidden_size,
            hidden_dim=self.intermediate_size,
            context_length=max_seq_len,
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
    initializer_range: float
    transformers_version: str
    dtype: Literal["bfloat16", "float16", "float32"]
    architectures: list[str]
    model_type: Literal["gemma4"]
    tie_word_embeddings: bool
    eos_token_id: int | list[int]

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

    quantization: QuantizationConfigType = None
    quantization_config: QuantizationConfigType = None

    @property
    def default_precision(self) -> DTypeLike:
        return jnp.dtype(self.dtype)

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
        if model.per_layer_embedding is None:
            return model

        if any(key.startswith("model.language_model.") for key in weights_dict):
            base = ParameterPath("model") / "language_model"
        elif any(key.startswith("language_model.") for key in weights_dict):
            base = ParameterPath("language_model")
        else:
            base = ParameterPath("model")

        new_per_layer_embedding = replace(
            model.per_layer_embedding,
            token_embedding=weights_dict[base / "embed_tokens_per_layer" / "weight"],
            model_projection=load_linear(
                model.per_layer_embedding.model_projection,
                weights_dict,
                base / "per_layer_model_projection",
            ),
            projection_norm=load_rmsnorm(
                model.per_layer_embedding.projection_norm,
                weights_dict,
                base / "per_layer_projection_norm",
            ),
        )

        layers_base = base / "layers"
        new_layers = []
        for i, layer in enumerate(model.transformer.layers):
            if layer.ple is None:
                new_layers.append(layer)
                continue
            layer_path = layers_base / i
            ple_updates: dict[str, Any] = {
                "gate": load_linear(layer.ple.gate, weights_dict, layer_path / "per_layer_input_gate"),
                "projection": load_linear(layer.ple.projection, weights_dict, layer_path / "per_layer_projection"),
                "norm": load_rmsnorm(layer.ple.norm, weights_dict, layer_path / "post_per_layer_input_norm"),
            }
            if layer.ple.layer_scalar is not None:
                ple_updates["layer_scalar"] = weights_dict[layer_path / "layer_scalar"]
            new_layers.append(replace(layer, ple=replace(layer.ple, **ple_updates)))

        return replace(
            model,
            per_layer_embedding=new_per_layer_embedding,
            transformer=replace(model.transformer, layers=tuple(new_layers)),
        )
