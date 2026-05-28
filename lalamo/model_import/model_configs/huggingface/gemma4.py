import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Self

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from lalamo.model import Model
from lalamo.model_import.loaders.huggingface import (
    load_huggingface_decoder,
    load_linear,
    load_rmsnorm,
)
from lalamo.models import LanguageModel
from lalamo.modules import (
    DecoderConfig,
    PLELayerConfig,
    PLEModelConfig,
    TiedEmbeddingConfig,
    TransformerConfig,
)
from lalamo.modules.activations import GELU
from lalamo.modules.decoder import Decoder
from lalamo.modules.linear import LinearConfig
from lalamo.modules.mlp import DenseMLPConfig
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.rope import ProportionalRoPEConfig, UnscaledRoPEConfig
from lalamo.modules.token_mixers.attention import AttentionConfig, AttentionProjectionMode
from lalamo.modules.transformer_layer import Gemma4MoEBlockConfig, TransformerLayerConfig
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.surgery import load_as
from lalamo.weight_matrix import CompressionImplementation

from .common import HuggingFaceLMConfig

__all__ = ["HFGemma4Config"]


@dataclass(frozen=True)
class RopeParameters:
    rope_theta: float
    rope_type: Literal["default", "proportional"]
    partial_rotary_factor: float | None = None


@dataclass(frozen=True)
class Gemma4RopeParameters:
    full_attention: RopeParameters
    sliding_attention: RopeParameters


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
    rope_parameters: Gemma4RopeParameters
    final_logit_softcapping: float | None
    vocab_size: int
    layer_types: list[Literal["sliding_attention", "full_attention"]]
    hidden_activation: Literal["gelu_pytorch_tanh"]
    hidden_size_per_layer_input: int
    vocab_size_per_layer_input: int
    num_kv_shared_layers: int
    use_double_wide_mlp: bool
    tie_word_embeddings: bool
    attention_k_eq_v: bool = False
    enable_moe_block: bool = False
    num_experts: int | None = None
    top_k_experts: int | None = None
    moe_intermediate_size: int | None = None

    def to_decoder_config(
        self,
        context_length: int | None,
        metadata_dict: Mapping[str, str],  # noqa: ARG002
    ) -> DecoderConfig:
        if not (0 <= self.num_kv_shared_layers <= self.num_hidden_layers):
            raise ValueError(
                f"num_kv_shared_layers must be in [0, {self.num_hidden_layers}], got {self.num_kv_shared_layers}.",
            )
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types length must equal num_hidden_layers,"
                f" got {len(self.layer_types)} and {self.num_hidden_layers}.",
            )
        if not self.tie_word_embeddings:
            raise ValueError("Gemma-4 import only supports tied word embeddings.")

        embedding_config = TiedEmbeddingConfig(
            input_scale=jnp.array(self.hidden_size**0.5, dtype=jnp.bfloat16).item(),
            logit_soft_cap=self.final_logit_softcapping,
        )

        rms_norm_config = NormalizationConfig(
            epsilon=self.rms_norm_eps,
            scale_offset=None,
            upcast_mode=UpcastMode.FULL_LAYER,
            subtract_mean=False,
        )

        linear_config = LinearConfig()

        mlp_config = DenseMLPConfig(
            linear_config=linear_config,
            activation=GELU(),
            has_up_biases=False,
            has_down_biases=False,
            up_clipping=None,
            gate_clipping=None,
        )

        max_sequence_length = self.max_position_embeddings if context_length is None else context_length
        full_attention_params = self.rope_parameters.full_attention
        sliding_attention_params = self.rope_parameters.sliding_attention
        full_partial_rotary_factor = (
            1.0 if full_attention_params.partial_rotary_factor is None else full_attention_params.partial_rotary_factor
        )
        match full_attention_params.rope_type:
            case "proportional":
                global_rope_config = ProportionalRoPEConfig(
                    base=full_attention_params.rope_theta,
                    max_sequence_length=max_sequence_length,
                    head_dim=self.global_head_dim,
                    partial_rotary_factor=full_partial_rotary_factor,
                )
            case "default":
                global_rope_config = UnscaledRoPEConfig(
                    base=full_attention_params.rope_theta,
                    max_sequence_length=max_sequence_length,
                    head_dim=int(full_partial_rotary_factor * self.global_head_dim),
                )
            case _:
                raise ValueError(f"Unsupported Gemma 4 full attention RoPE type {full_attention_params.rope_type}.")

        match sliding_attention_params.rope_type:
            case "default":
                local_rope_config = UnscaledRoPEConfig(
                    base=sliding_attention_params.rope_theta,
                    max_sequence_length=max_sequence_length,
                    head_dim=self.head_dim,
                )
            case "proportional":
                sliding_partial_rotary_factor = (
                    1.0
                    if sliding_attention_params.partial_rotary_factor is None
                    else sliding_attention_params.partial_rotary_factor
                )
                local_rope_config = ProportionalRoPEConfig(
                    base=sliding_attention_params.rope_theta,
                    max_sequence_length=max_sequence_length,
                    head_dim=self.head_dim,
                    partial_rotary_factor=sliding_partial_rotary_factor,
                )
            case _:
                raise ValueError(
                    f"Unsupported Gemma 4 sliding attention RoPE type {sliding_attention_params.rope_type}.",
                )

        if self.hidden_size_per_layer_input > 0:
            ple_layer_config = PLELayerConfig(
                linear_config=linear_config,
                norm_config=rms_norm_config,
                ple_dim=self.hidden_size_per_layer_input,
                activation=GELU(),
            )
            ple_model_config = PLEModelConfig(
                ple_dim=self.hidden_size_per_layer_input,
                num_layers=self.num_hidden_layers,
                ple_vocab_size=self.vocab_size_per_layer_input,
                ple_embed_scale=jnp.array(self.hidden_size_per_layer_input**0.5, dtype=jnp.bfloat16).item(),
                model_projection_scale=self.hidden_size**-0.5,
                input_scale=2.0**-0.5,
                linear_config=linear_config,
                norm_config=rms_norm_config,
            )
        else:
            ple_layer_config = None
            ple_model_config = None

        if self.enable_moe_block:
            if self.num_experts is None or self.top_k_experts is None or self.moe_intermediate_size is None:
                raise ValueError("Gemma 4 MoE config requires num_experts, top_k_experts, and moe_intermediate_size.")
            gemma4_moe_config = Gemma4MoEBlockConfig(
                expert_config=mlp_config,
                router_config=linear_config,
                norm_config=rms_norm_config,
                num_experts=self.num_experts,
                num_active_experts=self.top_k_experts,
                expert_hidden_dim=self.moe_intermediate_size,
                router_norm_epsilon=self.rms_norm_eps,
            )
        else:
            gemma4_moe_config = None

        first_kv_shared = self.num_hidden_layers - self.num_kv_shared_layers
        last_of_type: dict[str, int] = {}
        kv_source_per_layer: list[int] = []
        for i, lt in enumerate(self.layer_types):
            if i < first_kv_shared:
                kv_source_per_layer.append(i)
                last_of_type[lt] = i
            else:
                source = last_of_type.get(lt)
                assert source is not None, f"layer {i} marked shared but no prior {lt} layer exists"
                kv_source_per_layer.append(source)

        layer_configs = []
        for i, layer_type in enumerate(self.layer_types):
            is_global = layer_type == "full_attention"
            sliding_window_size = None if is_global else self.sliding_window
            layer_head_dim = self.global_head_dim if is_global else self.head_dim
            layer_rope_config = global_rope_config if is_global else local_rope_config

            kv_source = kv_source_per_layer[i]
            owns_kv_cache = kv_source == i
            if not owns_kv_cache:
                projection_mode = AttentionProjectionMode.BORROWED_Q
            elif is_global and self.attention_k_eq_v:
                projection_mode = AttentionProjectionMode.QK_SHARED_VALUE
            else:
                projection_mode = AttentionProjectionMode.QKV

            # `use_double_wide_mlp` only applies to layers that share a KV cache
            # with an earlier layer — those layers get a 2x-wide MLP to compensate.
            if self.use_double_wide_mlp and not owns_kv_cache:
                layer_intermediate = self.intermediate_size * 2
            else:
                layer_intermediate = self.intermediate_size

            num_kv_heads = self.num_key_value_heads
            if is_global and self.attention_k_eq_v and self.num_global_key_value_heads is not None:
                num_kv_heads = self.num_global_key_value_heads

            attention_config = AttentionConfig(
                qkv_projection_config=linear_config,
                out_projection_config=linear_config,
                query_norm_config=rms_norm_config,
                key_norm_config=rms_norm_config if owns_kv_cache else None,
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
                projection_mode=projection_mode,
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
                gemma4_moe_config=gemma4_moe_config,
                has_post_layer_scalar=True,
                rope_config=layer_rope_config,
            )
            layer_configs.append(transformer_layer_config)

        transformer_config = TransformerConfig(
            layer_configs=tuple(layer_configs),
            output_norm_config=rms_norm_config,
            model_dim=self.hidden_size,
            hidden_dim=self.intermediate_size,
            kv_source_per_layer=tuple(kv_source_per_layer),
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
    dtype: Literal["bfloat16", "float16", "float32"]
    model_type: Literal["gemma4"]
    eos_token_id: list[int]

    @classmethod
    def from_json(cls, json_path: Path | str) -> Self:
        json_path = Path(json_path)
        with open(json_path) as f:
            config = json.load(f)
        eos_token_id = config["eos_token_id"] if "eos_token_id" in config else config["text_config"]["eos_token_id"]
        config["eos_token_id"] = [eos_token_id] if isinstance(eos_token_id, int) else list(eos_token_id)
        return cls._converter.structure(config, cls)

    def to_decoder_config(
        self,
        context_length: int | None,
        metadata_dict: Mapping[str, str],
    ) -> DecoderConfig:
        return self.text_config.to_decoder_config(
            context_length=context_length,
            metadata_dict=metadata_dict,
        )

    def _load_weights(
        self,
        model: Model,
        weights_dict: Mapping[str, Array],
        *,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
    ) -> Model:
        assert isinstance(model, LanguageModel)
        decoder = load_huggingface_decoder(
            module=model.decoder,
            weights_dict=weights_dict,
            implementation=implementation,
        )
        decoder = self._load_ple_weights(decoder, weights_dict, implementation=implementation)
        return eqx.tree_at(lambda m: (m.decoder,), model, (decoder,))

    def _load_ple_weights(
        self,
        model: Decoder,
        weights_dict: Mapping[str, Array],
        *,
        implementation: CompressionImplementation,
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
            token_embedding=load_as(
                model.per_layer_embedding.token_embedding,
                weights_dict[base / "embed_tokens_per_layer" / "weight"],
            ),
            model_projection=load_linear(
                model.per_layer_embedding.model_projection,
                weights_dict,
                base / "per_layer_model_projection",
                implementation=implementation,
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
            new_ple = replace(
                layer.ple,
                gate=load_linear(
                    layer.ple.gate,
                    weights_dict,
                    layer_path / "per_layer_input_gate",
                    implementation=implementation,
                ),
                projection=load_linear(
                    layer.ple.projection,
                    weights_dict,
                    layer_path / "per_layer_projection",
                    implementation=implementation,
                ),
                norm=load_rmsnorm(layer.ple.norm, weights_dict, layer_path / "post_per_layer_input_norm"),
            )
            layer_updates: dict[str, Any] = {"ple": new_ple}
            if layer.post_layer_scalar is not None:
                layer_updates["post_layer_scalar"] = load_as(
                    layer.post_layer_scalar,
                    weights_dict[layer_path / "layer_scalar"],
                )
            new_layers.append(replace(layer, **layer_updates))

        return replace(
            model,
            per_layer_embedding=new_per_layer_embedding,
            transformer=replace(model.transformer, layers=tuple(new_layers)),
        )
