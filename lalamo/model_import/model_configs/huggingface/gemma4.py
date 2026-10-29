import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Self

from lalamo.model_import.loaders.binding import Binder
from lalamo.modules.activations import GELU
from lalamo.modules.decoder import DecoderConfig
from lalamo.modules.embedding import TiedEmbeddingConfig
from lalamo.modules.flow import Add, EmitConfig, SequenceConfig, SinkConfig, StepConfig, residual
from lalamo.modules.linear import LinearConfig
from lalamo.modules.mlp import DenseMLPConfig, Gemma4RouterConfig, RoutedExpertsConfig
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.per_layer import (
    MultiplyByLayerInput,
    PerLayerInputsConfig,
    ProjectionConfig,
    ScaleConfig,
)
from lalamo.modules.rope import ProportionalRoPEConfig, RoPEConfig, UnscaledRoPEConfig
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.utils.parameter_path import ParameterPath

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
    attention_k_eq_v: bool
    enable_moe_block: bool
    num_experts: int | None
    top_k_experts: int | None
    moe_intermediate_size: int | None = None

    def build_decoder(self, at: Binder) -> DecoderConfig:
        assert self.tie_word_embeddings, "Gemma-4 import only supports tied word embeddings"
        embedding_config = TiedEmbeddingConfig(
            input_scale=self.hidden_size**0.5,
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
            hidden_dim=self.intermediate_size,
        )

        router_config = None
        experts_config = None
        if self.enable_moe_block:
            if self.num_experts is None or self.top_k_experts is None or self.moe_intermediate_size is None:
                raise ValueError("Gemma4 MoE requires expert count, top-k, and intermediate size.")
            router_config = Gemma4RouterConfig(
                linear_config=linear_config,
                norm_config=replace(rms_norm_config, has_scale=False),
                num_experts=self.num_experts,
                num_active_experts=self.top_k_experts,
            )
            # The experts are routed by the separately emitted router.
            experts_config = RoutedExpertsConfig(
                expert_config=replace(mlp_config, hidden_dim=self.moe_intermediate_size),
                num_routed_experts=self.num_experts,
            )

        def rope_config(parameters: RopeParameters, head_dim: int) -> RoPEConfig:
            factor = parameters.partial_rotary_factor
            if factor is None:
                factor = 1.0
            if parameters.rope_type == "proportional":
                return ProportionalRoPEConfig(
                    base=parameters.rope_theta,
                    max_sequence_length=self.max_position_embeddings,
                    head_dim=head_dim,
                    partial_rotary_factor=factor,
                )
            return UnscaledRoPEConfig(
                base=parameters.rope_theta,
                max_sequence_length=self.max_position_embeddings,
                head_dim=int(factor * head_dim),
            )

        global_rope_config = rope_config(self.rope_parameters.full_attention, self.global_head_dim)
        local_rope_config = rope_config(self.rope_parameters.sliding_attention, self.head_dim)

        steps: list[StepConfig] = []
        per_layer_inputs = None
        if self.hidden_size_per_layer_input:
            per_layer_inputs = EmitConfig(
                "per_layer_inputs",
                at(
                    "model",
                    PerLayerInputsConfig(
                        ple_dim=self.hidden_size_per_layer_input,
                        num_layers=self.num_hidden_layers,
                        ple_vocab_size=self.vocab_size_per_layer_input,
                        ple_embed_scale=self.hidden_size_per_layer_input**0.5,
                        model_projection_scale=self.hidden_size**-0.5,
                        input_scale=2.0**-0.5,
                        linear_config=linear_config,
                        norm_config=rms_norm_config,
                    ),
                ),
            )
            steps.append(per_layer_inputs)

        # The last `num_kv_shared_layers` layers read the cache of the last earlier owner of the same type.
        first_kv_shared = self.num_hidden_layers - self.num_kv_shared_layers
        last_owner_of_type: dict[str, AttentionConfig] = {}
        taps: list[EmitConfig] = []
        for index, layer_type in enumerate(self.layer_types):
            path = ParameterPath("model.layers") / index
            is_global = layer_type == "full_attention"
            kv_source = last_owner_of_type[layer_type] if index >= first_kv_shared else None

            num_kv_heads = self.num_key_value_heads
            if is_global and self.attention_k_eq_v and self.num_global_key_value_heads is not None:
                num_kv_heads = self.num_global_key_value_heads

            attention = at(
                path / "self_attn",
                AttentionConfig(
                    qkvg_projection_config=linear_config,
                    out_projection_config=linear_config,
                    query_norm_config=rms_norm_config,
                    key_norm_config=rms_norm_config,
                    logit_soft_cap=None,
                    has_sinks=False,
                    has_qkvg_biases=self.attention_bias,
                    has_out_biases=self.attention_bias,
                    num_heads=self.num_attention_heads,
                    num_groups=num_kv_heads,
                    head_dim=self.global_head_dim if is_global else self.head_dim,
                    is_causal=True,
                    scale=1.0,
                    sliding_window_size=None if is_global else self.sliding_window,
                    value_norm_config=replace(rms_norm_config, has_scale=False) if kv_source is None else None,
                    kv_source=kv_source,
                    values_from_keys=self.attention_k_eq_v and is_global and kv_source is None,
                    rope_config=global_rope_config if is_global else local_rope_config,
                ),
            )
            if kv_source is None:
                last_owner_of_type[layer_type] = attention

            # `use_double_wide_mlp` only applies to layers that share a KV cache
            # with an earlier layer — those layers get a 2x-wide MLP to compensate.
            layer_mlp_config = mlp_config
            if self.use_double_wide_mlp and kv_source is not None:
                layer_mlp_config = replace(mlp_config, hidden_dim=self.intermediate_size * 2)

            dense_steps = (
                at(path / "pre_feedforward_layernorm", rms_norm_config),
                at(path / "mlp", layer_mlp_config),
            )
            if router_config is None or experts_config is None:
                feed_forward = residual(*dense_steps, at(path / "post_feedforward_layernorm", rms_norm_config))
            else:
                routing = EmitConfig("routing", at(path / "router", router_config))
                dense = EmitConfig(
                    "dense",
                    SequenceConfig((*dense_steps, at(path / "post_feedforward_layernorm_1", rms_norm_config))),
                )
                feed_forward = residual(
                    routing,
                    dense,
                    at(path / "pre_feedforward_layernorm_2", rms_norm_config),
                    SinkConfig((routing,), at(path, experts_config)),
                    at(path / "post_feedforward_layernorm_2", rms_norm_config),
                    SinkConfig((dense,), Add()),
                    at(path / "post_feedforward_layernorm", rms_norm_config),
                )

            steps += [
                residual(
                    at(path / "input_layernorm", rms_norm_config),
                    attention,
                    at(path / "post_attention_layernorm", rms_norm_config),
                ),
                feed_forward,
            ]
            if per_layer_inputs is not None:
                steps.append(
                    residual(
                        at(
                            path / "per_layer_input_gate",
                            ProjectionConfig(linear_config, output_dim=self.hidden_size_per_layer_input),
                        ),
                        GELU(),
                        SinkConfig((per_layer_inputs,), MultiplyByLayerInput(index)),
                        at(
                            path / "per_layer_projection",
                            ProjectionConfig(linear_config, input_dim=self.hidden_size_per_layer_input),
                        ),
                        at(path / "post_per_layer_input_norm", rms_norm_config),
                    ),
                )
            tap = EmitConfig(f"hidden_states.{index}")
            steps += [at(path / "layer_scalar", ScaleConfig()), tap]
            taps.append(tap)
        steps.append(at("model.norm", rms_norm_config))

        return DecoderConfig(
            at("model.embed_tokens", embedding_config),
            SequenceConfig(tuple(steps)),
            self.hidden_size,
            self.vocab_size,
            tuple(taps),
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
        eos_token_id = config.get("eos_token_id", config["text_config"]["eos_token_id"])
        config["eos_token_id"] = [eos_token_id] if isinstance(eos_token_id, int) else list(eos_token_id)
        return cls._converter.structure(config, cls)

    def build_decoder(self, at: Binder) -> DecoderConfig:
        return self.text_config.build_decoder(at)
