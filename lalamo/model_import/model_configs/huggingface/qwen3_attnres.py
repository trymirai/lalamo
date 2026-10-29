from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from jaxtyping import Array

from lalamo.model_import.loaders.binding import Binder
from lalamo.model_import.loaders.huggingface import load_rmsnorm
from lalamo.modules.activations import SiLU
from lalamo.modules.decoder import DecoderConfig
from lalamo.modules.embedding import TiedEmbeddingConfig
from lalamo.modules.flow import Add, EmitConfig, SequenceConfig, SinkConfig, StepConfig
from lalamo.modules.linear import LinearConfig
from lalamo.modules.mlp import DenseMLPConfig
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.residual_connections import AttentionResidual, AttentionResidualConfig
from lalamo.modules.rope import UnscaledRoPEConfig
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.surgery import load_as, load_as_at
from lalamo.weight_matrix import CompressionImplementation

from .common import HuggingFaceLMConfig


class AttentionResidualMode(StrEnum):
    FULL = "full"
    BLOCK = "block"


def _load_depth_reader(
    module: AttentionResidual,
    weights: Mapping[str, Array],
    path: ParameterPath,
    *,
    implementation: CompressionImplementation,  # noqa: ARG001
) -> AttentionResidual:
    assert module.recency_bias is not None
    return load_as_at(
        lambda reader: (reader.query, reader.key_norm, reader.recency_bias),
        module,
        (
            load_as(module.query, weights[ParameterPath(f"{path}_proj") / "weight"][0]),
            load_rmsnorm(module.key_norm, weights, ParameterPath(f"{path}_norm")),
            load_as(module.recency_bias, weights[ParameterPath(f"{path}_bias")]),
        ),
    )


@dataclass(frozen=True)
class HFQwen3AttnResConfig(HuggingFaceLMConfig):
    model_type: Literal["qwen3_attnres"]
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float
    vocab_size: int
    hidden_act: Literal["silu"]
    attention_bias: Literal[False]
    tie_word_embeddings: Literal[True]
    eos_token_id: Literal[None]  # noqa: PYI061  # cattrs requires Literal for a null-only field.
    attnres_mode: AttentionResidualMode
    attnres_gate_type: Literal["bias"]
    attnres_num_blocks: int
    attnres_recency_bias_init: float
    rope_parameters: "AttnResRoPEParameters"

    @property
    def eos_token_ids(self) -> list[int]:
        return []

    def build_decoder(self, at: Binder) -> DecoderConfig:
        norm = NormalizationConfig(self.rms_norm_eps, None, UpcastMode.ONLY_NORMALIZATION, subtract_mean=False)
        linear = LinearConfig()
        rope = UnscaledRoPEConfig(
            base=self.rope_parameters.rope_theta,
            max_sequence_length=self.max_position_embeddings,
            head_dim=self.head_dim,
        )
        embedding = EmitConfig("embedding")
        history = (embedding,)
        steps: list[StepConfig] = [embedding]
        taps = []
        layers_per_block = max(1, (self.num_hidden_layers + self.attnres_num_blocks - 1) // self.attnres_num_blocks)
        for index in range(self.num_hidden_layers):
            path = ParameterPath("model.layers") / index
            attention = AttentionConfig(
                qkvg_projection_config=linear,
                out_projection_config=linear,
                query_norm_config=norm,
                key_norm_config=norm,
                num_heads=self.num_attention_heads,
                num_groups=self.num_key_value_heads,
                head_dim=self.head_dim,
                is_causal=True,
                scale=None,
                sliding_window_size=None,
                logit_soft_cap=None,
                has_sinks=False,
                has_qkvg_biases=False,
                has_out_biases=False,
                rope_config=rope,
            )
            mlp = DenseMLPConfig(
                linear_config=linear,
                activation=SiLU(),
                has_up_biases=False,
                has_down_biases=False,
                up_clipping=None,
                gate_clipping=None,
                hidden_dim=self.intermediate_size,
            )
            for name, normalization, body in (
                ("attn", "input_layernorm", at(path / "self_attn", attention)),
                ("mlp", "post_attention_layernorm", at(path / "mlp", mlp)),
            ):
                reader = at(
                    ParameterPath(f"{path}.{name}_res"),
                    AttentionResidualConfig(norm, recency_bias_init=self.attnres_recency_bias_init),
                    load=_load_depth_reader,
                )
                contribution = EmitConfig(
                    name, SequenceConfig((SinkConfig(history, reader), at(path / normalization, norm), body))
                )
                steps.extend((contribution, SinkConfig((contribution,), Add())))
                # These checkpoints save cumulative states, including the current one, rather than paper-style deltas.
                if self.attnres_mode == AttentionResidualMode.FULL:
                    saved = EmitConfig(f"{name}_state")
                    steps.append(saved)
                    history += (saved,)
            tap = EmitConfig(f"hidden_states.{index}")
            steps.append(tap)
            taps.append(tap)
            if self.attnres_mode == AttentionResidualMode.BLOCK and (index + 1) % layers_per_block == 0:
                history += (tap,)
        steps.append(at("model.norm", norm))
        return DecoderConfig(
            embedding_config=at("model.embed_tokens", TiedEmbeddingConfig(input_scale=None, logit_soft_cap=None)),
            program=SequenceConfig(tuple(steps)),
            model_dim=self.hidden_size,
            vocab_size=self.vocab_size,
            taps=tuple(taps),
        )


@dataclass(frozen=True)
class AttnResRoPEParameters:
    rope_type: Literal["default"]
    rope_theta: float
