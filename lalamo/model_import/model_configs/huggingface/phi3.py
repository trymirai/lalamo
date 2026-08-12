from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from jaxtyping import Array

from lalamo.model import Model
from lalamo.modules.activations import SiLU
from lalamo.modules.decoder import DecoderConfig
from lalamo.modules.embedding import TiedEmbeddingConfig
from lalamo.modules.linear import LinearConfig
from lalamo.modules.mlp import DenseMLPConfig
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.rope import LongRoPEConfig
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.modules.transformer import TransformerConfig
from lalamo.modules.transformer_layer import TransformerLayerConfig
from lalamo.utils.lazy_collections import LazyDict
from lalamo.weight_matrix import CompressionImplementation

from .common import HuggingFaceLMConfig, QuantizationConfigType

__all__ = ["HFPhi3Config"]


@dataclass(frozen=True)
class Phi3LongRopeScalingConfig:
    type: Literal["longrope"]
    short_factor: list[float]
    long_factor: list[float]


@dataclass(frozen=True)
class HFPhi3Config(HuggingFaceLMConfig):
    eos_token_id: int | list[int]
    torch_dtype: Literal["bfloat16", "float16", "float32"]
    hidden_act: Literal["silu"]
    hidden_size: int
    intermediate_size: int
    max_position_embeddings: int
    model_type: Literal["phi3"]
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    vocab_size: int
    partial_rotary_factor: float

    original_max_position_embeddings: int
    rope_scaling: Phi3LongRopeScalingConfig
    quantization_config: QuantizationConfigType = None

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    def to_decoder_config(
        self,
        context_length: int | None,
        metadata_dict: Mapping[str, str],  # noqa: ARG002
    ) -> DecoderConfig:
        original_context_length = self.original_max_position_embeddings
        if context_length is not None and context_length > self.max_position_embeddings:
            raise ValueError(
                f"Requested context_length={context_length} exceeds the maximum context "
                f"{self.max_position_embeddings} for this model."
            )
        max_sequence_length = original_context_length if context_length is None else context_length
        assert self.tie_word_embeddings, "Phi-4-mini only has tied embeddings"
        embedding_config = TiedEmbeddingConfig(input_scale=None, logit_soft_cap=None)

        rope_config = LongRoPEConfig(
            base=self.rope_theta,
            max_sequence_length=max_sequence_length,
            head_dim=self.rotary_dim,
            short_factor=tuple(self.rope_scaling.short_factor),
            long_factor=tuple(self.rope_scaling.long_factor),
            original_context_length=original_context_length,
            scaling_factor=self.max_position_embeddings / original_context_length,
        )
        rmsnorm_config = NormalizationConfig(
            epsilon=self.rms_norm_eps,
            scale_offset=None,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=False,
        )
        linear_config = LinearConfig()
        mlp_config = DenseMLPConfig(
            linear_config=linear_config,
            activation=SiLU(),
            has_up_biases=False,
            has_down_biases=False,
            up_clipping=None,
            gate_clipping=None,
        )

        layer_configs = []
        for _ in range(self.num_hidden_layers):
            attention_config = AttentionConfig(
                qkv_projection_config=linear_config,
                out_projection_config=linear_config,
                query_norm_config=None,
                key_norm_config=None,
                logit_soft_cap=None,
                has_sinks=False,
                has_qkv_biases=False,
                has_out_biases=False,
                num_heads=self.num_attention_heads,
                num_groups=self.num_key_value_heads,
                head_dim=self.head_dim,
                is_causal=True,
                scale=None,
                sliding_window_size=None,
            )
            layer_configs.append(
                TransformerLayerConfig(
                    pre_mixer_norm_config=rmsnorm_config,
                    mixer_config=attention_config,
                    post_mixer_norm_config=None,
                    pre_mlp_norm_config=rmsnorm_config,
                    mlp_config=mlp_config,
                    post_mlp_norm_config=None,
                    rope_config=rope_config,
                )
            )
        transformer_config = TransformerConfig(
            layer_configs=tuple(layer_configs),
            output_norm_config=rmsnorm_config,
            model_dim=self.hidden_size,
            hidden_dim=self.intermediate_size,
        )
        return DecoderConfig(
            embedding_config=embedding_config,
            transformer_config=transformer_config,
            vocab_size=self.vocab_size,
        )

    def _load_weights(
        self,
        model: Model,
        weights_dict: Mapping[str, Array],
        *,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
    ) -> Model:
        return super()._load_weights(
            model,
            self._unfused_weights(weights_dict),
            implementation=implementation,
        )

    def _unfused_weights(self, weights_dict: Mapping[str, Array]) -> Mapping[str, Array]:
        head_dim = self.hidden_size // self.num_attention_heads
        q_rows = self.num_attention_heads * head_dim
        kv_rows = self.num_key_value_heads * head_dim
        gate_rows = self.intermediate_size

        slices: dict[str, tuple[str, int, int]] = {}
        passthrough: set[str] = set()
        for key in weights_dict:
            if key.endswith("self_attn.qkv_proj.weight"):
                prefix = key[: -len("qkv_proj.weight")]
                slices[prefix + "q_proj.weight"] = (key, 0, q_rows)
                slices[prefix + "k_proj.weight"] = (key, q_rows, q_rows + kv_rows)
                slices[prefix + "v_proj.weight"] = (key, q_rows + kv_rows, q_rows + 2 * kv_rows)
            elif key.endswith("mlp.gate_up_proj.weight"):
                prefix = key[: -len("gate_up_proj.weight")]
                slices[prefix + "gate_proj.weight"] = (key, 0, gate_rows)
                slices[prefix + "up_proj.weight"] = (key, gate_rows, 2 * gate_rows)
            else:
                passthrough.add(key)

        def get(requested_key: str) -> Array:
            if requested_key in slices:
                source, start, stop = slices[requested_key]
                return weights_dict[source][start:stop]
            return weights_dict[requested_key]

        return LazyDict(passthrough | set(slices), get)
