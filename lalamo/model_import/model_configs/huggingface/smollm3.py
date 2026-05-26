from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from lalamo.modules.activations import SiLU
from lalamo.modules.decoder import DecoderConfig
from lalamo.modules.embedding import TiedEmbeddingConfig, UntiedEmbeddingConfig
from lalamo.modules.linear import LinearConfig
from lalamo.modules.mlp import DenseMLPConfig
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.rope import UnscaledRoPEConfig
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.modules.transformer import TransformerConfig
from lalamo.modules.transformer_layer import TransformerLayerConfig

from .common import HuggingFaceLMConfig, QuantizationConfigType

__all__ = ["HFSmolLM3Config"]


@dataclass(frozen=True)
class HFSmolLM3Config(HuggingFaceLMConfig):
    torch_dtype: Literal["bfloat16", "float16", "float32"]
    architectures: list[Literal["SmolLM3ForCausalLM"]]
    attention_bias: bool
    attention_dropout: float
    bos_token_id: int | list[int]
    eos_token_id: int | list[int]
    hidden_act: Literal["silu_glu", "silu"]
    hidden_size: int
    initializer_range: float
    intermediate_size: int
    max_position_embeddings: int
    mlp_bias: bool
    model_type: Literal["smollm3"]
    no_rope_layers: list[int]
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    pretraining_tp: int
    rms_norm_eps: float
    rope_theta: float
    transformers_version: str
    use_cache: bool
    vocab_size: int
    tie_word_embeddings: bool = True  # quantized model don't have this field

    quantization: QuantizationConfigType = None
    quantization_config: QuantizationConfigType = None

    def to_decoder_config(
        self,
        context_length: int | None,
        metadata_dict: Mapping[str, str],  # noqa: ARG002
    ) -> DecoderConfig:
        max_sequence_length = self.max_position_embeddings if context_length is None else context_length
        head_dim = self.hidden_size // self.num_attention_heads
        if self.tie_word_embeddings:
            embedding_config = TiedEmbeddingConfig(
                input_scale=None,
                logit_soft_cap=None,
            )
        else:
            embedding_config = UntiedEmbeddingConfig(
                input_scale=None,
                logit_soft_cap=None,
            )

        rope_config = UnscaledRoPEConfig(
            base=self.rope_theta,
            max_sequence_length=max_sequence_length,
            head_dim=head_dim,
        )

        rmsnorm_config = NormalizationConfig(
            epsilon=self.rms_norm_eps,
            scale_offset=None,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=False,
        )

        linear_config = LinearConfig()

        if len(self.no_rope_layers) < self.num_hidden_layers:
            raise ValueError(
                "SmolLM3 requires no_rope_layers to be a per-layer mask with at least num_hidden_layers entries, "
                f"got {len(self.no_rope_layers)} entries for {self.num_hidden_layers} layers.",
            )
        uses_rope_by_layer = tuple(bool(flag) for flag in self.no_rope_layers[: self.num_hidden_layers])

        attention_config = AttentionConfig(
            qkv_projection_config=linear_config,
            out_projection_config=linear_config,
            query_norm_config=None,
            key_norm_config=None,
            logit_soft_cap=None,
            has_sinks=False,
            has_qkv_biases=self.attention_bias,
            has_out_biases=self.attention_bias,
            num_heads=self.num_attention_heads,
            num_groups=self.num_key_value_heads,
            head_dim=head_dim,
            is_causal=True,
            scale=None,
            sliding_window_size=None,
        )
        mlp_config = DenseMLPConfig(
            linear_config=linear_config,
            activation=SiLU(),
            has_up_biases=self.mlp_bias,
            has_down_biases=self.mlp_bias,
            up_clipping=None,
            gate_clipping=None,
        )
        layer_configs = tuple(
            TransformerLayerConfig(
                pre_mixer_norm_config=rmsnorm_config,
                mixer_config=attention_config,
                post_mixer_norm_config=None,
                pre_mlp_norm_config=rmsnorm_config,
                mlp_config=mlp_config,
                post_mlp_norm_config=None,
                rope_config=rope_config if uses_rope else None,
            )
            for uses_rope in uses_rope_by_layer
        )

        transformer_config = TransformerConfig(
            layer_configs=layer_configs,
            output_norm_config=rmsnorm_config,
            model_dim=self.hidden_size,
            hidden_dim=self.intermediate_size,
        )
        return DecoderConfig(
            embedding_config=embedding_config,
            transformer_config=transformer_config,
            vocab_size=self.vocab_size,
        )
