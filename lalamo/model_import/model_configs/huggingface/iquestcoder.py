from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from jaxtyping import DTypeLike

from lalamo.modules import (
    AttentionConfig,
    DecoderConfig,
    DenseMLPConfig,
    FullPrecisionLinearConfig,
    GroupQuantizedLinearConfig,
    LinearScalingRoPEConfig,
    LlamaRoPEConfig,
    NormalizationConfig,
    SiLU,
    TiedEmbeddingConfig,
    TransformerConfig,
    TransformerLayerConfig,
    UnscaledRoPEConfig,
    UntiedEmbeddingConfig,
    UpcastMode,
    YARNRoPEConfig,
)
from lalamo.quantization import QuantizationMode

from .common import AWQQuantizationConfig, GPTQQuantizationConfig, HuggingFaceLMConfig

__all__ = ["HFIQuestCoderConfig"]


@dataclass(frozen=True)
class HFIQuestCoderConfig(HuggingFaceLMConfig):
    """
    HuggingFace `config.json` adapter for IQuestCoder-style decoder-only models.

    The upstream configuration (`configuration_iquestcoder.py`) is broadly Llama-like
    but adds:
    - `clip_qkv`: optional symmetric clipping applied to projected Q/K/V
    - Qwen2-like sliding-window attention toggles
    """

    # Core HF fields (mirrors configuration_iquestcoder.py)
    torch_dtype: Literal["bfloat16", "float16", "float32"]
    architectures: list[str]
    attention_bias: bool
    attention_dropout: float
    bos_token_id: int | list[int]
    eos_token_id: int | list[int]
    hidden_act: Literal["silu"]
    hidden_size: int
    initializer_range: float
    intermediate_size: int
    max_position_embeddings: int
    mlp_bias: bool
    model_type: Literal["iquestcoder"]
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    transformers_version: str
    use_cache: bool
    vocab_size: int

    # Optional/extended HF fields
    # Present in custom-code configs via `auto_map` (used by HF for remote code loading).
    auto_map: dict[str, str] | None = None
    # Matches Llama-style configs; safe default keeps structuring tolerant.
    pretraining_tp: int = 1
    head_dim: int | None = None
    pad_token_id: int | None = None

    # IQuestCoder specifics (OLMo-inspired stability)
    clip_qkv: float | None = None

    # IQuestCoder specifics (Qwen2-inspired sliding window)
    use_sliding_window: bool = False
    sliding_window: int | None = None
    max_window_layers: int = 0

    # Quantization (same shape as other HF configs in lalamo)
    quantization_config: AWQQuantizationConfig | GPTQQuantizationConfig | None = None

    # RoPE scaling: HF allows multiple shapes ("type" or "rope_type"), so keep dict form.
    rope_scaling: dict[str, object] | None = None

    def _get_sliding_window_sizes(self) -> list[int | None]:
        if not self.use_sliding_window:
            return [None] * self.num_hidden_layers

        if self.sliding_window is None:
            raise ValueError("use_sliding_window=True but sliding_window is None")

        # Borrowed semantics from our Qwen2 adapter: apply SWA to the first max_window_layers.
        return [self.sliding_window if i < self.max_window_layers else None for i in range(self.num_hidden_layers)]

    def to_decoder_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
        metadata_dict: Mapping[str, str],  # noqa: ARG002
    ) -> DecoderConfig:
        # Embedding
        if self.tie_word_embeddings:
            embedding_config = TiedEmbeddingConfig(
                input_scale=None,
                logit_soft_cap=None,
                precision=activation_precision,
            )
        else:
            embedding_config = UntiedEmbeddingConfig(
                input_scale=None,
                logit_soft_cap=None,
                precision=activation_precision,
            )

        # RoPE
        if self.rope_scaling is None:
            rope_config = UnscaledRoPEConfig(
                precision=activation_precision,
                base=self.rope_theta,
                max_sequence_length=context_length or self.max_position_embeddings,
            )
        else:
            rope_scaling_type = self.rope_scaling.get("type") or self.rope_scaling.get("rope_type")
            if rope_scaling_type == "yarn":
                rope_config = YARNRoPEConfig(
                    precision=activation_precision,
                    base=self.rope_theta,
                    max_sequence_length=context_length or self.max_position_embeddings,
                    scaling_factor=float(self.rope_scaling["factor"]),
                    original_context_length=int(self.rope_scaling["original_max_position_embeddings"]),
                    beta_fast=float(self.rope_scaling["beta_fast"]),
                    beta_slow=float(self.rope_scaling["beta_slow"]),
                    truncate=bool(self.rope_scaling.get("truncate", True)),
                )
            elif rope_scaling_type == "llama3":
                rope_config = LlamaRoPEConfig(
                    precision=activation_precision,
                    base=self.rope_theta,
                    max_sequence_length=context_length or self.max_position_embeddings,
                    scaling_factor=float(self.rope_scaling["factor"]),
                    original_context_length=int(self.rope_scaling["original_max_position_embeddings"]),
                    low_frequency_factor=float(self.rope_scaling["low_freq_factor"]),
                    high_frequency_factor=float(self.rope_scaling["high_freq_factor"]),
                )
            elif rope_scaling_type == "linear":
                rope_config = LinearScalingRoPEConfig(
                    precision=activation_precision,
                    base=self.rope_theta,
                    max_sequence_length=context_length or self.max_position_embeddings,
                    scaling_factor=float(self.rope_scaling["factor"]),
                )
            else:
                raise ValueError(f"Unsupported rope_scaling type: {rope_scaling_type!r}")

        rmsnorm_config = NormalizationConfig(
            scale_precision=activation_precision,
            accumulation_precision=accumulation_precision,
            epsilon=self.rms_norm_eps,
            scale_offset=None,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=False,
        )

        # Linear layers
        if self.quantization_config is None:
            linear_config = FullPrecisionLinearConfig(
                precision=activation_precision,
            )
        else:
            linear_config = GroupQuantizedLinearConfig(
                group_size=self.quantization_config.group_size,
                weight_quantization_mode=QuantizationMode.from_num_bits(self.quantization_config.bits),
                activation_quantization_mode=None,
                activation_precision=activation_precision,
            )

        head_dim = self.head_dim if self.head_dim is not None else self.hidden_size // self.num_attention_heads

        qkv_clipping = (-self.clip_qkv, self.clip_qkv) if self.clip_qkv is not None else None

        mlp_config = DenseMLPConfig(
            linear_config=linear_config,
            activation=SiLU(),
            has_up_biases=self.mlp_bias,
            has_down_biases=self.mlp_bias,
            up_clipping=None,
            gate_clipping=None,
        )

        sliding_window_sizes = self._get_sliding_window_sizes()
        layer_configs = []
        for sliding_window_size in sliding_window_sizes:
            attention_config = AttentionConfig(
                qkv_projection_config=linear_config,
                out_projection_config=linear_config,
                query_norm_config=None,
                key_norm_config=None,
                logit_soft_cap=None,
                has_sinks=False,
                has_qkv_biases=self.attention_bias,
                has_out_biases=False,
                num_heads=self.num_attention_heads,
                num_groups=self.num_key_value_heads,
                head_dim=head_dim,
                is_causal=True,
                scale=None,
                sliding_window_size=sliding_window_size,
                qkv_clipping=qkv_clipping,
            )
            transformer_layer_config = TransformerLayerConfig(
                pre_mixer_norm_config=rmsnorm_config,
                mixer_config=attention_config,
                post_mixer_norm_config=None,
                pre_mlp_norm_config=rmsnorm_config,
                mlp_config=mlp_config,
                post_mlp_norm_config=None,
            )
            layer_configs.append(transformer_layer_config)

        transformer_config = TransformerConfig(
            global_rope_config=rope_config,
            local_rope_config=None,
            layer_configs=tuple(layer_configs),
            output_norm_config=rmsnorm_config,
            model_dim=self.hidden_size,
            hidden_dim=self.intermediate_size,
            context_length=context_length or self.max_position_embeddings,
        )

        return DecoderConfig(
            embedding_config=embedding_config,
            transformer_config=transformer_config,
            vocab_size=self.vocab_size,
        )
