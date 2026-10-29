from dataclasses import dataclass
from functools import partial
from typing import Literal

from lalamo.model_import.loaders.binding import Binder
from lalamo.model_import.loaders.huggingface import load_mlp
from lalamo.modules.activations import SiLU
from lalamo.modules.decoder import DecoderConfig
from lalamo.modules.embedding import TiedEmbeddingConfig, UntiedEmbeddingConfig
from lalamo.modules.linear import LinearConfig
from lalamo.modules.mlp import DenseMLPConfig
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.rope import UnscaledRoPEConfig
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.modules.token_mixers.convolutions import SeparableCausalConvConfig
from lalamo.modules.token_mixers.short_conv import ShortConvConfig

from .common import HuggingFaceLMConfig, residual_decoder


@dataclass(frozen=True)
class RopeParameters:
    rope_theta: float
    rope_type: str = "default"


@dataclass(frozen=True)
class QuantizationConfig:
    group_size: int
    bits: int


@dataclass(frozen=True)
class HFLFM2Config(HuggingFaceLMConfig):
    architectures: list[Literal["Lfm2ForCausalLM"]]
    block_auto_adjust_ff_dim: bool
    block_dim: int
    block_ff_dim: int
    block_ffn_dim_multiplier: float
    block_mlp_init_scale: float
    block_multiple_of: int
    block_norm_eps: float
    block_out_init_scale: float
    block_use_swiglu: bool
    block_use_xavier_init: bool
    bos_token_id: int
    conv_L_cache: int  # noqa: N815
    conv_bias: bool
    conv_dim: int
    conv_use_xavier_init: bool
    eos_token_id: int
    hidden_size: int
    initializer_range: float
    max_position_embeddings: int
    model_type: Literal["lfm2"]
    norm_eps: float
    num_attention_heads: int
    num_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    pad_token_id: int
    transformers_version: str
    use_cache: bool
    use_pos_enc: bool
    vocab_size: int

    dtype: Literal["bfloat16", "float16", "float32"] | None = None
    torch_dtype: Literal["bfloat16", "float16", "float32"] | None = None
    intermediate_size: int | None = None
    conv_dim_out: int | None = None
    layer_types: list[Literal["conv", "full_attention"]] | None = None
    full_attn_idxs: list[int] | None = None
    tie_embedding: bool = True
    theta: float | None = None

    rope_theta: float | None = None
    rope_parameters: RopeParameters | None = None
    quantization: QuantizationConfig | None = None
    quantization_config: QuantizationConfig | None = None

    @property
    def resolved_rope_theta(self) -> float:
        flat = self.rope_theta
        nested = self.rope_parameters.rope_theta if self.rope_parameters else None

        if flat is None and nested is None:
            raise ValueError("Neither rope_theta nor rope_parameters is specified.")
        if flat is not None and nested is not None and flat != nested:
            raise ValueError(f"Conflicting rope_theta: {flat} (flat) vs {nested} (rope_parameters).")

        if nested is not None:
            return nested
        assert flat is not None
        return flat

    def build_decoder(self, at: Binder) -> DecoderConfig:
        assert self.num_attention_heads == self.num_heads
        head_dim = self.hidden_size // self.num_attention_heads

        if self.tie_embedding:
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
            base=self.resolved_rope_theta,
            max_sequence_length=self.max_position_embeddings,
            head_dim=head_dim,
        )

        linear_config = LinearConfig()

        block_norm_config = NormalizationConfig(
            epsilon=self.block_norm_eps,
            scale_offset=None,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=False,
        )

        attention_config = AttentionConfig(
            qkvg_projection_config=linear_config,
            out_projection_config=linear_config,
            query_norm_config=block_norm_config,
            key_norm_config=block_norm_config,
            num_heads=self.num_attention_heads,
            num_groups=self.num_key_value_heads,
            head_dim=head_dim,
            is_causal=True,
            scale=None,
            sliding_window_size=None,
            logit_soft_cap=None,
            has_sinks=False,
            has_qkvg_biases=False,
            has_out_biases=False,
            rope_config=rope_config,
        )

        short_conv_config = ShortConvConfig(
            in_projection_config=linear_config,
            conv_config=SeparableCausalConvConfig(has_biases=self.conv_bias),
            out_projection_config=linear_config,
            kernel_size=self.conv_L_cache,
        )

        if self.layer_types is not None:
            layer_types = self.layer_types
        elif self.full_attn_idxs is not None:
            layer_types = [
                "full_attention" if i in self.full_attn_idxs else "conv" for i in range(self.num_hidden_layers)
            ]
        else:
            raise RuntimeError("Either layer_types or full_attn_idxs must be present.")

        output_norm_config = NormalizationConfig(
            epsilon=self.norm_eps,
            scale_offset=None,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=False,
        )

        if not self.block_auto_adjust_ff_dim:
            hidden_dim = self.block_ff_dim if self.intermediate_size is None else self.intermediate_size
        else:
            hidden_dim_adjusted = self.block_ff_dim * self.block_ffn_dim_multiplier * (2 / 3)
            hidden_dim = int(
                (hidden_dim_adjusted + self.block_multiple_of - 1) // self.block_multiple_of * self.block_multiple_of,
            )

        mlp_config = DenseMLPConfig(
            linear_config=linear_config,
            activation=SiLU(),
            has_up_biases=False,
            has_down_biases=False,
            up_clipping=None,
            gate_clipping=None,
            hidden_dim=hidden_dim,
        )

        mixer_configs = {"conv": short_conv_config, "full_attention": attention_config}
        mixer_keys = {"conv": "conv", "full_attention": "self_attn"}
        load_feed_forward = partial(load_mlp, up_proj_key="w3", gate_proj_key="w1", down_proj_key="w2")
        return residual_decoder(
            at,
            embedding=at("model.embed_tokens", embedding_config),
            layers=[
                (
                    at(f"model.layers.{index}.{mixer_keys[layer_type]}", mixer_configs[layer_type]),
                    at(f"model.layers.{index}.feed_forward", mlp_config, load=load_feed_forward),
                )
                for index, layer_type in enumerate(layer_types)
            ],
            norm=block_norm_config,
            model_dim=self.hidden_size,
            vocab_size=self.vocab_size,
            mixer_norm="operator_norm",
            mlp_norm="ffn_norm",
            output_norm="embedding_norm",
            output_norm_config=output_norm_config,
        )
