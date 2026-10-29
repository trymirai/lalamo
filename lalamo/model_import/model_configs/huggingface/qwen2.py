from dataclasses import dataclass, replace
from typing import Literal

from lalamo.model_import.loaders.binding import Binder
from lalamo.modules.activations import SiLU
from lalamo.modules.decoder import DecoderConfig
from lalamo.modules.embedding import TiedEmbeddingConfig, UntiedEmbeddingConfig
from lalamo.modules.linear import LinearConfig
from lalamo.modules.mlp import DenseMLPConfig
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.rope import UnscaledRoPEConfig
from lalamo.modules.token_mixers.attention import AttentionConfig

from .common import AWQQuantizationConfig, GPTQQuantizationConfig, HuggingFaceLMConfig, residual_decoder

__all__ = ["HFQwen2Config"]


@dataclass(frozen=True)
class HFQwen2Config(HuggingFaceLMConfig):
    torch_dtype: Literal["bfloat16", "float16", "float32"]
    architectures: list[Literal["Qwen2ForCausalLM"]]
    attention_dropout: float
    bos_token_id: int | list[int]
    eos_token_id: int | list[int]
    hidden_act: Literal["silu"]
    hidden_size: int
    initializer_range: float
    intermediate_size: int
    max_position_embeddings: int
    max_window_layers: int
    model_type: Literal["qwen2"]
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    sliding_window: int
    tie_word_embeddings: bool
    transformers_version: str
    use_cache: bool
    use_sliding_window: bool
    vocab_size: int

    quantization_config: AWQQuantizationConfig | GPTQQuantizationConfig | None = None

    def _get_sliding_window_sizes(self) -> list[int | None]:
        if not self.use_sliding_window:
            return [None] * self.num_hidden_layers

        sliding_window_sizes = []
        for i in range(self.num_hidden_layers):
            if i < self.max_window_layers:
                sliding_window_sizes.append(self.sliding_window)
            else:
                sliding_window_sizes.append(None)
        return sliding_window_sizes

    def build_decoder(self, at: Binder) -> DecoderConfig:
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
        head_dim = self.hidden_size // self.num_attention_heads
        rope_config = UnscaledRoPEConfig(
            base=self.rope_theta,
            max_sequence_length=self.max_position_embeddings,
            head_dim=head_dim,
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
            hidden_dim=self.intermediate_size,
        )

        attention_config = AttentionConfig(
            qkvg_projection_config=linear_config,
            out_projection_config=linear_config,
            query_norm_config=None,
            key_norm_config=None,
            logit_soft_cap=None,
            has_sinks=False,
            has_qkvg_biases=True,
            has_out_biases=False,
            num_heads=self.num_attention_heads,
            num_groups=self.num_key_value_heads,
            head_dim=head_dim,
            is_causal=True,
            scale=None,
            sliding_window_size=None,
            rope_config=rope_config,
        )
        windows = self._get_sliding_window_sizes()
        return residual_decoder(
            at,
            embedding=at("model.embed_tokens", embedding_config),
            layers=[
                (
                    at(f"model.layers.{index}.self_attn", replace(attention_config, sliding_window_size=window)),
                    at(f"model.layers.{index}.mlp", mlp_config),
                )
                for index, window in enumerate(windows)
            ],
            norm=rmsnorm_config,
            model_dim=self.hidden_size,
            vocab_size=self.vocab_size,
        )
