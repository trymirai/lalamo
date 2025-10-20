from dataclasses import dataclass
from typing import Literal

from jaxtyping import DTypeLike

from lalamo.modules import (
    DecoderConfig,
    DecoderLayerConfig,
    DenseMLPConfig,
    FullPrecisionLinearConfig,
    RMSNormConfig,
    TiedEmbeddingConfig,
    UntiedEmbeddingConfig,
    UpcastMode,
)
from lalamo.modules.activations import SiLU

from .common import HuggingFaceConfig


@dataclass(frozen=True)
class HFLlambaMlpConfig:
    intermediate_size: int
    bias: bool
    act_fn: Literal["silu"]


@dataclass(frozen=True)
class HFLlambaSsmConfig:
    d_state: int
    n_v_heads: int
    n_qk_heads: int
    expand: int
    chunk_size: int
    activation: Literal["identity"]
    bias: bool


@dataclass(frozen=True)
class HFLlambaConfig(HuggingFaceConfig):
    model_type: Literal["llamba"]
    vocab_size: int
    tie_embeddings: bool
    pad_vocab_size_multiple: int
    lm_head_bias: bool
    d_model: int
    n_layer: int
    resid_dropout: float
    norm_epsilon: float
    mlp_cfg: HFLlambaMlpConfig
    ssm_cfg: HFLlambaSsmConfig

    @property
    def eos_token_ids(self) -> list[int]:
        return [128001, 128008, 128009]

    def to_decoder_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,
    ) -> DecoderConfig:
        if self.tie_embeddings:
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

        # seem to be identical to llama, but verify
        rmsnorm_config = RMSNormConfig(
            scale_precision=activation_precision,
            accumulation_precision=accumulation_precision,
            epsilon=1e-6,
            scale_offset=None,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
        )
        linear_config = FullPrecisionLinearConfig(
            precision=activation_precision,
        )
        mlp_config = DenseMLPConfig(
            linear_config=linear_config,
            activation=SiLU(),
            has_up_biases=False,
            has_down_biases=False,
            up_clipping=None,
            gate_clipping=None,
        )

        decoder_layer_config = DecoderLayerConfig(
            pre_attention_norm_config=rmsnorm_config,
            attention_config=None,
            post_attention_norm_config=None,
            pre_mlp_norm_config=rmsnorm_config,
            mlp_config=mlp_config,
            post_mlp_norm_config=None,
        )

        return DecoderConfig(
            embedding_config=embedding_config,
            global_rope_config=None,
            local_rope_config=None,
            layer_config=decoder_layer_config,
            output_norm_config=rmsnorm_config,
            vocab_size=self.vocab_size,
            model_dim=self.d_model,
            hidden_dim=self.mlp_cfg.intermediate_size,
            num_heads=None,
            num_groups=None,
            head_dim=None,
            attention_scale=None,
            num_layers=self.n_layer,
            sliding_window_sizes=None,
            context_length=context_length or 4096,  # mamba doesn't have context length.
        )
