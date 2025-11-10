from dataclasses import dataclass
from typing import Literal

from jaxtyping import DTypeLike

from lalamo.modules import (
    DecoderConfig,
    DecoderLayerConfig,
    DenseMLPConfig,
    FullPrecisionLinearConfig,
    Identity,
    Mamba2Config,
    RMSNormConfig,
    SiLU,
    TiedEmbeddingConfig,
    UntiedEmbeddingConfig,
    UpcastMode,
)

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
    conv_bias: bool = True
    d_conv: int = 4


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

        rmsnorm_config = RMSNormConfig(
            scale_precision=activation_precision,
            accumulation_precision=accumulation_precision,
            epsilon=self.norm_epsilon,
            scale_offset=None,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
        )

        linear_config = FullPrecisionLinearConfig(
            precision=activation_precision,
        )

        mlp_config = DenseMLPConfig(
            linear_config=linear_config,
            activation=SiLU(),
            has_up_biases=self.mlp_cfg.bias,
            has_down_biases=self.mlp_cfg.bias,
            up_clipping=None,
            gate_clipping=None,
        )

        inner_dim = self.ssm_cfg.expand * self.d_model
        head_dim = inner_dim // self.ssm_cfg.n_v_heads

        if self.ssm_cfg.activation == "identity":
            activation = Identity()
        elif self.ssm_cfg.activation == "silu":
            activation = SiLU()
        else:
            activation = SiLU()  # fallback

        mamba_config = Mamba2Config(
            in_projection_config=linear_config,
            out_projection_config=linear_config,
            num_value_heads=self.ssm_cfg.n_v_heads,
            num_groups=self.ssm_cfg.n_qk_heads,
            head_dim=head_dim,
            state_dim=self.ssm_cfg.d_state,
            conv_kernel_size=self.ssm_cfg.d_conv,
            expand=self.ssm_cfg.expand,
            activation=activation,
            has_in_biases=self.ssm_cfg.bias,
            has_out_biases=self.ssm_cfg.bias,
            has_conv_biases=self.ssm_cfg.conv_bias,
        )

        decoder_layer_config = DecoderLayerConfig(
            pre_mixer_norm_config=rmsnorm_config,
            mixer_config=mamba_config,
            post_mixer_norm_config=None,
            pre_mlp_norm_config=rmsnorm_config,
            mlp_config=mlp_config,
            post_mlp_norm_config=None,
        )

        return DecoderConfig(
            embedding_config=embedding_config,
            global_rope_config=None,
            local_rope_config=None,
            layer_configs=(decoder_layer_config,) * self.n_layer,
            output_norm_config=rmsnorm_config,
            vocab_size=self.vocab_size,
            model_dim=self.d_model,
            hidden_dim=self.mlp_cfg.intermediate_size,
            context_length=context_length or 4096,
        )
