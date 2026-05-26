from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from lalamo.modules.activations import Identity, SiLU
from lalamo.modules.decoder import DecoderConfig
from lalamo.modules.embedding import TiedEmbeddingConfig, UntiedEmbeddingConfig
from lalamo.modules.linear import LinearConfig
from lalamo.modules.mlp import DenseMLPConfig
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.token_mixers.convolutions import SeparableCausalConvConfig
from lalamo.modules.token_mixers.mamba import Mamba2Config
from lalamo.modules.transformer import TransformerConfig
from lalamo.modules.transformer_layer import TransformerLayerConfig

from .common import HuggingFaceLMConfig


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
    activation: Literal["identity"]
    bias: bool
    conv_bias: bool = True
    d_conv: int = 4


@dataclass(frozen=True)
class HFLlambaConfig(HuggingFaceLMConfig):
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
        context_length: int | None,  # noqa: ARG002
        metadata_dict: Mapping[str, str],  # noqa: ARG002
    ) -> DecoderConfig:
        if self.tie_embeddings:
            embedding_config = TiedEmbeddingConfig(
                input_scale=None,
                logit_soft_cap=None,
            )
        else:
            embedding_config = UntiedEmbeddingConfig(
                input_scale=None,
                logit_soft_cap=None,
            )

        rmsnorm_config = NormalizationConfig(
            epsilon=self.norm_epsilon,
            scale_offset=None,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=False,
        )

        linear_config = LinearConfig()

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
            conv_config=SeparableCausalConvConfig(
                has_biases=self.ssm_cfg.conv_bias,
            ),
            activation=activation,
            kernel_size=self.ssm_cfg.d_conv,
            num_heads=self.ssm_cfg.n_v_heads,
            num_groups=self.ssm_cfg.n_qk_heads,
            head_dim=head_dim,
            state_dim=self.ssm_cfg.d_state,
            expansion_factor=self.ssm_cfg.expand,
            has_in_biases=self.ssm_cfg.bias,
            has_out_biases=self.ssm_cfg.bias,
        )

        transformer_layer_config = TransformerLayerConfig(
            pre_mixer_norm_config=rmsnorm_config,
            mixer_config=mamba_config,
            post_mixer_norm_config=None,
            pre_mlp_norm_config=rmsnorm_config,
            mlp_config=mlp_config,
            post_mlp_norm_config=None,
        )
        transformer_config = TransformerConfig(
            layer_configs=(transformer_layer_config,) * self.n_layer,
            output_norm_config=rmsnorm_config,
            model_dim=self.d_model,
            hidden_dim=self.mlp_cfg.intermediate_size,
        )

        return DecoderConfig(
            embedding_config=embedding_config,
            transformer_config=transformer_config,
            vocab_size=self.vocab_size,
        )
