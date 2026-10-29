from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Literal

import equinox as eqx
from jaxtyping import Array

from lalamo.model import Model
from lalamo.model_import.loaders import huggingface_loader, load_huggingface_decoder
from lalamo.model_import.loaders.binding import Binder
from lalamo.model_import.loaders.huggingface import load_mlp, load_untied_embedding
from lalamo.models import LanguageModel
from lalamo.modules.activations import Identity, SiLU
from lalamo.modules.decoder import DecoderConfig
from lalamo.modules.embedding import TiedEmbeddingConfig, UntiedEmbeddingConfig
from lalamo.modules.flow import EmitConfig, SequenceConfig, StepConfig, residual
from lalamo.modules.linear import LinearConfig
from lalamo.modules.mlp import DenseMLPConfig
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.token_mixers.convolutions import SeparableCausalConvConfig
from lalamo.modules.token_mixers.mamba import Mamba2Config
from lalamo.utils.parameter_path import ParameterPath
from lalamo.weight_matrix import CompressionImplementation

from .common import HuggingFaceLMConfig


class LlambaCheckpointLayout(Enum):
    CARTESIA = "cartesia"
    # MLX conversions store the mixer and the MLP of every block as separate consecutive layers.
    MLX = "mlx"

    @staticmethod
    def of(weights: Mapping[str, Array]) -> "LlambaCheckpointLayout":
        if any(key.startswith("embedding.encoder.") for key in weights):
            return LlambaCheckpointLayout.MLX
        return LlambaCheckpointLayout.CARTESIA


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

    def build_decoder(
        self,
        at: Binder,
        layout: LlambaCheckpointLayout = LlambaCheckpointLayout.CARTESIA,
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
            hidden_dim=self.mlp_cfg.intermediate_size,
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
            has_in_biases=self.ssm_cfg.bias,
            has_out_biases=self.ssm_cfg.bias,
        )

        load_mlx_mlp = partial(load_mlp, up_proj_key="gate_proj", gate_proj_key="in_proj", down_proj_key="out_proj")
        steps: list[StepConfig] = []
        taps: list[EmitConfig] = []
        for index in range(self.n_layer):
            tap = EmitConfig(f"hidden_states.{index}")
            match layout:
                case LlambaCheckpointLayout.CARTESIA:
                    path = ParameterPath("backbone.layers") / index
                    mixer_steps = (at(path / "input_layernorm", rmsnorm_config), at(path / "mixer", mamba_config))
                    mlp_steps = (at(path / "post_attention_layernorm", rmsnorm_config), at(path / "mlp", mlp_config))
                case LlambaCheckpointLayout.MLX:
                    mixer_path = ParameterPath("model.layers") / (2 * index)
                    mlp_path = ParameterPath("model.layers") / (2 * index + 1)
                    mixer_steps = (at(mixer_path / "norm", rmsnorm_config), at(mixer_path / "layer", mamba_config))
                    mlp_steps = (
                        at(mlp_path / "norm", rmsnorm_config),
                        at(mlp_path / "layer", mlp_config, load=load_mlx_mlp),
                    )
            steps += [residual(*mixer_steps), residual(*mlp_steps), tap]
            taps.append(tap)

        match layout:
            case LlambaCheckpointLayout.CARTESIA:
                embedding = at("backbone.embedding", embedding_config)
                output_norm = at("backbone.final_layernorm", rmsnorm_config)
            case LlambaCheckpointLayout.MLX:
                embedding = at(
                    "embedding.encoder",
                    embedding_config,
                    load=partial(load_untied_embedding, lm_head_path=ParameterPath("head.linear")),
                )
                output_norm = at("model.norm", rmsnorm_config)

        return DecoderConfig(
            embedding,
            SequenceConfig((*steps, output_norm)),
            self.d_model,
            self.vocab_size,
            tuple(taps),
        )

    def _load_weights(
        self,
        model: Model,
        weights_dict: Mapping[str, Array],
        *,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
    ) -> Model:
        assert isinstance(model, LanguageModel)
        binder = Binder(huggingface_loader)
        config = self.build_decoder(binder, LlambaCheckpointLayout.of(weights_dict))
        decoder = load_huggingface_decoder(model.decoder, config, binder, weights_dict, implementation=implementation)
        return eqx.tree_at(lambda m: (m.decoder,), model, (decoder,))
