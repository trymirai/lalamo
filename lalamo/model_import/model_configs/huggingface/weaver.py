from dataclasses import dataclass
from typing import ClassVar, Self

import cattrs

from lalamo.model_import.loaders.binding import Binder
from lalamo.modules.activations import SiLU
from lalamo.modules.flow import SequenceConfig, StepConfig, residual
from lalamo.modules.linear import LinearConfig
from lalamo.modules.mlp import DenseMLPConfig
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.rope import UnscaledRoPEConfig
from lalamo.modules.speculators.weaver import WeaverConfig
from lalamo.modules.token_mixers.attention import AttentionConfig

__all__ = ["HFWeaverConfig"]


@dataclass(frozen=True)
class HFWeaverConfig:
    _converter: ClassVar[cattrs.Converter] = cattrs.Converter()

    d_model: int
    d_embed: int
    d_rank: int
    num_layers: int
    num_heads: int
    mlp_channels: int
    max_depth: int
    candidate_pool_size: int
    rope_base: float

    @classmethod
    def from_dict(cls, config: dict[str, object]) -> Self:
        config = dict(config)
        config["max_depth"] = config.pop("K")
        config["mlp_channels"] = config.pop("mlp_dim")
        return cls._converter.structure(config, cls)

    def build_weaver(self, at: Binder) -> WeaverConfig:
        if self.d_rank % self.num_heads:
            raise ValueError("Weaver rank must be divisible by the number of heads.")
        linear = LinearConfig()
        norm = NormalizationConfig(
            epsilon=1e-6,
            scale_offset=None,
            upcast_mode=UpcastMode.FULL_LAYER,
            subtract_mean=False,
            has_biases=True,
        )
        rope = UnscaledRoPEConfig(
            base=self.rope_base,
            max_sequence_length=self.max_depth + 1,
            head_dim=self.d_rank // self.num_heads,
        )
        steps: list[StepConfig] = []
        for index in range(self.num_layers):
            path = f"blocks.{index}"
            steps += [
                residual(
                    at(f"{path}.norm_attn", norm),
                    at(
                        path,
                        AttentionConfig(
                            qkvg_projection_config=linear,
                            out_projection_config=linear,
                            query_norm_config=None,
                            key_norm_config=None,
                            num_heads=self.num_heads,
                            num_groups=self.num_heads,
                            head_dim=rope.head_dim,
                            is_causal=True,
                            scale=None,
                            sliding_window_size=None,
                            logit_soft_cap=None,
                            has_sinks=False,
                            has_qkvg_biases=False,
                            has_out_biases=False,
                            rope_config=rope,
                        ),
                    ),
                ),
                residual(
                    at(f"{path}.norm_mlp", norm),
                    at(
                        path,
                        DenseMLPConfig(
                            linear_config=linear,
                            activation=SiLU(),
                            has_up_biases=True,
                            has_down_biases=True,
                            gate_clipping=None,
                            up_clipping=None,
                            hidden_dim=self.mlp_channels,
                        ),
                    ),
                ),
            ]
        steps.append(at("out_norm", norm))
        return WeaverConfig(
            model_dim=self.d_rank,
            target_model_dim=self.d_model,
            target_embedding_dim=self.d_embed,
            max_depth=self.max_depth,
            candidate_pool_size=self.candidate_pool_size,
            linear_config=linear,
            norm_config=norm,
            program=SequenceConfig(tuple(steps)),
        )
