from dataclasses import dataclass
from typing import ClassVar, Self

import cattrs

from lalamo.modules import LinearConfig, NormalizationConfig, WeaverConfig
from lalamo.modules.normalization import UpcastMode

__all__ = [
    "HFWeaverConfig",
]


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

    @classmethod
    def from_dict(cls, config: dict[str, object]) -> Self:
        config = dict(config)
        config["max_depth"] = config.pop("K")
        config["mlp_channels"] = config.pop("mlp_dim")
        return cls._converter.structure(config, cls)

    def to_weaver_config(self) -> WeaverConfig:
        return WeaverConfig(
            d_model=self.d_model,
            d_embed=self.d_embed,
            d_rank=self.d_rank,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            mlp_channels=self.mlp_channels,
            max_depth=self.max_depth,
            candidate_pool_size=self.candidate_pool_size,
            linear_config=LinearConfig(),
            norm_config=NormalizationConfig(
                epsilon=1e-6,
                scale_offset=None,
                upcast_mode=UpcastMode.FULL_LAYER,
                subtract_mean=False,
                has_biases=True,
            ),
        )
