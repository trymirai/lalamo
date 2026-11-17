from dataclasses import dataclass
from math import isfinite
from typing import NamedTuple, Self

from jaxtyping import Array, Float

from lalamo.initializer import Initializer
from lalamo.module import LalamoConfig, LalamoModule
from lalamo.modules.linear import Linear, LinearConfig
from lalamo.modules.normalization import Normalization, NormalizationConfig

__all__ = [
    "ConfidenceHead",
    "ConfidenceHeadConfig",
    "ConfidenceHeadTreeAttentionConfig",
]


def _init_biased_linear(initializer: Initializer, input_dim: int, output_dim: int) -> Linear:
    return LinearConfig().init(initializer, input_dim, (output_dim,), has_biases=True)


@dataclass(frozen=True)
class ConfidenceHeadTreeAttentionConfig(LalamoConfig):
    model_dim: int
    num_layers: int
    num_heads: int
    hidden_dim: int

    def __post_init__(self) -> None:
        if min(self.model_dim, self.num_heads, self.hidden_dim) <= 0 or self.num_layers < 0:
            raise ValueError("Confidence-head tree dimensions must be positive and num_layers nonnegative")
        if self.model_dim % self.num_heads:
            raise ValueError("Confidence-head tree model_dim must be divisible by num_heads")

    def init(
        self,
        initializer: Initializer,
        *,
        head_model_dim: int,
        max_depth: int,
        norm_config: NormalizationConfig,
    ) -> "ConfidenceHeadTree":
        return ConfidenceHeadTree(
            node_in=_init_biased_linear(initializer, head_model_dim, self.model_dim),
            child_in=_init_biased_linear(initializer, head_model_dim + 4, self.model_dim),
            depth_embedding=initializer.zeros((max_depth + 1, self.model_dim)),
            relation_bias=initializer.zeros((self.num_heads, 6)),
            layers=tuple(
                ConfidenceHeadTreeLayer(
                    attention_norm=norm_config.init(initializer, self.model_dim),
                    qkv=_init_biased_linear(initializer, self.model_dim, 3 * self.model_dim),
                    out=_init_biased_linear(initializer, self.model_dim, self.model_dim),
                    mlp_norm=norm_config.init(initializer, self.model_dim),
                    up=_init_biased_linear(initializer, self.model_dim, self.hidden_dim),
                    down=_init_biased_linear(initializer, self.hidden_dim, self.model_dim),
                )
                for _ in range(self.num_layers)
            ),
            final_norm=norm_config.init(initializer, self.model_dim),
            couple_node=_init_biased_linear(initializer, self.model_dim, head_model_dim),
            couple_child=_init_biased_linear(initializer, self.model_dim, head_model_dim),
        )


# These named groups preserve Uzu's checkpoint paths; its fused kernels read the affine tensors without spec metadata.
class ConfidenceHeadAffine(NamedTuple):
    weights: Float[Array, "out_channels in_channels"]
    biases: Float[Array, " out_channels"]

    @classmethod
    def init(cls, initializer: Initializer, input_dim: int, output_dim: int) -> Self:
        return cls(
            weights=initializer.zeros((output_dim, input_dim)),
            biases=initializer.zeros((output_dim,)),
        )


class ConfidenceHeadPool(NamedTuple):
    entry: ConfidenceHeadAffine
    hidden: ConfidenceHeadAffine
    out: ConfidenceHeadAffine


class ConfidenceHeadSlot(NamedTuple):
    stats: Float[Array, "2 4"]


class ConfidenceHeadEdge(NamedTuple):
    input_norm: Float[Array, "2 inputs"]
    entry: ConfidenceHeadAffine
    block_norm: Float[Array, "2 channels"]
    up: ConfidenceHeadAffine
    down: ConfidenceHeadAffine
    final_norm: Float[Array, "2 channels"]
    readout: ConfidenceHeadAffine


class ConfidenceHeadTreeLayer(NamedTuple):
    attention_norm: Normalization
    qkv: Linear
    out: Linear
    mlp_norm: Normalization
    up: Linear
    down: Linear


class ConfidenceHeadTree(NamedTuple):
    node_in: Linear
    child_in: Linear
    depth_embedding: Float[Array, "depth channels"]
    relation_bias: Float[Array, "heads 6"]
    layers: tuple[ConfidenceHeadTreeLayer, ...]
    final_norm: Normalization
    couple_node: Linear
    couple_child: Linear


@dataclass(frozen=True)
class ConfidenceHeadConfig(LalamoConfig):
    model_dim: int
    pool_hidden_dim: int
    edge_hidden_dim: int
    logit_cap: float
    max_depth: int
    norm_config: NormalizationConfig
    tree_attention_config: ConfidenceHeadTreeAttentionConfig | None

    def __post_init__(self) -> None:
        if min(self.model_dim, self.pool_hidden_dim, self.edge_hidden_dim) <= 0 or self.max_depth < 0:
            raise ValueError("Confidence-head dimensions must be positive and max_depth nonnegative")
        if not isfinite(self.logit_cap) or self.logit_cap <= 0:
            raise ValueError("Confidence-head logit_cap must be finite and positive")

    def init(self, initializer: Initializer, target_model_dim: int) -> "ConfidenceHead":
        tree = None
        if self.tree_attention_config is not None:
            tree = self.tree_attention_config.init(
                initializer,
                head_model_dim=self.model_dim,
                max_depth=self.max_depth,
                norm_config=self.norm_config,
            )
        edge_input_dim = 3 * self.model_dim + 4
        return ConfidenceHead(
            config=self,
            sharding_config=initializer.sharding_config,
            draft_norm=self.norm_config.init(initializer, target_model_dim),
            draft_projection=_init_biased_linear(initializer, target_model_dim, self.model_dim),
            token_norm=self.norm_config.init(initializer, target_model_dim),
            token_projection=_init_biased_linear(initializer, target_model_dim, self.model_dim),
            pool_set=ConfidenceHeadPool(
                entry=ConfidenceHeadAffine.init(initializer, 3, self.pool_hidden_dim),
                hidden=ConfidenceHeadAffine.init(initializer, self.pool_hidden_dim, self.pool_hidden_dim),
                out=ConfidenceHeadAffine.init(initializer, 2 * self.pool_hidden_dim, self.model_dim),
            ),
            slot=ConfidenceHeadSlot(initializer.zeros((2, 4))),
            edge=ConfidenceHeadEdge(
                input_norm=initializer.zeros((2, edge_input_dim)),
                entry=ConfidenceHeadAffine.init(initializer, edge_input_dim, self.model_dim),
                block_norm=initializer.zeros((2, self.model_dim)),
                up=ConfidenceHeadAffine.init(initializer, self.model_dim, self.edge_hidden_dim),
                down=ConfidenceHeadAffine.init(initializer, self.edge_hidden_dim, self.model_dim),
                final_norm=initializer.zeros((2, self.model_dim)),
                readout=ConfidenceHeadAffine.init(initializer, self.model_dim, 3),
            ),
            tree=tree,
        )


class ConfidenceHead(LalamoModule[ConfidenceHeadConfig]):
    draft_norm: Normalization
    draft_projection: Linear
    token_norm: Normalization
    token_projection: Linear
    pool_set: ConfidenceHeadPool
    slot: ConfidenceHeadSlot
    edge: ConfidenceHeadEdge
    tree: ConfidenceHeadTree | None
