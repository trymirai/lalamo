from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Self

from jax.lax import DotAlgorithmPreset

from lalamo.module import ForwardPassMode
from lalamo.weight_matrix import GradientEstimator

from .embedding import EmbeddingForwardPassConfig
from .flow import FlowForwardPassConfig

__all__ = ["ModelForwardPassConfig"]


@dataclass(frozen=True)
class ModelForwardPassConfig:
    embedding_forward_pass_config: EmbeddingForwardPassConfig = dataclass_field(
        default_factory=EmbeddingForwardPassConfig,
    )
    flow_forward_pass_config: FlowForwardPassConfig = dataclass_field(default_factory=FlowForwardPassConfig)

    @classmethod
    def for_tracer_tests(cls) -> Self:
        return cls(
            embedding_forward_pass_config=EmbeddingForwardPassConfig.for_tracer_tests(),
            flow_forward_pass_config=FlowForwardPassConfig.for_tracer_tests(),
        )

    @classmethod
    def for_inference(
        cls,
        mode: ForwardPassMode = ForwardPassMode.MULTI_TOKEN,
        precision: DotAlgorithmPreset = DotAlgorithmPreset.DEFAULT,
    ) -> Self:
        return cls(
            embedding_forward_pass_config=EmbeddingForwardPassConfig.for_inference(precision),
            flow_forward_pass_config=FlowForwardPassConfig.for_inference(mode, precision),
        )

    @classmethod
    def for_training(
        cls,
        gradient_estimator: GradientEstimator = GradientEstimator.DETERMINISTIC_ROUNDING,
        precision: DotAlgorithmPreset = DotAlgorithmPreset.DEFAULT,
    ) -> Self:
        return cls(
            embedding_forward_pass_config=EmbeddingForwardPassConfig.for_training(gradient_estimator, precision),
            flow_forward_pass_config=FlowForwardPassConfig.for_training(gradient_estimator, precision),
        )
