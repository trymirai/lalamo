from dataclasses import dataclass, replace
from typing import Self

from jax import Array

from lalamo.common import DTypeLike, ParameterTree
from lalamo.modules import Classifier, ClassifierConfig, LalamoModule


class RoutingConfig:
    pass


@dataclass(frozen=True)
class RouterModelConfig:
    classifier_config: ClassifierConfig


class RouterModel(LalamoModule[RouterModelConfig]):
    classifier: Classifier

    @property
    def activation_precision(self) -> DTypeLike:
        return self.classifier.activation_precision

    def export_weights(self) -> ParameterTree:
        return self.classifier.export_weights()

    def import_weights(
        self,
        weights: ParameterTree[Array],
    ) -> Self:
        return replace(
            self,
            classifier=self.classifier.import_weights(weights),
        )
