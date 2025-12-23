from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Self

import equinox as eqx
import jax
from jax import numpy as jnp
from jax import vmap
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree, require_tree
from lalamo.modules import Activation
from lalamo.modules.normalization import NormalizationConfig
from lalamo.modules.transformer import (
    Normalization,
    Transformer,
    TransformerConfig,
    TransformerForwardPassConfig,
)
from lalamo.modules.utils import vmap_twice

from .common import ForwardPassMode, LalamoModule
from .embedding import EmbeddingBase, EmbeddingConfig
from .linear import LinearBase, LinearConfig
from .rope import PositionalEmbeddings
from .transformer_layer import TransformerLayerResult

__all__ = [
    "Classifier",
    "ClassifierActivationTrace",
    "ClassifierConfig",
    "ClassifierResult",
]


class PoolingType(StrEnum):
    CLS = "cls"
    MEAN = "mean"


@dataclass(frozen=True)
class PredictionHeadConfig:
    dense_config: LinearConfig
    activation: Activation
    normalization_config: NormalizationConfig
    readout_config: LinearConfig
    use_dense_bias: bool

    def empty(self, input_size: int, num_labels: int) -> "PredictionHead":
        dense_layer = self.dense_config.empty(
            input_dim=input_size,
            output_dims=(input_size,),
            has_biases=self.use_dense_bias,
        )
        norm = self.normalization_config.empty(input_size)
        readout = self.readout_config.empty(input_dim=input_size, output_dims=(num_labels,), has_biases=True)

        return PredictionHead(
            config=self,
            dense=dense_layer,
            activation=self.activation,
            norm=norm,
            readout=readout,
        )

    def random_init(self, input_size: int, num_labels: int, key: PRNGKeyArray) -> "PredictionHead":
        dense_key, readout_key = jax.random.split(key)
        dense_layer = self.dense_config.random_init(
            input_size, (input_size,), has_biases=self.use_dense_bias, key=dense_key,
        )
        norm = self.normalization_config.empty(input_size)
        readout = self.readout_config.random_init(
            input_dim=input_size,
            output_dims=(num_labels,),
            has_biases=True,
            key=readout_key,
        )

        return PredictionHead(
            config=self,
            dense=dense_layer,
            activation=self.activation,
            norm=norm,
            readout=readout,
        )


class PredictionHead(LalamoModule[PredictionHeadConfig]):
    dense: LinearBase
    activation: Activation
    norm: Normalization
    readout: LinearBase

    def __call__(self, inner_features: Float[Array, "batch channels"]) -> Float[Array, "batch logits"]:
        return vmap(self.call_unbatched)(inner_features)

    def call_unbatched(
        self,
        inner_features: Float[Array, " in_channels"],
    ) -> Float[Array, " logits"]:
        (dense_outs,) = self.dense(inner_features)
        dense_outs = self.activation(dense_outs)
        norm_outs = self.norm(dense_outs)
        (result,) = self.readout(norm_outs)
        return result

    @property
    def activation_precision(self) -> DTypeLike:
        return self.dense.activation_precision

    def export_weights(self) -> ParameterTree:
        result = dict(
            dense=self.dense.export_weights(),
            norm=self.norm.export_weights(),
            readout=self.readout.export_weights(),
        )
        return result

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            dense=self.dense.import_weights(require_tree(weights["dense"])),
            norm=self.norm.import_weights(require_tree(weights["norm"])),
            readout=self.readout.import_weights(require_tree(weights["readout"])),
        )


class ClassifierActivationTrace(eqx.Module):
    token_ids: Int[Array, "batch tokens"]
    token_positions: Int[Array, "batch tokens"]

    local_positional_embeddings: PositionalEmbeddings
    global_positional_embeddings: PositionalEmbeddings

    embedding_norm_output: Float[Array, "batch tokens channels"]
    layer_results: tuple[TransformerLayerResult, ...]
    output_norm: Float[Array, "batch tokens channels"]
    output_pooling: Float[Array, "batch channels"]
    logits: Float[Array, "batch logits"]

    def export(self) -> ParameterTree:
        result = dict(
            token_ids=self.token_ids,
            token_positions=self.token_positions,
            local_positional_embeddings=self.local_positional_embeddings.export(),
            global_positional_embeddings=self.global_positional_embeddings.export(),
            layer_results=[layer_result.export() for layer_result in self.layer_results],
            output_norm=self.output_norm,
            output_pooling=self.output_pooling,
            logits=self.logits,
        )
        return result


class ClassifierResult(eqx.Module):
    logits: Float[Array, "batch logits"]
    activation_trace: ClassifierActivationTrace | None = None

    def export(self) -> ParameterTree:
        result: dict[str, ParameterTree | Array] = dict(
            logits=self.logits,
        )
        if self.activation_trace is not None:
            result["activation_trace"] = self.activation_trace.export()
        return result


@dataclass(frozen=True)
class ClassifierConfig:
    embedding_config: EmbeddingConfig
    embedding_norm_config: NormalizationConfig
    transformer_config: TransformerConfig
    prediction_head_config: PredictionHeadConfig
    readout_config: LinearConfig

    vocab_size: int
    model_dim: int
    hidden_dim: int
    attention_scale: float | None
    num_layers: int
    context_length: int
    num_labels: int
    classifier_pooling: PoolingType

    output_labels: tuple[str, ...] | None

    def random_init(
        self,
        *,
        key: PRNGKeyArray,
    ) -> "Classifier":
        embedding_key, transformer_key, prediction_head_key = jax.random.split(key, num=3)
        embedding = self.embedding_config.random_init(
            vocab_size=self.vocab_size,
            model_dim=self.model_dim,
            key=embedding_key,
        )
        embedding_norm = self.embedding_norm_config.empty(self.model_dim)
        transformer = self.transformer_config.random_init(
            key=transformer_key,
        )
        prediction_head = self.prediction_head_config.random_init(
            input_size=self.hidden_dim,
            num_labels=self.num_labels,
            key=prediction_head_key,
        )
        return Classifier(
            self,
            embedding=embedding,
            embedding_norm=embedding_norm,
            transformer=transformer,
            prediction_head=prediction_head,
        )

    def empty(self) -> "Classifier":
        embedding = self.embedding_config.empty(
            vocab_size=self.vocab_size,
            model_dim=self.model_dim,
        )
        embedding_norm = self.embedding_norm_config.empty(self.model_dim)
        transformer = self.transformer_config.empty()
        prediction_head = self.prediction_head_config.empty(
            input_size=self.hidden_dim,
            num_labels=self.num_labels,
        )
        return Classifier(
            self,
            embedding=embedding,
            embedding_norm=embedding_norm,
            transformer=transformer,
            prediction_head=prediction_head,
        )


class Classifier(LalamoModule[ClassifierConfig]):
    embedding: EmbeddingBase
    embedding_norm: Normalization
    transformer: Transformer
    prediction_head: PredictionHead

    @property
    def activation_precision(self) -> DTypeLike:
        return self.embedding.activation_precision

    def __post_init__(self) -> None:
        if self.config.output_labels is not None and len(self.config.output_labels) != self.config.num_labels:
            raise ValueError("Number of output logits is different from provided list of labels")

    @eqx.filter_jit
    def __call__(
        self,
        token_ids: Int[Array, "batch tokens"],
        token_positions: Int[Array, "batch tokens"],
        return_activation_trace: bool = False,
        lengths_without_padding: Int[Array, " batch"] | None = None,
        forward_pass_mode: ForwardPassMode = ForwardPassMode.MULTI_TOKEN,
        forward_pass_config: TransformerForwardPassConfig | None = None,
    ) -> ClassifierResult:
        inner_features = self.embedding.embed(token_ids)
        normalized_embeddings = vmap_twice(self.embedding_norm)(inner_features)

        transformer_result = self.transformer(
            inner_features=normalized_embeddings,
            token_positions=token_positions,
            state=None,
            return_updated_state=False,
            return_layer_results=return_activation_trace,
            return_positional_embeddings=return_activation_trace,
            lengths_without_padding=lengths_without_padding,
            forward_pass_mode=forward_pass_mode,
            forward_pass_config=forward_pass_config,
        )

        if self.config.classifier_pooling == PoolingType.CLS:
            pooled_output = transformer_result.outputs[:, 0, :]
        elif self.config.classifier_pooling == PoolingType.MEAN:
            attention_mask = jnp.ones((*token_ids.shape, 1), dtype=transformer_result.outputs.dtype)
            pooled_output = (transformer_result.outputs * attention_mask).sum(axis=1) / attention_mask.sum(axis=1)
        else:
            raise TypeError(f"classifier_pooling of unknown type: {self.config.classifier_pooling}")

        logits = self.prediction_head(pooled_output)

        if return_activation_trace:
            assert transformer_result.layer_results is not None
            assert transformer_result.global_positional_embeddings is not None
            assert transformer_result.local_positional_embeddings is not None
            activation_trace = ClassifierActivationTrace(
                token_ids=token_ids,
                token_positions=token_positions,
                global_positional_embeddings=transformer_result.global_positional_embeddings,
                local_positional_embeddings=transformer_result.local_positional_embeddings,
                embedding_norm_output=normalized_embeddings,
                layer_results=tuple(transformer_result.layer_results),
                output_norm=transformer_result.outputs,
                output_pooling=pooled_output,
                logits=logits,
            )
        else:
            activation_trace = None

        return ClassifierResult(
            logits=logits,
            activation_trace=activation_trace,
        )

    def export_weights(self) -> ParameterTree:
        result = dict(
            embedding=self.embedding.export_weights(),
            embedding_norm=self.embedding_norm.export_weights(),
            transformer=self.transformer.export_weights(),
            prediction_head=self.prediction_head.export_weights(),
        )
        return result

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            embedding=self.embedding.import_weights(require_tree(weights["embedding"])),
            embedding_norm=self.embedding_norm.import_weights(require_tree(weights["embedding_norm"])),
            transformer=self.transformer.import_weights(require_tree(weights["transformer"])),
            prediction_head=self.prediction_head.import_weights(require_tree(weights["prediction_head"])),
        )
