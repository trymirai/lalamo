from dataclasses import dataclass
from enum import StrEnum

import equinox as eqx
from jax import numpy as jnp
from jax import vmap
from jaxtyping import Array, DTypeLike, Float, Int

from .activations import ActivationBase
from .common import ForwardPassMode, Initializer, LalamoConfig, LalamoModule
from .embedding import EmbeddingBase, EmbeddingConfigBase
from .linear import LinearBase, LinearConfigBase
from .normalization import NormalizationConfig
from .rope import PositionalEmbeddings
from .transformer import (
    Normalization,
    Transformer,
    TransformerConfig,
    TransformerForwardPassConfig,
)
from .transformer_layer import TransformerLayerResult
from .utils import vmap_twice

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
class PredictionHeadConfig(LalamoConfig):
    dense_config: LinearConfigBase
    activation: ActivationBase
    normalization_config: NormalizationConfig
    readout_config: LinearConfigBase
    use_dense_bias: bool

    def init(self, initializer: Initializer, input_size: int, num_labels: int) -> "PredictionHead":
        dense_layer = self.dense_config.init(
            initializer,
            input_dim=input_size,
            output_dims=(input_size,),
            has_biases=self.use_dense_bias,
        )
        norm = self.normalization_config.init(initializer, input_size)
        readout = self.readout_config.init(
            initializer,
            input_dim=input_size,
            output_dims=(num_labels,),
            has_biases=True,
        )

        return PredictionHead(
            dense=dense_layer,
            activation=self.activation,
            norm=norm,
            readout=readout,
        )


class PredictionHead(LalamoModule):
    dense: LinearBase
    activation: ActivationBase
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


class ClassifierResult(eqx.Module):
    logits: Float[Array, "batch logits"]
    activation_trace: ClassifierActivationTrace | None = None


@dataclass(frozen=True)
class ClassifierConfig(LalamoConfig):
    embedding_config: EmbeddingConfigBase
    embedding_norm_config: NormalizationConfig
    transformer_config: TransformerConfig
    prediction_head_config: PredictionHeadConfig
    readout_config: LinearConfigBase

    vocab_size: int
    model_dim: int
    hidden_dim: int
    attention_scale: float | None
    num_layers: int
    context_length: int
    num_labels: int
    classifier_pooling: PoolingType

    output_labels: tuple[str, ...] | None

    def init(self, initializer: Initializer) -> "Classifier":
        embedding = self.embedding_config.init(
            initializer,
            vocab_size=self.vocab_size,
            model_dim=self.model_dim,
        )
        embedding_norm = self.embedding_norm_config.init(initializer, self.model_dim)
        transformer = self.transformer_config.init(initializer)
        prediction_head = self.prediction_head_config.init(
            initializer,
            input_size=self.hidden_dim,
            num_labels=self.num_labels,
        )
        return Classifier(
            embedding=embedding,
            embedding_norm=embedding_norm,
            transformer=transformer,
            prediction_head=prediction_head,
            classifier_pooling=self.classifier_pooling,
            output_labels=self.output_labels,
            num_labels=self.num_labels,
        )


class Classifier(LalamoModule):
    embedding: EmbeddingBase
    embedding_norm: Normalization
    transformer: Transformer
    prediction_head: PredictionHead

    classifier_pooling: PoolingType = eqx.field(static=True)
    output_labels: tuple[str, ...] | None = eqx.field(static=True)
    num_labels: int = eqx.field(static=True)

    @property
    def activation_precision(self) -> DTypeLike:
        return self.embedding.activation_precision

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
            return_activation_trace=return_activation_trace,
            return_positional_embeddings=return_activation_trace,
            lengths_without_padding=lengths_without_padding,
            forward_pass_mode=forward_pass_mode,
            forward_pass_config=forward_pass_config,
        )

        if self.classifier_pooling == PoolingType.CLS:
            pooled_output = transformer_result.outputs[:, 0, :]
        elif self.classifier_pooling == PoolingType.MEAN:
            attention_mask = jnp.ones((*token_ids.shape, 1), dtype=transformer_result.outputs.dtype)
            pooled_output = (transformer_result.outputs * attention_mask).sum(axis=1) / attention_mask.sum(axis=1)
        else:
            raise TypeError(f"classifier_pooling of unknown type: {self.classifier_pooling}")

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
