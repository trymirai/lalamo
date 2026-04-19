from dataclasses import dataclass
from enum import StrEnum

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Int, Key

from lalamo.exportable import Exportable
from lalamo.initializer import Initializer
from lalamo.module import ForwardPassMode, LalamoConfig, LalamoModule

from .activations import Activation
from .embedding import EmbeddingBase, EmbeddingConfig
from .linear import Linear, LinearConfig
from .normalization import Normalization, NormalizationConfig
from .rope import PositionalEmbeddings
from .transformer import Transformer, TransformerConfig, TransformerForwardPassConfig
from .transformer_layer import TransformerLayerResult
from .utils import vmap_twice, vmap_twice_with_dequant_key, vmap_with_dequant_key

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
    dense_config: LinearConfig
    activation: Activation
    normalization_config: NormalizationConfig
    readout_config: LinearConfig
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
    readout: Linear

    def __call__(
        self,
        inner_features: Float[Array, "batch channels"],
        *,
        dequant_key: Key[Array, ""],
    ) -> Float[Array, "batch logits"]:
        return vmap_with_dequant_key(self.call_unbatched, inner_features, dequant_key=dequant_key)

    def call_unbatched(
        self,
        inner_features: Float[Array, " in_channels"],
        *,
        dequant_key: Key[Array, ""],
    ) -> Float[Array, " logits"]:
        dense_dequant_key, readout_dequant_key = jax.random.split(dequant_key)
        (dense_outs,) = self.dense(inner_features, dequant_key=dense_dequant_key)
        dense_outs = self.activation(dense_outs)
        norm_outs = self.norm(dense_outs)
        (result,) = self.readout(norm_outs, dequant_key=readout_dequant_key)
        return result

    def export_weights(self) -> ParameterTree:
        result = dict(
            dense=self.dense.export_weights(),
            norm=self.norm.export_weights(),
            readout=self.readout.export_weights(),
        )
        return result

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        weights = require_mapping(weights)
        return replace(
            self,
            dense=self.dense.import_weights(require_tree(weights["dense"])),
            norm=self.norm.import_weights(require_tree(weights["norm"])),
            readout=self.readout.import_weights(require_tree(weights["readout"])),
        )


class ClassifierActivationTrace(Exportable, eqx.Module):
    token_ids: Int[Array, "batch tokens"]
    token_positions: Int[Array, "batch tokens"]

    rope_embeddings: tuple[PositionalEmbeddings, ...] | None

    embedding_norm_output: Float[Array, "batch tokens channels"]
    layer_results: tuple[TransformerLayerResult, ...]
    output_norm: Float[Array, "batch tokens channels"]
    output_pooling: Float[Array, "batch channels"]
    logits: Float[Array, "batch logits"]


class ClassifierResult(Exportable, eqx.Module):
    logits: Float[Array, "batch logits"]
    activation_trace: ClassifierActivationTrace | None = None


@dataclass(frozen=True)
class ClassifierConfig(LalamoConfig):
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
            config=self,
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

    @eqx.filter_jit
    def __call__(
        self,
        token_ids: Int[Array, "batch tokens"],
        token_positions: Int[Array, "batch tokens"],
        return_activation_trace: bool = False,
        lengths_without_padding: Int[Array, " batch"] | None = None,
        forward_pass_mode: ForwardPassMode = ForwardPassMode.MULTI_TOKEN,
        forward_pass_config: TransformerForwardPassConfig | None = None,
        *,
        dequant_key: Key[Array, ""],
    ) -> ClassifierResult:
        if forward_pass_config is None:
            forward_pass_config = TransformerForwardPassConfig()
        embedding_dequant_key, transformer_dequant_key, prediction_head_dequant_key = jax.random.split(
            dequant_key,
            3,
        )
        inner_features = vmap_twice_with_dequant_key(
            self.embedding.embed,
            token_ids,
            dequant_key=embedding_dequant_key,
        )
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
            dequant_key=transformer_dequant_key,
        )

        if self.config.classifier_pooling == PoolingType.CLS:
            pooled_output = transformer_result.outputs[:, 0, :]
        elif self.config.classifier_pooling == PoolingType.MEAN:
            attention_mask = jnp.ones((*token_ids.shape, 1), dtype=transformer_result.outputs.dtype)
            pooled_output = (transformer_result.outputs * attention_mask).sum(axis=1) / attention_mask.sum(axis=1)
        else:
            raise TypeError(f"classifier_pooling of unknown type: {self.config.classifier_pooling}")

        logits = self.prediction_head(pooled_output, dequant_key=prediction_head_dequant_key)

        if return_activation_trace:
            assert transformer_result.layer_results is not None
            activation_trace = ClassifierActivationTrace(
                token_ids=token_ids,
                token_positions=token_positions,
                rope_embeddings=transformer_result.rope_embeddings,
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
