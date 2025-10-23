from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Self

import equinox as eqx
import jax
from jax import vmap
from jax.random import PRNGKey
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray
import jax.numpy as jnp

from lalamo.common import ParameterTree
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.transformer import TransformerConfig, Transformer, TransformerForwardPassConfig, Normalization

from .activations import Activation, SiLU, GELU
from .common import LalamoModule, ForwardPassMode
from .decoder_layer import TransformerLayerResult
from .linear import FullPrecisionLinearConfig, LinearBase, LinearConfig
from .embedding import EmbeddingBase, EmbeddingConfig
from .rope import PositionalEmbeddings
from .linear import FullPrecisionLinear

__all__ = [
    "Classifier",
    "ClassifierActivationTrace",
    "ClassifierConfig",
    "ClassifierResult",
]

def activation_from_str(activation: str) -> Activation:
    supported_activations = {
        "silu" : SiLU,
        "gely" : GELU,
    }
    if activation in supported_activations:
        return supported_activations[activation]()
    else:
        raise ValueError("Only activations from the following list are supported by Classifier: {}".format(supported_activations.keys()))

@dataclass(frozen=True)
class PredictionHeadConfig:
    input_size: int
    output_size: int
    use_bias: bool
    activation: Activation
    norm_size: int
    norm_eps: float
    use_norm_bias: bool

    def empty(self, activation_precision: DTypeLike) -> "PredictionHead":
        dense_config = FullPrecisionLinearConfig(activation_precision)
        dense_layer = dense_config.empty(self.input_size, (self.output_size,), self.use_bias)
        activation = self.activation
        norm_config = NormalizationConfig(
            scale_precision=activation_precision,
            accumulation_precision=jnp.float32,
            epsilon=self.norm_eps,
            scale_offset=0.0,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=True
        )
        norm = norm_config.empty(self.norm_size)

        return PredictionHead(config=self, dense=dense_layer, activation=activation, norm=norm)

    def random_init(self, activation_precision: DTypeLike, key: PRNGKeyArray) -> "PredictionHead":
        dense_key, norm_key = jax.random.split(key)
        dense_config = FullPrecisionLinearConfig(activation_precision)
        dense_layer = dense_config.random_init(self.input_size, (self.output_size,), self.use_bias, key=dense_key)
        activation = self.activation
        norm_config = NormalizationConfig(
            scale_precision=activation_precision,
            accumulation_precision=jnp.float32,
            epsilon = self.norm_eps,
            scale_offset = 0.0,
            upcast_mode = UpcastMode.ONLY_NORMALIZATION,
            subtract_mean = True
        )
        norm = norm_config.empty(self.input_size)

        return PredictionHead(config=self, dense=dense_layer, activation=activation, norm=norm)


class PredictionHead(LalamoModule[PredictionHeadConfig]):
    dense: LinearBase
    activation: Activation
    norm: Normalization

    def __call__(self, inner_features: Float[Array, " in_channels"])->Float[Array, " out_channels"]:
        dense_out = self.dense(inner_features)
        assert len(dense_out) == 1, "PredictionHead, expecting dense output to have only 1 tensor"
        return self.norm(self.activation(dense_out[0]))

    @property
    def activation_precision(self) -> DTypeLike:
        return self.activation_precision
    
    def export_weights(self) -> ParameterTree:
        result = dict(
            dense=self.dense.export_weights(),
            norm=self.norm.export_weights()
        )
        return result

    def import_weights(
        self,
        weights: ParameterTree[Array],
    ) -> Self:
        assert isinstance(weights, Mapping)
        assert isinstance(weights["dense"], Mapping)
        assert isinstance(weights["norm"], Mapping)
        return replace(
            self,
            dense=self.dense.import_weights(weights["dense"]),
            norm=self.norm.import_weights(weights["norm"])
        )

class ClassifierActivationTrace(eqx.Module):
    token_ids: Int[Array, " tokens"]
    token_positions: Int[Array, " tokens"]

    local_positional_embeddings: PositionalEmbeddings
    global_positional_embeddings: PositionalEmbeddings

    layer_results: tuple[TransformerLayerResult, ...]

    output_norm: Float[Array, "tokens channels"]

    def export(self) -> ParameterTree:
        result = dict(
            token_ids=self.token_ids,
            token_positions=self.token_positions,
            local_positional_embeddings=self.local_positional_embeddings.export(),
            global_positional_embeddings=self.global_positional_embeddings.export(),
            layer_results=[layer_result.export() for layer_result in self.layer_results],
            output_norm=self.output_norm,
        )
        return result


class ClassifierResult(eqx.Module):
    logits: Float[Array, "tokens channels"]
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
    transformer_config: TransformerConfig
    prediction_head_config: PredictionHeadConfig
    final_linear_config: LinearConfig

    vocab_size: int
    model_dim: int
    hidden_dim: int
    num_heads: int
    # num_groups: int  NOTE: this one seem to be not used in ModertBert attention
    head_dim: int
    attention_scale: float | None
    num_layers: int
    sliding_window_sizes: tuple[int | None, ...] | None
    context_length: int
    num_labels: int

    def random_init(
        self,
        activation_precision: DTypeLike,
        *,
        key: PRNGKeyArray,
    ) -> "Classifier":
        embedding_key, transformer_key, classifier_key = jax.random.split(key, num=3)
        embedding = self.embedding_config.random_init(
            vocab_size=self.vocab_size,
            model_dim=self.model_dim,
            key=embedding_key,
        )
        transformer = self.transformer_config.random_init(
            key=transformer_key
        )
        final_linear = self.final_linear_config.random_init(
            self.hidden_dim,
            (self.num_labels,),
            has_biases=True,
            key=classifier_key,
        )
        prediction_head = self.prediction_head_config.random_init(activation_precision, key)
        return Classifier(
            self,
            embedding=embedding,
            transformer=transformer,
            prediction_head=prediction_head,
            final_linear=final_linear
        )

    def empty(
        self,
        activation_precision: DTypeLike,
    ) -> "Classifier":
        embedding = self.embedding_config.empty(
            vocab_size=self.vocab_size,
            model_dim=self.model_dim,
        )
        transformer= self.transformer_config.empty()
        final_linear = self.final_linear_config.empty(self.hidden_dim, (self.num_labels,), True)
        prediction_head = self.prediction_head_config.empty(activation_precision)
        return Classifier(
            self,
            embedding=embedding,
            transformer=transformer,
            prediction_head=prediction_head,
            final_linear=final_linear,
        )


class Classifier(LalamoModule[ClassifierConfig]):
    embedding: EmbeddingBase
    transformer: Transformer
    prediction_head: PredictionHead
    final_linear: LinearBase

    @property
    def activation_precision(self) -> DTypeLike:
        return self.embedding.activation_precision

    @eqx.filter_jit
    def __call__(
        self,
        token_ids: Int[Array, " tokens"],
        token_positions: Int[Array, " tokens"],
        return_activation_trace: bool = False,
        lengths_without_padding: Int[Array, " batch"] | None = None,
        forward_pass_mode: ForwardPassMode = ForwardPassMode.MULTI_TOKEN,
        forward_pass_config: TransformerForwardPassConfig | None = None,
    ) -> ClassifierResult:
        inner_features = self.embedding.embed(token_ids)

        transformer_result = self.transformer(
            inner_features=inner_features,
            token_positions=token_positions,
            kv_cache=None,
            return_updated_kv_cache=False,
            return_layer_results=return_activation_trace,
            return_positional_embeddings=return_activation_trace,
            lengths_without_padding=lengths_without_padding,
            forward_pass_mode=forward_pass_mode,
            forward_pass_config=forward_pass_config,
        )

        prediction_output = self.prediction_head(transformer_result.outputs)
        (classifier_output,) = self.final_linear(prediction_output)

        logits = vmap(self.embedding.readout, in_axes=0)(classifier_output[0])

        if return_activation_trace:
            assert transformer_result.layer_results is not None
            assert transformer_result.global_positional_embeddings is not None
            assert transformer_result.local_positional_embeddings is not None
            activation_trace = ClassifierActivationTrace(
                token_ids=token_ids,
                token_positions=token_positions,
                global_positional_embeddings=transformer_result.global_positional_embeddings,
                local_positional_embeddings=transformer_result.local_positional_embeddings,
                layer_results=tuple(transformer_result.layer_results),
                output_norm=transformer_result.outputs,
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
            transformer=self.transformer.export_weights(),
            prediction_head=self.prediction_head.export_weights(),
            final_linear=self.final_linear.export_weights()
        )
        return result

    def import_weights(
        self,
        weights: ParameterTree[Array],
    ) -> Self:
        assert isinstance(weights, Mapping)
        assert isinstance(weights["embedding"], Mapping)
        assert isinstance(weights["transformer"], Mapping)
        assert isinstance(weights["prediction_head"], Mapping)
        assert isinstance(weights["final_linear"], Mapping)
        return replace(
            self,
            embedding=self.embedding.import_weights(weights["embedding"]),
            transformer=self.transformer.import_weights(weights["transformer"]),
            prediction_head=self.prediction_head.import_weights(weights["prediction_head"]),
            final_linear=self.final_linear.import_weights(weights["final_linear"])
        )
