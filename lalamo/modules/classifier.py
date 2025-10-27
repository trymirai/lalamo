from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Self

import equinox as eqx
import jax
from jax import vmap
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree
from lalamo.modules.normalization import NormalizationConfig
from lalamo.modules.transformer import Normalization, Transformer, TransformerConfig, TransformerForwardPassConfig

from .activations import GELU, Activation, SiLU
from .common import ForwardPassMode, LalamoModule
from .embedding import EmbeddingBase, EmbeddingConfig
from .linear import LinearBase, LinearConfig
from .rope import PositionalEmbeddings
from .transformer_layer import TransformerLayerResult
from .utils import vmap_twice

__all__ = [
    "Classifier",
    "ClassifierActivationTrace",
    "ClassifierConfig",
    "ClassifierResult",
]

def activation_from_str(activation: str) -> Activation:
    supported_activations = {
        "silu" : SiLU,
        "gelu" : GELU,
    }
    if activation in supported_activations:
        return supported_activations[activation]

    raise ValueError(
        f"Only activations from the following list are supported by Classifier: {supported_activations.keys()}"
    )

@dataclass(frozen=True)
class PredictionHeadConfig:
    dense_config: LinearConfig
    activation: Activation
    normalization_config: NormalizationConfig
    use_dense_bias: bool
    use_norm_bias: bool #NOTE: currently not used because Normalization class does not support bias

    def empty(self, input_size:int, output_size: int) -> "PredictionHead":
        dense_layer = self.dense_config.empty(
            input_dim=input_size,
            output_dims=(output_size,),
            has_biases=self.use_dense_bias,
        )
        activation = self.activation
        norm = self.normalization_config.empty(output_size)

        return PredictionHead(config=self, dense=dense_layer, activation=activation, norm=norm)

    def random_init(self, input_size:int, output_size: int, key: PRNGKeyArray) -> "PredictionHead":
        dense_key, _ = jax.random.split(key)
        dense_layer = self.dense_config.random_init(
            input_size,
            (output_size,),
            has_biases=self.use_dense_bias,
            key=dense_key)
        activation = self.activation
        norm = self.normalization_config.empty(output_size)

        return PredictionHead(config=self, dense=dense_layer, activation=activation, norm=norm)


class PredictionHead(LalamoModule[PredictionHeadConfig]):
    dense: LinearBase
    activation: Activation
    norm: Normalization

    def __call__(self, inner_features: Float[Array, " in_channels"])->Float[Array, " out_channels"]:
        dense_outs = vmap_twice(self.dense)(inner_features)
        dense_outs = vmap_twice(self.activation)(inner_features)
        return vmap_twice(self.norm)(dense_outs)

    @property
    def activation_precision(self) -> DTypeLike:
        return self.activation_precision

    def export_weights(self) -> ParameterTree:
        result = dict(
            dense=self.dense.export_weights(),
            norm=self.norm.export_weights(),
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
            norm=self.norm.import_weights(weights["norm"]),
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
    num_groups: int  #NOTE: this one seem to be not used in ModertBert attention
    head_dim: int
    attention_scale: float | None
    num_layers: int
    sliding_window_sizes: tuple[int | None, ...] | None
    context_length: int
    num_labels: int

    def random_init(
        self,
        *,
        key: PRNGKeyArray,
        skip_pre_attention_norm:bool = False,
    ) -> "Classifier":
        embedding_key, transformer_key, classifier_key = jax.random.split(key, num=3)
        embedding = self.embedding_config.random_init(
            vocab_size=self.vocab_size,
            model_dim=self.model_dim,
            key=embedding_key,
        )
        transformer = self.transformer_config.random_init(
            key=transformer_key,
            is_causal=False,
            skip_pre_attention_norm=skip_pre_attention_norm
        )
        final_linear = self.final_linear_config.random_init(
            input_dim=self.hidden_dim,
            output_dims=(self.num_labels,),
            has_biases=True,
            key=classifier_key,
        )
        prediction_head = self.prediction_head_config.random_init(
            input_size=self.hidden_dim,
            output_size=self.hidden_dim,
            key=key,
        )
        return Classifier(
            self,
            embedding=embedding,
            transformer=transformer,
            prediction_head=prediction_head,
            final_linear=final_linear,
        )

    def empty(
        self,
        skip_pre_attention_norm:bool = False,
    ) -> "Classifier":
        embedding = self.embedding_config.empty(
            vocab_size=self.vocab_size,
            model_dim=self.model_dim,
        )
        transformer= self.transformer_config.empty(is_causal=False, skip_pre_attention_norm=skip_pre_attention_norm)
        final_linear = self.final_linear_config.empty(
            input_dim=self.hidden_dim,
            output_dims=(self.num_labels,),
            has_biases=True,
        )
        prediction_head = self.prediction_head_config.empty(
            input_size=self.hidden_dim,
            output_size=self.hidden_dim,
        )
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
        (logits,) = vmap_twice(self.final_linear)(prediction_output)

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
            final_linear=self.final_linear.export_weights(),
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
            final_linear=self.final_linear.import_weights(weights["final_linear"]),
        )
