from dataclasses import dataclass
from enum import StrEnum

import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array, Float, Int

from lalamo.exportable import Exportable
from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoConfig, LalamoModule, LogicalAxis

from .activations import Activation
from .embedding import EmbeddingBase, EmbeddingConfig
from .flow import (
    Context,
    EmitConfig,
    Sequence,
    SequenceConfig,
    StepTrace,
    Value,
    run,
    validate_program,
)
from .forward_pass import ModelForwardPassConfig
from .linear import Linear, LinearConfig
from .normalization import Normalization, NormalizationConfig
from .utils import call_vmapped

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
            sharding_config=initializer.sharding_config,
            dense=dense_layer,
            norm=norm,
            readout=readout,
        )


class PredictionHead(LalamoModule[PredictionHeadConfig]):
    dense: Linear
    norm: Normalization
    readout: Linear

    @eqx.filter_jit
    def __call__(
        self,
        inner_features: Float[Array, "batch channels"],
        *,
        keychain: Keychain,
    ) -> Float[Array, "batch logits"]:
        return call_vmapped(
            self.call_unbatched,
            inner_features,
            keychain=keychain,
            added_sharding_axis=self.sharding_config.resolve_axis(LogicalAxis.BATCH),
        )

    @eqx.filter_jit
    def call_unbatched(
        self,
        inner_features: Float[Array, " in_channels"],
        *,
        keychain: Keychain,
    ) -> Float[Array, " logits"]:
        dense_keychain, readout_keychain = keychain.split()
        (dense_outs,) = self.dense(
            inner_features,
            keychain=dense_keychain,
        )
        dense_outs = self.config.activation(dense_outs)
        norm_outs = self.norm(dense_outs)
        (result,) = self.readout(
            norm_outs,
            keychain=readout_keychain,
        )
        return result


class ClassifierActivationTrace(Exportable, eqx.Module):
    token_ids: Int[Array, "batch tokens"]
    token_positions: Int[Array, "batch tokens"]
    embeddings: Float[Array, "batch tokens channels"]
    steps: tuple[StepTrace, ...]
    taps: tuple[Value, ...]
    outputs: Float[Array, "batch tokens channels"]
    output_pooling: Float[Array, "batch channels"]
    logits: Float[Array, "batch logits"]


class ClassifierResult(Exportable, eqx.Module):
    logits: Float[Array, "batch logits"]
    activation_trace: ClassifierActivationTrace | None = None


@dataclass(frozen=True)
class ClassifierConfig(LalamoConfig):
    embedding_config: EmbeddingConfig
    program: SequenceConfig
    prediction_head_config: PredictionHeadConfig

    vocab_size: int
    model_dim: int
    num_labels: int
    classifier_pooling: PoolingType
    output_labels: tuple[str, ...] | None = None
    taps: tuple[EmitConfig, ...] = ()

    def __post_init__(self) -> None:
        validate_program(self.program, self.taps)

    def init(self, initializer: Initializer) -> "Classifier":
        return Classifier(
            config=self,
            sharding_config=initializer.sharding_config,
            embedding=self.embedding_config.init(initializer, model_dim=self.model_dim, vocab_size=self.vocab_size),
            program=self.program.init(initializer, self.model_dim),
            prediction_head=self.prediction_head_config.init(
                initializer,
                input_size=self.model_dim,
                num_labels=self.num_labels,
            ),
        )


class Classifier(LalamoModule[ClassifierConfig]):
    embedding: EmbeddingBase
    program: Sequence
    prediction_head: PredictionHead

    @eqx.filter_jit
    def __call__(
        self,
        token_ids: Int[Array, "batch tokens"],
        token_positions: Int[Array, "batch tokens"],
        return_activation_trace: bool = False,
        lengths_without_padding: Int[Array, " batch"] | None = None,
        forward_pass_config: ModelForwardPassConfig = ModelForwardPassConfig(),
        *,
        keychain: Keychain,
    ) -> ClassifierResult:
        embedding_keychain, program_keychain, prediction_head_keychain = keychain.split(3)
        embeddings = self.embedding.embed(
            token_ids,
            forward_pass_config=forward_pass_config.embedding_forward_pass_config,
            keychain=embedding_keychain,
        )
        result = run(
            self.program,
            embeddings,
            Context(
                token_ids,
                token_positions,
                lengths_without_padding,
                None,
                forward_pass_config.flow_forward_pass_config,
                self.sharding_config,
            ),
            keychain=program_keychain,
            taps=self.config.taps if return_activation_trace else (),
            trace=return_activation_trace,
        )
        outputs = result.outputs

        if self.config.classifier_pooling == PoolingType.CLS:
            pooled_output = outputs[:, 0, :]
        elif self.config.classifier_pooling == PoolingType.MEAN:
            attention_mask = jnp.ones((*token_ids.shape, 1), dtype=outputs.dtype)
            pooled_output = (outputs * attention_mask).sum(axis=1) / attention_mask.sum(axis=1)
        else:
            raise TypeError(f"classifier_pooling of unknown type: {self.config.classifier_pooling}")

        logits = self.prediction_head(
            pooled_output,
            keychain=prediction_head_keychain,
        )

        if return_activation_trace:
            assert result.trace is not None
            activation_trace = ClassifierActivationTrace(
                token_ids=token_ids,
                token_positions=token_positions,
                embeddings=embeddings,
                steps=result.trace,
                taps=result.taps,
                outputs=outputs,
                output_pooling=pooled_output,
                logits=logits,
            )
        else:
            activation_trace = None

        return ClassifierResult(
            logits=logits,
            activation_trace=activation_trace,
        )
