import functools
import gc
import importlib.util
import os
from abc import abstractmethod
from dataclasses import dataclass, replace
from enum import Enum
from typing import Self

import jax
import jax.numpy as jnp
import pytest
import torch
from jax.sharding import AxisType
from jaxtyping import Array

from lalamo import ClassifierModel, LanguageModel
from lalamo.model_import.common import _load_foreign_config, import_model
from lalamo.model_import.loaders.binding import Binder, Binding
from lalamo.model_import.model_configs import ForeignClassifierConfig, ForeignConfig, ForeignLMConfig
from lalamo.model_registry import ModelRegistry
from lalamo.module import Keychain
from lalamo.modules.classifier import ClassifierActivationTrace, ClassifierConfig, ClassifierResult
from lalamo.modules.decoder import DecoderActivationTrace, DecoderConfig, DecoderResult
from lalamo.modules.flow import Sequence, StepTrace, TokenMixerBase, leaves
from lalamo.modules.forward_pass import ModelForwardPassConfig
from lalamo.modules.mlp import MLPBase
from lalamo.modules.normalization import Normalization
from lalamo.modules.rope import PositionalEmbeddings
from lalamo.modules.token_mixers.attention import Attention
from lalamo.utils.memory import get_available_bytes_on_default_device
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.sharding import ShardingConfig
from tests.common import assert_close, checkify_forward
from tests.helpers import si, unsi

MLX_AVAILABLE = importlib.util.find_spec("mlx")
if MLX_AVAILABLE:
    import mlx.core as mx

FRACTION_OF_ALLOWED_VIOLATIONS = 0.03


def _tracer_sharding_config() -> ShardingConfig:
    devices = jax.devices()
    first_device, *_ = devices
    if first_device.platform == "cpu":
        devices = devices[:1]
    mesh = jax.make_mesh(
        (len(devices),),
        ("replica",),
        axis_types=(AxisType.Auto,),
        devices=devices,
    )
    return ShardingConfig(mesh=mesh)


class DType(Enum):
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"

    @property
    def torch_dtype(self) -> torch.dtype:
        return getattr(torch, self.value)

    @property
    def mlx_dtype(self) -> "mx.Dtype":
        return getattr(mx, self.value)

    @property
    def jax_dtype(self) -> jnp.dtype:
        return jnp.dtype(self.value)


@dataclass(frozen=True)
class ModelTestSpec:
    model_repo: str
    dtype: DType | None = None
    num_tokens: int = 512
    token_stride: int = 64
    convert_memory_limit: int | None = None
    minimum_memory_for_trace: int | None = None

    @property
    def test_id(self) -> str:
        return f"{self.model_repo}{(f'/{self.dtype.value}' if self.dtype is not None else '')}"


ActivationTrace = ClassifierActivationTrace | DecoderActivationTrace
InferenceResult = ClassifierResult | DecoderResult


class ModelTracer[ArrayT, ModuleT]:
    """Matches every traced program primitive against the reference module stored at its checkpoint path.

    Primitives are fed lalamo's own inputs, so each comparison isolates one module; taps and outputs are then
    compared end to end with the reference forward pass.
    """

    @abstractmethod
    def from_jax(self, array: Array) -> ArrayT: ...

    @abstractmethod
    def to_jax(self, array: ArrayT) -> Array: ...

    @abstractmethod
    def module_at(self, path: ParameterPath) -> ModuleT | None: ...

    @abstractmethod
    def embedding(self, token_ids: ArrayT) -> ArrayT: ...

    @abstractmethod
    def norm(self, norm: ModuleT, x: ArrayT) -> ArrayT: ...

    @abstractmethod
    def mixer(
        self,
        mixer: ModuleT,
        x: ArrayT,
        positional_embeddings: tuple[ArrayT, ArrayT] | None,
        path: ParameterPath,
    ) -> ArrayT: ...

    @abstractmethod
    def mlp(self, mlp: ModuleT, x: ArrayT) -> ArrayT: ...

    @abstractmethod
    def readout(self, x: ArrayT) -> ArrayT: ...

    @abstractmethod
    def forward(self, input_ids: ArrayT, position_ids: ArrayT) -> tuple[tuple[ArrayT, ...], ArrayT, ArrayT]:
        """Returns the reference hidden states (embeddings first), the final norm output and the logits."""

    def rope(self, mixer: ModuleT, path: ParameterPath, positions: ArrayT) -> tuple[ArrayT, ArrayT] | None:  # noqa: ARG002
        """The reference cosines and sines used by the mixer at `path`, if the reference exposes them."""
        return None

    def build(self, foreign_config: ForeignConfig, at: Binder) -> DecoderConfig | ClassifierConfig:
        match foreign_config:
            case ForeignLMConfig():
                return foreign_config.build_decoder(at)
            case ForeignClassifierConfig():
                return foreign_config.build_classifier(at)
        raise TypeError(f"Cannot trace {type(foreign_config).__name__}.")

    def match_step(
        self,
        binding: Binding | None,
        step: StepTrace,
        primitive: object,
        token_positions: Array,
    ) -> None:
        reference = self.module_at(binding.path) if binding is not None else None
        if binding is None or reference is None:
            return
        name = str(binding.path)
        inputs = self.from_jax(step.inputs)
        match primitive:
            case Normalization():
                reference_outputs = self.norm(reference, inputs)
            case TokenMixerBase():
                positional_embeddings = None
                if isinstance(primitive, Attention) and primitive.config.rope_config is not None:
                    embeddings = jax.vmap(primitive.config.rope_config.compute_positional_embeddings)(token_positions)
                    self.match_rope(embeddings, reference, binding.path, token_positions, name)
                    positional_embeddings = (self.from_jax(embeddings.cosines), self.from_jax(embeddings.sines))
                reference_outputs = self.mixer(reference, inputs, positional_embeddings, binding.path)
            case MLPBase() if not step.operands:
                reference_outputs = self.mlp(reference, inputs)
            case _:
                return
        assert_close(
            result=step.outputs,
            reference=self.to_jax(reference_outputs),
            operation_name=name,
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
        )

    def match_rope(
        self,
        embeddings: PositionalEmbeddings,
        reference: ModuleT,
        path: ParameterPath,
        token_positions: Array,
        name: str,
    ) -> None:
        reference_embeddings = self.rope(reference, path, self.from_jax(token_positions))
        if reference_embeddings is None:
            return
        for label, values, native_reference_values in zip(
            ("cosines", "sines"),
            (embeddings.cosines, embeddings.sines),
            reference_embeddings,
            strict=True,
        ):
            reference_values = self.to_jax(native_reference_values)
            *_, head_dim = values.shape
            # Some references only return the first half of rotate-half tables.
            compared_values = values[..., : head_dim // 2] if head_dim == reference_values.shape[-1] * 2 else values
            assert_close(
                result=compared_values.astype(jnp.float32),
                reference=reference_values,
                operation_name=f"{name} RoPE {label}",
                fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
            )

    def match_activations(
        self,
        result: InferenceResult,
        program: Sequence,
        sources: tuple[Binding | None, ...],
    ) -> None:
        trace = result.activation_trace
        assert trace is not None

        assert_close(
            result=trace.embeddings,
            reference=self.to_jax(self.embedding(self.from_jax(trace.token_ids))),
            operation_name="Embedding",
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
        )
        for binding, step, primitive in zip(sources, trace.steps, leaves(program), strict=True):
            self.match_step(binding, step, primitive, trace.token_positions)

        normalized_outputs = trace.output_pooling if isinstance(trace, ClassifierActivationTrace) else trace.outputs
        assert_close(
            result=result.logits,
            reference=self.to_jax(self.readout(self.from_jax(normalized_outputs[None, ...]))).squeeze(0),
            operation_name="Readout",
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
        )

        reference_hidden_states, reference_outputs, reference_logits = self.forward(
            self.from_jax(trace.token_ids),
            self.from_jax(trace.token_positions),
        )
        # Taps mark layer outputs, which the reference reports after its embeddings.
        _, *reference_layer_outputs = reference_hidden_states
        for index, (tap, reference_tap) in enumerate(zip(trace.taps, reference_layer_outputs, strict=False)):
            assert_close(
                result=tap,
                reference=self.to_jax(reference_tap),
                operation_name=f"End2End tap {index}",
                fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
            )
        assert_close(
            result=trace.outputs,
            reference=self.to_jax(reference_outputs),
            operation_name="End2End outputs",
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
        )
        assert_close(
            result=jax.nn.softmax(result.logits, axis=-1),
            reference=jax.nn.softmax(self.to_jax(reference_logits), axis=-1),
            operation_name="End2End Token Probabilities",
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
        )

    @classmethod
    @abstractmethod
    def load(cls, model_repo: str, dtype: DType | None) -> Self: ...


@functools.cache
def configure_precision_for_tests() -> None:
    jax.config.update("jax_default_matmul_precision", "highest")
    torch.backends.cudnn.allow_tf32 = False
    if torch.backends.cuda.is_built():
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


def _test_model(test_spec: ModelTestSpec, model_tracer: type[ModelTracer]) -> None:
    if test_spec.minimum_memory_for_trace is not None:
        if "LALAMO_MEMORY_FOR_TRACE" in os.environ:
            default_device_bytes = unsi(os.environ["LALAMO_MEMORY_FOR_TRACE"])
        else:
            default_device_bytes = get_available_bytes_on_default_device()

        if default_device_bytes is not None and test_spec.minimum_memory_for_trace > default_device_bytes:
            pytest.skip(
                f"test requires {si(test_spec.minimum_memory_for_trace)}"
                f" but default device has only {si(default_device_bytes)} of memory",
            )

    configure_precision_for_tests()

    sharding_config = _tracer_sharding_config()
    batch_sequence_sharding = sharding_config.make_sharding((None, None))
    token_ids = jax.device_put(
        jnp.arange(0, test_spec.num_tokens, dtype=jnp.int32)[None, :],
        batch_sequence_sharding,
    )
    token_positions = jax.device_put(
        jnp.arange(
            0,
            test_spec.num_tokens * test_spec.token_stride,
            test_spec.token_stride,
            dtype=jnp.int32,
        )[None, :],
        batch_sequence_sharding,
    )

    tracer = model_tracer.load(
        test_spec.model_repo,
        dtype=test_spec.dtype,
    )
    binder = Binder()
    foreign_config = _load_foreign_config(ModelRegistry.init().repo_to_model[test_spec.model_repo])
    traced_config = tracer.build(foreign_config, binder)
    sources = binder.sources(traced_config.program)

    model = None
    inference_results = None
    try:
        import_dtype = None
        if test_spec.dtype is not None:
            import_dtype = test_spec.dtype.jax_dtype
        imported_model = import_model(
            test_spec.model_repo,
            sharding_config=sharding_config,
            dtype=import_dtype,
        )
        model = imported_model.model
        keychain = Keychain.init(0, sharding_config=sharding_config)
        with jax.disable_jit():
            if isinstance(model, LanguageModel):
                program = model.decoder.program
                forward_pass_config = ModelForwardPassConfig.for_tracer_tests()
                if test_spec.dtype is not None:
                    forward_pass_config = replace(
                        forward_pass_config,
                        embedding_forward_pass_config=replace(
                            forward_pass_config.embedding_forward_pass_config,
                            activation_dtype=test_spec.dtype.jax_dtype,
                        ),
                    )
                err, inference_results = checkify_forward(model.decoder)(
                    token_ids=token_ids,
                    token_positions=token_positions,
                    return_updated_state=True,
                    return_activation_trace=True,
                    forward_pass_config=forward_pass_config,
                    keychain=keychain,
                )
                err.throw()
            elif isinstance(model, ClassifierModel):
                program = model.classifier.program
                forward_pass_config = ModelForwardPassConfig.for_tracer_tests()
                err, inference_results = checkify_forward(model.classifier)(
                    token_ids=token_ids,
                    token_positions=token_positions,
                    return_activation_trace=True,
                    forward_pass_config=forward_pass_config,
                    keychain=keychain,
                )
                err.throw()
            else:
                raise TypeError(f"Unsupported model type for tracing: {type(model).__name__}")

        tracer.match_activations(inference_results, program, sources)
    finally:
        if model is not None:
            del model
        if inference_results is not None:
            del inference_results
        del tracer
        gc.collect()
        jax.clear_caches()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
