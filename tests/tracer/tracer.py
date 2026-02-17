import functools
import gc
import importlib.util
import os
from abc import abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Self

import jax
import jax.numpy as jnp
import pytest
import torch
from jaxtyping import Array
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssAttention

from lalamo import ClassifierModel, LanguageModel, import_model
from lalamo.common import get_default_device_bytes
from lalamo.model_import.common import ModelType
from lalamo.modules.classifier import ClassifierActivationTrace, ClassifierResult
from lalamo.modules.decoder import (
    DecoderActivationTrace,
    DecoderResult,
)
from tests.common import assert_close, checkify_forward
from tests.helpers import si, unsi

MLX_AVAILABLE = importlib.util.find_spec("mlx")
if MLX_AVAILABLE:
    import mlx.core as mx

FRACTION_OF_ALLOWED_VIOLATIONS = 0.03


class DType(Enum):
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"

    @property
    def torch_dtype(self) -> torch.dtype:
        return getattr(torch, self.value)

    @property
    def mlx_dtype(self) -> "mx.Dtype":
        return getattr(mx, self.value)  # type: ignore

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


class ModelTracer[ArrayT, LayerT, RMSNormT, AttentionT, MlpT]:
    @abstractmethod
    def from_jax(self, array: Array) -> ArrayT: ...

    @abstractmethod
    def to_jax(self, array: ArrayT) -> Array: ...

    @abstractmethod
    def embedding(self, token_ids: ArrayT) -> ArrayT: ...

    @abstractmethod
    def global_rope(self, x: ArrayT, position_ids: ArrayT) -> tuple[ArrayT, ArrayT]: ...

    @abstractmethod
    def local_rope(self, x: ArrayT, position_ids: ArrayT) -> tuple[ArrayT, ArrayT]: ...

    @abstractmethod
    def rmsnorm(self, rmsnorm: RMSNormT, x: ArrayT) -> ArrayT: ...

    @abstractmethod
    def attention(
        self,
        attention: AttentionT,
        hidden_states: ArrayT,
        position_embeddings: tuple[ArrayT, ArrayT] | None,
    ) -> ArrayT: ...

    @abstractmethod
    def mlp(self, mlp: MlpT, x: ArrayT) -> ArrayT: ...

    @abstractmethod
    def layer(
        self,
        layer: LayerT,
        hidden_states: ArrayT,
        position_embeddings: tuple[ArrayT, ArrayT] | None,
    ) -> ArrayT: ...

    @abstractmethod
    def layer_pre_attention_norm(self, layer: LayerT) -> RMSNormT: ...

    @abstractmethod
    def layer_attention(self, layer: LayerT) -> AttentionT: ...

    @abstractmethod
    def layer_post_attention_norm(self, layer: LayerT) -> RMSNormT | None: ...

    @abstractmethod
    def layer_pre_mlp_norm(self, layer: LayerT) -> RMSNormT: ...

    @abstractmethod
    def layer_mlp(self, layer: LayerT) -> MlpT: ...

    @abstractmethod
    def layer_post_mlp_norm(self, layer: LayerT) -> RMSNormT | None: ...

    @abstractmethod
    def iterate_layers(self) -> Iterable[LayerT]: ...

    @abstractmethod
    def output_norm(self) -> RMSNormT: ...

    @abstractmethod
    def readout(self, x: ArrayT) -> ArrayT: ...

    @abstractmethod
    def forward(self, input_ids: ArrayT, position_ids: ArrayT) -> tuple[tuple[ArrayT, ...], ArrayT, ArrayT]: ...

    @abstractmethod
    def normalized_output(self, result: InferenceResult) -> ArrayT: ...

    def match_embedding(self, activation_trace: ActivationTrace) -> None:
        first_layer_results, *_ = activation_trace.layer_results
        assert first_layer_results.activation_trace is not None
        llm_results = first_layer_results.activation_trace.inputs

        ref_input = self.from_jax(activation_trace.token_ids[None, ...])
        ref_native_embedding = self.embedding(ref_input)
        ref_embedding = self.to_jax(ref_native_embedding).squeeze(0)

        assert_close(
            result=llm_results,
            reference=ref_embedding,
            operation_name="Embedding",
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
        )

    def match_global_rope(self, activation_trace: ActivationTrace) -> None:
        llm_results = activation_trace.global_positional_embeddings
        assert llm_results is not None

        if llm_results is None:
            return

        ref_x = self.from_jax(jnp.array((), jnp.float32))
        ref_position_ids = self.from_jax(activation_trace.token_positions)
        ref_native_cosines, ref_native_sines = self.global_rope(ref_x, ref_position_ids)
        ref_cosines = self.to_jax(ref_native_cosines)
        ref_sines = self.to_jax(ref_native_sines)

        _, _, head_dim = llm_results.cosines.shape
        llm_cosines = llm_results.cosines
        llm_sines = llm_results.sines
        if head_dim == ref_cosines.shape[-1] * 2:
            # GPT-OSS has a different rope implementation in hf
            llm_cosines = llm_cosines[:, :, : head_dim // 2].astype(jnp.float32)
            llm_sines = llm_sines[:, :, : head_dim // 2].astype(jnp.float32)

        assert_close(
            result=llm_cosines,
            reference=ref_cosines,
            operation_name="Global RoPE Cosines",
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
        )
        assert_close(
            result=llm_sines,
            reference=ref_sines,
            operation_name="Global RoPE Sines",
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
        )

    def match_local_rope(self, activation_trace: ActivationTrace) -> None:
        llm_results = activation_trace.local_positional_embeddings
        assert llm_results is not None

        if llm_results is None:
            return

        ref_x = self.from_jax(jnp.array((), jnp.float32))
        ref_position_ids = self.from_jax(activation_trace.token_positions)
        ref_native_cosines, ref_native_sines = self.local_rope(ref_x, ref_position_ids)
        ref_cosines = self.to_jax(ref_native_cosines)
        ref_sines = self.to_jax(ref_native_sines)

        _, _, head_dim = llm_results.cosines.shape
        llm_cosines = llm_results.cosines
        llm_sines = llm_results.sines
        if head_dim == ref_cosines.shape[-1] * 2:
            # GPT-OSS has a different rope implementation in hf
            llm_cosines = llm_cosines[:, :, : head_dim // 2].astype(jnp.float32)
            llm_sines = llm_sines[:, :, : head_dim // 2].astype(jnp.float32)

        assert_close(
            result=llm_cosines,
            reference=ref_cosines,
            operation_name="Local RoPE Cosines",
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
        )
        assert_close(
            result=llm_sines,
            reference=ref_sines,
            operation_name="Local RoPE Sines",
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
        )

    def match_rmsnorm(self, llm_inputs: Array, llm_outputs: Array, ref_layer: RMSNormT, name: str) -> None:
        ref_inputs = self.from_jax(llm_inputs)
        torch_outputs = self.rmsnorm(ref_layer, ref_inputs)
        ref_outputs = self.to_jax(torch_outputs)

        assert_close(
            result=llm_outputs,
            reference=ref_outputs,
            operation_name=name,
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
        )

    def match_attention(
        self,
        llm_inputs: Array,
        llm_outputs: Array,
        ref_attention: AttentionT,
        position_embeddings: tuple[Array, Array] | None,
        name: str,
    ) -> None:
        ref_inputs = self.from_jax(llm_inputs)

        if position_embeddings is not None:
            jax_cosines, jax_sines = position_embeddings
            head_dim = jax_cosines.shape[-1] // 2

            if isinstance(ref_attention, GptOssAttention):
                jax_cosines = jax_cosines[:, :, :head_dim]
                jax_sines = jax_sines[:, :, :head_dim]

            ref_position_embeddings = (self.from_jax(jax_cosines), self.from_jax(jax_sines))
        else:
            ref_position_embeddings = None

        ref_native_outputs = self.attention(ref_attention, ref_inputs, ref_position_embeddings)
        ref_outputs = self.to_jax(ref_native_outputs)

        assert_close(
            result=llm_outputs,
            reference=ref_outputs,
            operation_name=name,
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
        )

    def match_mlp(self, llm_inputs: Array, llm_outputs: Array, ref_mlp: MlpT, name: str) -> None:
        ref_inputs = self.from_jax(llm_inputs)
        ref_native_outputs = self.mlp(ref_mlp, ref_inputs)
        ref_outputs = self.to_jax(ref_native_outputs)
        assert_close(
            result=llm_outputs,
            reference=ref_outputs,
            operation_name=name,
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
        )

    def match_layer(self, ref_layer: LayerT, layer_index: int, full_activation_trace: ActivationTrace) -> None:
        layer_result = full_activation_trace.layer_results[layer_index]
        activation_trace = layer_result.activation_trace
        assert activation_trace is not None

        ref_pre_attention_norm = self.layer_pre_attention_norm(ref_layer)
        self.match_rmsnorm(
            activation_trace.inputs,
            activation_trace.pre_mixer_norm,
            ref_pre_attention_norm,
            f"Layer {layer_index} Pre Attention RMSNorm",
        )

        ref_attention = self.layer_attention(ref_layer)
        if activation_trace.positional_embeddings is not None:
            position_embeddings = (
                activation_trace.positional_embeddings.cosines,
                activation_trace.positional_embeddings.sines,
            )
        else:
            position_embeddings = None
        self.match_attention(
            activation_trace.pre_mixer_norm,
            activation_trace.mixer,
            ref_attention,
            position_embeddings,
            f"Layer {layer_index} Attention",
        )

        ref_post_attention_norm = self.layer_post_attention_norm(ref_layer)
        if ref_post_attention_norm is not None:
            assert activation_trace.post_mixer_norm is not None
            self.match_rmsnorm(
                activation_trace.mixer,
                activation_trace.post_mixer_norm,
                ref_post_attention_norm,
                f"Layer {layer_index} Post Attention RMSNorm",
            )

        ref_pre_mlp_norm = self.layer_pre_mlp_norm(ref_layer)
        self.match_rmsnorm(
            activation_trace.mlp_inputs,
            activation_trace.pre_mlp_norm,
            ref_pre_mlp_norm,
            f"Layer {layer_index} Pre MLP RMSNorm",
        )

        ref_mlp = self.layer_mlp(ref_layer)
        self.match_mlp(
            activation_trace.pre_mlp_norm,
            activation_trace.mlp,
            ref_mlp,
            f"Layer {layer_index} MLP",
        )

        ref_post_mlp_norm = self.layer_post_mlp_norm(ref_layer)
        if ref_post_mlp_norm is not None:
            assert activation_trace.post_mlp_norm is not None
            self.match_rmsnorm(
                activation_trace.mlp,
                activation_trace.post_mlp_norm,
                ref_post_mlp_norm,
                f"Layer {layer_index} Post MLP RMSNorm",
            )

        # Test full decoder layer
        ref_inputs = self.from_jax(activation_trace.inputs)
        if activation_trace.positional_embeddings is not None:
            ref_position_embeddings = (
                self.from_jax(activation_trace.positional_embeddings.cosines),
                self.from_jax(activation_trace.positional_embeddings.sines),
            )
        else:
            ref_position_embeddings = None

        ref_native_outputs = self.layer(ref_layer, ref_inputs, ref_position_embeddings)

        ref_outputs = self.to_jax(ref_native_outputs)

        if ref_outputs.ndim != 3:
            ref_outputs = ref_outputs[None, ...]

        assert ref_outputs.ndim == 3

        assert_close(
            result=layer_result.outputs,
            reference=ref_outputs,
            operation_name=f"Layer {layer_index} Full Output",
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
        )

    def match_readout(self, result: DecoderResult | ClassifierResult) -> None:
        assert result.activation_trace is not None

        llm_logits = result.logits

        ref_normalized_outputs = self.normalized_output(result)
        ref_native_logits = self.readout(ref_normalized_outputs)
        ref_logits = self.to_jax(ref_native_logits).squeeze(0)

        assert_close(
            result=llm_logits,
            reference=ref_logits,
            operation_name="Readout (lm_head)",
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
        )

    def match_activations(self, result: InferenceResult) -> None:
        # assert isinstance(result, DecoderResult)
        assert result.activation_trace is not None
        self.match_global_rope(result.activation_trace)
        self.match_local_rope(result.activation_trace)
        self.match_embedding(result.activation_trace)

        for i, ref_layer in enumerate(self.iterate_layers()):
            self.match_layer(ref_layer, i, result.activation_trace)

        self.match_rmsnorm(
            result.activation_trace.layer_results[-1].outputs,
            result.activation_trace.output_norm,
            self.output_norm(),
            "Output RMSNorm",
        )

        self.match_readout(result)

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
            default_device_bytes = get_default_device_bytes()

        if default_device_bytes is not None and test_spec.minimum_memory_for_trace > default_device_bytes:
            pytest.skip(
                f"test requires {si(test_spec.minimum_memory_for_trace)}"
                f" but default device has only {si(default_device_bytes)} of memory",
            )

    configure_precision_for_tests()

    token_ids = jnp.arange(0, test_spec.num_tokens, dtype=jnp.int32)[None, :]
    token_positions = jnp.arange(
        0,
        test_spec.num_tokens * test_spec.token_stride,
        test_spec.token_stride,
        dtype=jnp.int32,
    )[None, :]

    tracer = model_tracer.load(
        test_spec.model_repo,
        dtype=test_spec.dtype,
    )

    model = None
    inference_results = None
    try:
        model, model_metadata = import_model(
            test_spec.model_repo,
            context_length=test_spec.num_tokens * test_spec.token_stride,
            precision=test_spec.dtype.jax_dtype if test_spec.dtype is not None else None,
        )
        with jax.disable_jit():
            match model_metadata.model_type:
                case ModelType.LANGUAGE_MODEL:
                    assert isinstance(model, LanguageModel)
                    err, inference_results = checkify_forward(model.model)(
                        token_ids=token_ids,
                        token_positions=token_positions,
                        return_updated_state=True,
                        return_activation_trace=True,
                    )
                    err.throw()

                case ModelType.CLASSIFIER_MODEL:
                    assert isinstance(model, ClassifierModel)
                    err, inference_results = checkify_forward(model.model)(
                        token_ids=token_ids,
                        token_positions=token_positions,
                        return_activation_trace=True,
                    )
                    err.throw()

        tracer.match_activations(inference_results)
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
