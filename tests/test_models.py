import functools
import gc
from abc import abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Self

import jax
import jax.numpy as jnp
from jaxtyping import Array
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssAttention

from lalamo import import_model
from lalamo.modules.decoder import DecoderActivationTrace, DecoderLayerResult, DecoderResult
from tests.common import assert_close, checkify_forward

try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
import torch

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
        return getattr(mx, self.value) # type: ignore

    @property
    def jax_dtype(self) -> jnp.dtype:
        return jnp.dtype(self.value)


@dataclass(frozen=True)
class ModelTestSpec:
    model_repo: str
    dtype: DType | None = None
    num_tokens: int = 512
    token_stride: int = 64


class DecoderTracer[ArrayT, LayerT, RMSNormT, AttentionT, MlpT]:
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
        position_embeddings: tuple[ArrayT, ArrayT],
    ) -> ArrayT: ...

    @abstractmethod
    def mlp(self, mlp: MlpT, x: ArrayT) -> ArrayT: ...

    @abstractmethod
    def layer(self, layer: LayerT, hidden_states: ArrayT, position_embeddings: tuple[ArrayT, ArrayT]) -> ArrayT: ...

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

    def match_embedding(self, activation_trace: DecoderActivationTrace) -> None:
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

    def match_global_rope(self, activation_trace: DecoderActivationTrace) -> None:
        llm_results = activation_trace.global_positional_embeddings

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

    def match_local_rope(self, activation_trace: DecoderActivationTrace) -> None:
        llm_results = activation_trace.local_positional_embeddings

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
        position_embeddings: tuple[Array, Array],
        name: str,
    ) -> None:
        ref_inputs = self.from_jax(llm_inputs)

        jax_cosines, jax_sines = position_embeddings
        head_dim = jax_cosines.shape[-1] // 2

        if isinstance(ref_attention, GptOssAttention):
            jax_cosines = jax_cosines[:, :, :head_dim]
            jax_sines = jax_sines[:, :, :head_dim]

        cosines = self.from_jax(jax_cosines)
        sines = self.from_jax(jax_sines)

        ref_native_outputs = self.attention(ref_attention, ref_inputs, (cosines, sines))
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

    def match_layer(
        self,
        layer_result: DecoderLayerResult,
        ref_layer: LayerT,
        layer_index: int,
    ) -> None:
        activation_trace = layer_result.activation_trace
        assert activation_trace is not None

        ref_pre_attention_norm = self.layer_pre_attention_norm(ref_layer)
        self.match_rmsnorm(
            activation_trace.inputs,
            activation_trace.pre_attention_norm,
            ref_pre_attention_norm,
            f"Layer {layer_index} Pre Attention RMSNorm",
        )

        ref_attention = self.layer_attention(ref_layer)
        self.match_attention(
            activation_trace.pre_attention_norm,
            activation_trace.attention,
            ref_attention,
            (activation_trace.positional_embeddings.cosines, activation_trace.positional_embeddings.sines),
            f"Layer {layer_index} Attention",
        )

        ref_post_attention_norm = self.layer_post_attention_norm(ref_layer)
        if ref_post_attention_norm is not None:
            assert activation_trace.post_attention_norm is not None
            self.match_rmsnorm(
                activation_trace.attention,
                activation_trace.post_attention_norm,
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
        cosines = self.from_jax(activation_trace.positional_embeddings.cosines)
        sines = self.from_jax(activation_trace.positional_embeddings.sines)

        ref_native_outputs = self.layer(ref_layer, ref_inputs, (cosines, sines))

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

    def match_readout(self, result: DecoderResult) -> None:
        assert result.activation_trace is not None

        llm_logits = result.logits

        ref_normalized_outputs = self.from_jax(result.activation_trace.output_norm[None, ...])
        ref_native_logits = self.readout(ref_normalized_outputs)
        ref_logits = self.to_jax(ref_native_logits).squeeze(0)

        assert_close(
            result=llm_logits,
            reference=ref_logits,
            operation_name="Readout (lm_head)",
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
        )

    def match_activations(self, result: DecoderResult) -> None:
        assert result.activation_trace is not None
        self.match_global_rope(result.activation_trace)
        self.match_local_rope(result.activation_trace)
        self.match_embedding(result.activation_trace)

        for i, (ref_layer, layer_result) in enumerate(
            zip(self.iterate_layers(), result.activation_trace.layer_results, strict=True),
        ):
            self.match_layer(layer_result, ref_layer, i)

        self.match_rmsnorm(
            result.activation_trace.layer_results[-1].outputs,
            result.activation_trace.output_norm,
            self.output_norm(),
            "Output RMSNorm",
        )

        self.match_readout(result)

        hf_input_ids = self.from_jax(result.activation_trace.token_ids)
        hf_token_positions = self.from_jax(result.activation_trace.token_positions)
        hf_hidden_states, hf_last_norm_output, hf_output_logits = self.forward(hf_input_ids, hf_token_positions)

        for i, (hf_layer_inputs, layer_result) in enumerate(
            zip(hf_hidden_states, result.activation_trace.layer_results, strict=False),
        ):
            layer_activation_trace = layer_result.activation_trace
            assert layer_activation_trace is not None
            ref_layer_inputs = self.to_jax(hf_layer_inputs)
            assert_close(
                result=layer_activation_trace.inputs,
                reference=ref_layer_inputs,
                fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
                operation_name=f"End2End Layer {i} inputs",
            )

        ref_last_norm_output = self.to_jax(hf_last_norm_output)
        assert_close(
            result=result.activation_trace.output_norm,
            reference=ref_last_norm_output,
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
            operation_name="End2End Output RMSNorm",
        )

        ref_probas = jax.nn.softmax(self.to_jax(hf_output_logits), axis=-1)
        llm_probas = jax.nn.softmax(result.logits, axis=-1)
        assert_close(
            result=llm_probas,
            reference=ref_probas,
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
            operation_name="End2End Token Probabilities",
        )

    @classmethod
    @abstractmethod
    def load(cls, model_repo: str, dtype: DType | None) -> Self: ...


@functools.cache
def configure_precision_for_tests() -> None:
    jax.config.update("jax_default_matmul_precision", "highest")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


def _test_model(test_spec: ModelTestSpec, decoder_tracer: type[DecoderTracer]) -> None:
    configure_precision_for_tests()

    llm_model, *_ = import_model(
        test_spec.model_repo,
        context_length=test_spec.num_tokens * test_spec.token_stride,
        precision=test_spec.dtype.jax_dtype if test_spec.dtype is not None else None,
    )
    tracer = decoder_tracer.load(
        test_spec.model_repo,
        dtype=test_spec.dtype,
    )

    token_ids = jnp.arange(0, test_spec.num_tokens, dtype=jnp.int32)[None, :]
    token_positions = jnp.arange(
        0,
        test_spec.num_tokens * test_spec.token_stride,
        test_spec.token_stride,
        dtype=jnp.int32,
    )[None, :]

    with jax.disable_jit():
        err, llm_result = checkify_forward(llm_model.decoder)(
            token_ids=token_ids,
            token_positions=token_positions,
            return_updated_kv_cache=True,
            return_activation_trace=True,
        )
        err.throw()

    del llm_model
    gc.collect()

    tracer.match_activations(llm_result)
