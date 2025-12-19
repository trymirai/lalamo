from collections.abc import Iterable
from dataclasses import dataclass
from typing import Self

import mlx.core as mx
from jaxtyping import Array
from mlx import nn
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import load

from lalamo.modules.decoder import DecoderResult
from lalamo.modules.mlx_interop import jax_to_mlx, mlx_to_jax
from tests.tracer.tracer import ActivationTrace, DType, InferenceResult, ModelTracer


def _build_mlx_attention_mask(hidden_states: mx.array) -> mx.array:
    batch, seqlen, _ = hidden_states.shape

    mask = mx.triu(mx.full((seqlen, seqlen), -1e9, dtype=hidden_states.dtype), k=1)

    return mx.broadcast_to(mask[None, None, :, :], (batch, 1, seqlen, seqlen))


@dataclass(frozen=True)
class MLXDecoderTracer(ModelTracer[mx.array, nn.Module, nn.RMSNorm, nn.Module, nn.Module]):
    mlx_model: nn.Module
    mlx_tokenizer: TokenizerWrapper

    def from_jax(self, array: Array) -> mx.array:
        return jax_to_mlx(array)

    def to_jax(self, array: mx.array) -> Array:
        return mlx_to_jax(array)

    def embedding(self, token_ids: mx.array) -> mx.array:
        assert self.mlx_model.model is not None

        return self.mlx_model.model.embed_tokens(token_ids)

    # mlx rope isn't precomputed, just skip
    def match_global_rope(self, activation_trace: ActivationTrace) -> None:
        pass

    def match_local_rope(self, activation_trace: ActivationTrace) -> None:
        pass

    def rmsnorm(self, rmsnorm: nn.Module, x: mx.array) -> mx.array:
        return rmsnorm(x)

    def attention(
        self,
        attention: nn.Module,
        hidden_states: mx.array,
        position_embeddings: tuple[mx.array, mx.array] | None,
    ) -> mx.array:
        _ = position_embeddings  # mlx doesn't accept position embeddings

        attention_mask = _build_mlx_attention_mask(hidden_states)

        attention_output = attention(hidden_states, attention_mask)
        return attention_output

    def mlp(self, mlp: nn.Module, x: mx.array) -> mx.array:
        return mlp(x)

    def layer(
        self,
        layer: nn.Module,
        hidden_states: mx.array,
        position_embeddings: tuple[mx.array, mx.array] | None,
    ) -> mx.array:
        _ = position_embeddings  # mlx doesn't accept position embeddings

        attention_mask = _build_mlx_attention_mask(hidden_states)

        return layer(hidden_states, attention_mask)

    def layer_pre_attention_norm(self, layer: nn.Module) -> nn.RMSNorm:
        assert layer.input_layernorm is not None

        return layer.input_layernorm

    def layer_attention(self, layer: nn.Module) -> nn.Module:
        assert layer.self_attn is not None

        return layer.self_attn

    def layer_mlp(self, layer: nn.Module) -> nn.Module:
        assert layer.mlp is not None

        return layer.mlp

    def layer_pre_mlp_norm(self, layer: nn.Module) -> nn.RMSNorm:
        assert layer.post_attention_layernorm is not None

        return layer.post_attention_layernorm

    def iterate_layers(self) -> Iterable[nn.Module]:
        assert self.mlx_model.model is not None

        return self.mlx_model.model.layers

    def output_norm(self) -> nn.RMSNorm:
        assert self.mlx_model.model is not None

        return self.mlx_model.model.norm

    def readout(self, x: mx.array) -> mx.array:
        assert self.mlx_model.model is not None
        return self.mlx_model.model.embed_tokens.as_linear(x)

    def forward(
        self,
        input_ids: mx.array,
        position_ids: mx.array,
    ) -> tuple[tuple[mx.array, ...], mx.array, mx.array]:
        assert mx.array_equal(position_ids.squeeze(), mx.arange(position_ids.shape[-1])), (
            "mlx always does sequential position_ids"
        )

        assert self.mlx_model.model is not None

        hidden_states = [self.embedding(input_ids)]
        for layer in self.iterate_layers():
            hidden_states.append(self.layer(layer, hidden_states[-1], None))

        last_norm_output = self.mlx_model.model(input_ids)
        output_logits = self.mlx_model(input_ids)

        return (tuple(hidden_states[:-1]), last_norm_output, output_logits)

    def normalized_output(self, result: InferenceResult) -> mx.array:
        assert result.activation_trace is not None
        assert isinstance(result, DecoderResult)
        return self.from_jax(result.activation_trace.output_norm[None, ...])

    @classmethod
    def load(cls, model_repo: str, dtype: DType | None) -> Self:
        mlx_model, mlx_tokenizer, *_ = load(model_repo)

        if dtype is not None:
            mlx_model.apply(lambda x: x.astype(dtype.mlx_dtype) if x.dtype == mx.bfloat16 else x)

        return cls(mlx_model, mlx_tokenizer)
