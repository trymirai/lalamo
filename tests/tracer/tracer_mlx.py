from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self

import mlx.core as mx
from jaxtyping import Array
from mlx import nn
from mlx_lm.utils import load

from lalamo.utils.mlx_interop import jax_to_mlx, mlx_to_jax
from lalamo.utils.parameter_path import ParameterPath
from tests.tracer.tracer import DType, ModelTracer


def _build_mlx_attention_mask(hidden_states: mx.array) -> mx.array:
    batch, seqlen, _ = hidden_states.shape

    mask = mx.triu(mx.full((seqlen, seqlen), -1e9, dtype=hidden_states.dtype), k=1)

    return mx.broadcast_to(mask[None, None, :, :], (batch, 1, seqlen, seqlen))


@dataclass(frozen=True)
class MLXDecoderTracer(ModelTracer[mx.array, nn.Module]):
    mlx_model: nn.Module
    modules: Mapping[str, nn.Module]

    def from_jax(self, array: Array) -> mx.array:
        return jax_to_mlx(array)

    def to_jax(self, array: mx.array) -> Array:
        return mlx_to_jax(array)

    def module_at(self, path: ParameterPath) -> nn.Module | None:
        return self.modules.get(str(path))

    @property
    def text_model(self) -> nn.Module:
        text_model = self.mlx_model.model
        assert text_model is not None
        return text_model

    @property
    def embed_tokens(self) -> nn.Module:
        embed_tokens = self.text_model.embed_tokens
        assert embed_tokens is not None
        return embed_tokens

    def embedding(self, token_ids: mx.array) -> mx.array:
        return self.embed_tokens(token_ids)

    def norm(self, norm: nn.Module, x: mx.array) -> mx.array:
        return norm(x)

    def mixer(
        self,
        mixer: nn.Module,
        x: mx.array,
        positional_embeddings: tuple[mx.array, mx.array] | None,  # noqa: ARG002
        path: ParameterPath,  # noqa: ARG002
    ) -> mx.array:
        # MLX attention computes its own rotary embeddings from sequential positions.
        return mixer(x, _build_mlx_attention_mask(x))

    def mlp(self, mlp: nn.Module, x: mx.array) -> mx.array:
        return mlp(x)

    def readout(self, x: mx.array) -> mx.array:
        as_linear = self.embed_tokens.as_linear
        assert as_linear is not None
        return as_linear(x)

    def forward(self, input_ids: mx.array, position_ids: mx.array) -> tuple[tuple[mx.array, ...], mx.array, mx.array]:
        assert mx.array_equal(position_ids.squeeze(), mx.arange(position_ids.shape[-1])), (
            "mlx always does sequential position_ids"
        )
        hidden_states = [self.embedding(input_ids)]
        layers = self.text_model.layers
        assert layers is not None
        for layer in layers:
            hidden_states.append(layer(hidden_states[-1], _build_mlx_attention_mask(hidden_states[-1])))
        return (tuple(hidden_states[:-1]), self.text_model(input_ids), self.mlx_model(input_ids))

    @classmethod
    def load(cls, model_repo: str, dtype: DType | None) -> Self:
        mlx_model, *_ = load(model_repo)

        if dtype is not None:
            mlx_model.apply(lambda x: x.astype(dtype.mlx_dtype) if x.dtype == mx.bfloat16 else x)

        return cls(mlx_model, dict(mlx_model.named_modules()))
