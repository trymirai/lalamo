from collections.abc import Iterable
from dataclasses import dataclass
from itertools import batched
from typing import Self

import cartesia_mlx as cmx
import mlx.core as mx
from jaxtyping import Array
from mlx import nn

from lalamo.modules.decoder import DecoderResult
from lalamo.modules.mlx_interop import jax_to_mlx, mlx_to_jax
from tests.tracer.tracer import ActivationTrace, DType, InferenceResult, ModelTracer


@dataclass(frozen=True)
class LlambaDecoderTracer(ModelTracer[mx.array, tuple[nn.Module, nn.Module], nn.RMSNorm, nn.Module, nn.Module]):
    model: nn.Module

    def from_jax(self, array: Array) -> mx.array:
        return jax_to_mlx(array)

    def to_jax(self, array: mx.array) -> Array:
        return mlx_to_jax(array)

    def embedding(self, token_ids: mx.array) -> mx.array:
        assert self.model.embedding is not None

        return self.model.embedding(token_ids)

    # no ropes
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
        assert position_embeddings is None

        ssd_output, _ssd_states = attention(hidden_states)
        return ssd_output

    def mlp(self, mlp: nn.Module, x: mx.array) -> mx.array:
        return mlp(x)

    def layer(
        self,
        layer: tuple[nn.Module, nn.Module],
        hidden_states: mx.array,
        position_embeddings: tuple[mx.array, mx.array] | None,
    ) -> mx.array:
        assert position_embeddings is None

        ssd_output, _ssd_state = layer[0](hidden_states)
        mlp_output = layer[1](ssd_output)

        return mlp_output

    def layer_pre_attention_norm(self, layer: tuple[nn.Module, nn.Module]) -> nn.RMSNorm:
        assert layer[0].norm is not None

        return layer[0].norm

    def layer_attention(self, layer: tuple[nn.Module, nn.Module]) -> nn.Module:
        assert layer[0].layer is not None

        return layer[0].layer

    def layer_mlp(self, layer: tuple[nn.Module, nn.Module]) -> nn.Module:
        assert layer[1].layer is not None

        return layer[1].layer

    def layer_pre_mlp_norm(self, layer: tuple[nn.Module, nn.Module]) -> nn.RMSNorm:
        assert layer[1].norm is not None

        return layer[1].norm

    def iterate_layers(self) -> Iterable[tuple[nn.Module, nn.Module]]:
        assert self.model.model is not None

        return batched(self.model.model.layers, n=2)

    def output_norm(self) -> nn.RMSNorm:
        assert self.model.model is not None

        return self.model.model.norm

    def readout(self, x: mx.array) -> mx.array:
        assert self.model.head is not None
        return self.model.head(x)

    def forward(
        self,
        input_ids: mx.array,
        position_ids: mx.array,
    ) -> tuple[tuple[mx.array, ...], mx.array, mx.array]:
        assert mx.array_equal(position_ids.squeeze(), mx.arange(position_ids.shape[-1])), (
            "mlx always does sequential position_ids"
        )

        assert self.model.embedding is not None
        assert self.model.model is not None
        assert self.model.head is not None

        hidden_states = [self.embedding(input_ids)]
        for layer in self.iterate_layers():
            hidden_states.append(self.layer(layer, hidden_states[-1], None))

        last_norm_output, _ssm_states = self.model.model(self.model.embedding(input_ids))
        output_logits = self.model.head(last_norm_output)

        return (tuple(hidden_states[:-1]), last_norm_output, output_logits)

    def normalized_output(self, result: InferenceResult) -> mx.array:
        assert result.activation_trace is not None
        assert isinstance(result, DecoderResult)
        return self.from_jax(result.activation_trace.output_norm[None, ...])

    @classmethod
    def load(cls, model_repo: str, dtype: DType | None) -> Self:
        model = cmx.from_pretrained(model_repo)

        if dtype is not None:
            model.set_dtype(dtype.mlx_dtype)

        return cls(model)
