from collections.abc import Mapping
from dataclasses import dataclass
from itertools import batched
from typing import Self

import cartesia_mlx as cmx
import mlx.core as mx
from jaxtyping import Array
from mlx import nn

from lalamo.model_import.loaders.binding import Binder
from lalamo.model_import.model_configs import ForeignConfig, HFLlambaConfig
from lalamo.model_import.model_configs.huggingface.llamba import LlambaCheckpointLayout
from lalamo.modules.classifier import ClassifierConfig
from lalamo.modules.decoder import DecoderConfig
from lalamo.utils.mlx_interop import jax_to_mlx, mlx_to_jax
from lalamo.utils.parameter_path import ParameterPath
from tests.tracer.tracer import DType, ModelTracer


@dataclass(frozen=True)
class LlambaDecoderTracer(ModelTracer[mx.array, nn.Module]):
    model: nn.Module
    modules: Mapping[str, nn.Module]

    def from_jax(self, array: Array) -> mx.array:
        return jax_to_mlx(array)

    def to_jax(self, array: mx.array) -> Array:
        return mlx_to_jax(array)

    def build(self, foreign_config: ForeignConfig, at: Binder) -> DecoderConfig | ClassifierConfig:
        # The traced checkpoints are MLX conversions.
        assert isinstance(foreign_config, HFLlambaConfig)
        return foreign_config.build_decoder(at, LlambaCheckpointLayout.MLX)

    def module_at(self, path: ParameterPath) -> nn.Module | None:
        return self.modules.get(str(path))

    def _child(self, name: str) -> nn.Module:
        child = getattr(self.model, name)
        assert child is not None
        return child

    def embedding(self, token_ids: mx.array) -> mx.array:
        return self._child("embedding")(token_ids)

    def norm(self, norm: nn.Module, x: mx.array) -> mx.array:
        return norm(x)

    def mixer(
        self,
        mixer: nn.Module,
        x: mx.array,
        positional_embeddings: tuple[mx.array, mx.array] | None,
        path: ParameterPath,  # noqa: ARG002
    ) -> mx.array:
        assert positional_embeddings is None
        ssd_output, _ssd_states = mixer(x)
        return ssd_output

    def mlp(self, mlp: nn.Module, x: mx.array) -> mx.array:
        return mlp(x)

    def readout(self, x: mx.array) -> mx.array:
        return self._child("head")(x)

    def forward(self, input_ids: mx.array, position_ids: mx.array) -> tuple[tuple[mx.array, ...], mx.array, mx.array]:
        assert mx.array_equal(position_ids.squeeze(), mx.arange(position_ids.shape[-1])), (
            "mlx always does sequential position_ids"
        )
        hidden_states = [self.embedding(input_ids)]
        backbone = self._child("model")
        layers = backbone.layers
        assert layers is not None
        for mixer_block, mlp_block in batched(layers, n=2):
            ssd_output, _ssd_state = mixer_block(hidden_states[-1])
            hidden_states.append(mlp_block(ssd_output))
        last_norm_output, _ssm_states = backbone(self.embedding(input_ids))
        return (tuple(hidden_states[:-1]), last_norm_output, self.readout(last_norm_output))

    @classmethod
    def load(cls, model_repo: str, dtype: DType | None) -> Self:
        model = cmx.from_pretrained(model_repo)

        if dtype is not None:
            model.set_dtype(dtype.mlx_dtype)

        return cls(model, dict(model.named_modules()))
