from collections.abc import Iterable
from dataclasses import dataclass
from typing import Self

import torch
from jaxtyping import Array
from torch import LongTensor, Tensor
from transformers import AutoModelForCausalLM
from transformers.models.lfm2.modeling_lfm2 import (
    Lfm2Attention,
    Lfm2DecoderLayer,
    Lfm2ForCausalLM,
    Lfm2MLP,
    Lfm2RMSNorm,
    Lfm2ShortConv,
)

from lalamo.modules import DecoderResult
from lalamo.modules.torch_interop import jax_to_torch, torch_to_jax
from tests.tracer.tracer import DType, InferenceResult, ModelTracer
from tests.tracer.tracer_huggingface import _build_hf_attention_mask, _load_hf_model


@dataclass(frozen=True)
class LFM2DecoderTracer(
    ModelTracer[
        Tensor,
        Lfm2DecoderLayer,
        Lfm2RMSNorm,
        Lfm2Attention | Lfm2ShortConv,
        Lfm2MLP,
    ],
):
    hf_model: Lfm2ForCausalLM
    device: torch.device

    def from_jax(self, array: Array) -> Tensor:
        return jax_to_torch(array).to(self.device)

    def to_jax(self, array: Tensor) -> Array:
        return torch_to_jax(array)

    def embedding(self, token_ids: Tensor) -> Tensor:
        return self.hf_model.model.embed_tokens.forward(token_ids)

    def global_rope(self, x: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        return self.hf_model.model.rotary_emb.forward(x, position_ids)

    def local_rope(self, x: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        hf_rope = getattr(self.hf_model.model, "rotary_emb_local", self.hf_model.model.rotary_emb)
        return hf_rope.forward(x, position_ids)

    def rmsnorm(self, rmsnorm: Lfm2RMSNorm, x: Tensor) -> Tensor:
        return rmsnorm.forward(x)

    def attention(
        self,
        attention: Lfm2Attention | Lfm2ShortConv,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor] | None,
    ) -> Tensor:
        attention_mask = _build_hf_attention_mask(hidden_states, attention) # type: ignore

        if isinstance(attention, Lfm2Attention):
            assert position_embeddings is not None

            attention_output, _ = attention.forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )
        else:
            attention_output = attention.forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
            )

        return attention_output

    def mlp(self, mlp: Lfm2MLP, x: Tensor) -> Tensor:
        return mlp.forward(x)

    def layer(
        self,
        layer: Lfm2DecoderLayer,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor] | None,
    ) -> Tensor:
        torch_outputs, *_ = layer.forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings, # type: ignore
        )

        return torch_outputs

    def layer_pre_attention_norm(self, layer: Lfm2DecoderLayer) -> Lfm2RMSNorm:
        return layer.operator_norm

    def layer_pre_mlp_norm(self, layer: Lfm2DecoderLayer) -> Lfm2RMSNorm:
        return layer.ffn_norm

    def layer_attention(self, layer: Lfm2DecoderLayer) -> Lfm2Attention | Lfm2ShortConv:
        return layer.self_attn if layer.is_attention_layer else layer.conv

    def layer_mlp(self, layer: Lfm2DecoderLayer) -> Lfm2MLP:
        assert isinstance(layer.feed_forward, Lfm2MLP)

        return layer.feed_forward

    def iterate_layers(self) -> Iterable[Lfm2DecoderLayer]:
        return self.hf_model.model.layers # type: ignore

    def output_norm(self) -> Lfm2RMSNorm:
        return self.hf_model.model.embedding_norm

    def readout(self, x: Tensor) -> Tensor:
        return self.hf_model.lm_head(x)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
    ) -> tuple[tuple[Tensor, ...], Tensor, Tensor]:
        hf_outputs = self.hf_model.forward(
            input_ids=LongTensor(input_ids.long()),
            position_ids=LongTensor(position_ids.long()),
            output_hidden_states=True,
        )

        assert hf_outputs.hidden_states is not None
        assert hf_outputs.logits is not None

        *hf_hidden_states, hf_last_norm_output = hf_outputs.hidden_states

        return (tuple(hf_hidden_states), hf_last_norm_output, hf_outputs.logits)

    @torch.no_grad()
    def match_activations(self, result: InferenceResult) -> None:
        return super().match_activations(result)

    def normalized_output(self, result: InferenceResult) -> Tensor:
        assert result.activation_trace is not None
        assert isinstance(result, DecoderResult)
        return self.from_jax(result.activation_trace.output_norm[None, ...])

    @classmethod
    def load(cls, model_repo: str, dtype: DType | None) -> Self:
        hf_model, device = _load_hf_model(AutoModelForCausalLM, model_repo, dtype)

        return cls(hf_model, device)
