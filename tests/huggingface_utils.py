from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from jaxtyping import Array
from torch import Tensor, nn
from transformers import AutoModelForCausalLM
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.processing_utils import Unpack

from fartsovka.modules import DecoderActivationTrace, DecoderLayerResult, PositionalEmbeddings
from fartsovka.utils import jax_to_torch, torch_to_jax
from tests.common import assert_close


class HFRotaryEmbedding(Protocol):
    def forward(self, x: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]: ...


class HFWordEmbedding(Protocol):
    def forward(self, input_ids: Tensor) -> Tensor: ...


class HFMLP(Protocol):
    def forward(self, x: Tensor) -> Tensor: ...


class HFRMSNorm(Protocol):
    def forward(self, x: Tensor) -> Tensor: ...


class HFAttention(Protocol):
    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Tensor | None,
        past_key_value: Cache | None = None,
        cache_position: Tensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tensor: ...


class HFDecoderLayer(Protocol):
    self_attn: HFAttention
    mlp: HFMLP
    input_layernorm: HFRMSNorm
    post_attention_layernorm: HFRMSNorm

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings_global: Tensor,
        position_embeddings_local: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: Tensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tensor: ...


class HFTextModel(Protocol):
    layers: Sequence[HFDecoderLayer]
    norm: HFRMSNorm
    embed_tokens: HFWordEmbedding
    rotary_emb: HFRotaryEmbedding

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: Tensor | None = None,
        labels: Tensor | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        output_hidden_states: bool | None = None,
        cache_position: Tensor | None = None,
        logits_to_keep: int | Tensor = 0,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast: ...


class HFModelForCausalLM(Protocol):
    model: HFTextModel
    lm_head: nn.Linear

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: Tensor | None = None,
        labels: Tensor | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        output_hidden_states: bool | None = None,
        cache_position: Tensor | None = None,
        logits_to_keep: int | Tensor = 0,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> CausalLMOutputWithPast: ...


@dataclass
class HFDecoderTracer:
    hf_model: HFModelForCausalLM

    def match_embedding(self, activation_trace: DecoderActivationTrace) -> None:
        first_layer_results, *_ = activation_trace.layer_results
        assert first_layer_results.activation_trace is not None
        fs_results = first_layer_results.activation_trace.inputs
        hf_embedding = self.hf_model.model.embed_tokens

        ref_input = jax_to_torch(activation_trace.token_ids)[None, ...]
        torch_embedding = hf_embedding.forward(ref_input)
        ref_embedding = torch_to_jax(torch_embedding).squeeze(0)
        assert_close(
            result=fs_results,
            reference=ref_embedding,
            operation_name="Embedding",
        )

    def match_layer(
        self,
        layer_result: DecoderLayerResult,
        hf_layer: HFDecoderLayer,
        layer_index: int,
    ) -> None:
        activation_trace = layer_result.activation_trace
        assert activation_trace is not None

        # Gemma and Llama/Qwen сщтагыштпдн have very different naming conventions.
        if hasattr(hf_layer, "post_feedforward_layernorm"):
            hf_pre_attention_norm = hf_layer.input_layernorm
            hf_post_attention_norm = hf_layer.post_attention_layernorm
            hf_pre_mlp_norm = hf_layer.pre_feedforward_layernorm  # type: ignore
            hf_post_mlp_norm = hf_layer.post_feedforward_layernorm  # type: ignore
        else:
            hf_pre_attention_norm = hf_layer.input_layernorm
            hf_post_attention_norm = None
            hf_pre_mlp_norm = hf_layer.post_attention_layernorm
            hf_post_mlp_norm = None

        self.match_rmsnorm(
            activation_trace.inputs,
            activation_trace.pre_attention_norm,
            hf_pre_attention_norm,
            f"Layer {layer_index} Pre Attention RMSNorm",
        )

        self.match_attention(
            activation_trace.pre_attention_norm,
            activation_trace.attention,
            hf_layer.self_attn,
            activation_trace.positional_embeddings,
            activation_trace.mask,
            f"Layer {layer_index} Attention",
        )

        if hf_post_attention_norm is not None:
            assert activation_trace.post_attention_norm is not None
            self.match_rmsnorm(
                activation_trace.attention,
                activation_trace.post_attention_norm,
                hf_post_attention_norm,
                f"Layer {layer_index} Post Attention RMSNorm",
            )
        self.match_rmsnorm(
            activation_trace.mlp_inputs,
            activation_trace.pre_mlp_norm,
            hf_pre_mlp_norm,
            f"Layer {layer_index} Pre MLP RMSNorm",
        )
        if hf_post_mlp_norm is not None:
            assert activation_trace.post_mlp_norm is not None
            self.match_rmsnorm(
                activation_trace.mlp,
                activation_trace.post_mlp_norm,
                hf_post_mlp_norm,
                f"Layer {layer_index} Post MLP RMSNorm",
            )

    def match_rmsnorm(self, fs_inputs: Array, fs_outputs: Array, hf_layer: HFRMSNorm, name: str) -> None:
        ref_inputs = jax_to_torch(fs_inputs)[None, ...]
        torch_outputs = hf_layer.forward(ref_inputs)
        ref_outputs = torch_to_jax(torch_outputs).squeeze(0)
        assert_close(
            result=fs_outputs,
            reference=ref_outputs,
            operation_name=name,
        )

    def match_attention(
        self,
        fs_inputs: Array,
        fs_outputs: Array,
        hf_attention: HFAttention,
        position_embeddings: PositionalEmbeddings,
        mask: Array | None,
        name: str,
    ) -> None:
        ref_inputs = jax_to_torch(fs_inputs)[None, ...]
        cosines = jax_to_torch(position_embeddings.cosines)[None, ...]
        sines = jax_to_torch(position_embeddings.sines)[None, ...]

        if mask is not None:
            torch_mask = jax_to_torch(mask)[None, ...]
        else:
            torch_mask = None

        torch_outputs, _ = hf_attention.forward(
            hidden_states=ref_inputs,
            position_embeddings=(cosines, sines),
            attention_mask=torch_mask,
        )
        ref_outputs = torch_to_jax(torch_outputs).squeeze(0)
        assert_close(
            result=fs_outputs,
            reference=ref_outputs,
            operation_name=name,
        )

    def match_local_rope(self, activation_trace: DecoderActivationTrace) -> None:
        fs_results = activation_trace.local_positional_embeddings
        hf_global_rope = getattr(self.hf_model.model, "rotary_emb_local", self.hf_model.model.rotary_emb)

        dummy_input = torch.zeros((), dtype=torch.float32)
        ref_input = jax_to_torch(activation_trace.token_positions)
        torch_cosines, torch_sines = hf_global_rope.forward(dummy_input, ref_input[None, ...])
        ref_cosines = torch_to_jax(torch_cosines).squeeze(0)
        ref_sines = torch_to_jax(torch_sines).squeeze(0)
        assert_close(
            result=fs_results.cosines,
            reference=ref_cosines,
            operation_name="Local RoPE Cosines",
        )
        assert_close(result=fs_results.sines, reference=ref_sines, operation_name="Local RoPE Sines")

    def match_global_rope(self, activation_trace: DecoderActivationTrace) -> None:
        fs_results = activation_trace.global_positional_embeddings
        hf_global_rope = self.hf_model.model.rotary_emb

        dummy_input = torch.zeros((), dtype=torch.float32)
        ref_input = jax_to_torch(activation_trace.token_positions)
        torch_cosines, torch_sines = hf_global_rope.forward(dummy_input, ref_input[None, ...])
        ref_cosines = torch_to_jax(torch_cosines).squeeze(0)
        ref_sines = torch_to_jax(torch_sines).squeeze(0)
        assert_close(
            result=fs_results.cosines,
            reference=ref_cosines,
            operation_name="Global RoPE Cosines",
        )
        assert_close(result=fs_results.sines, reference=ref_sines, operation_name="Global RoPE Sines")

    def match_activation_trace(self, activation_trace: DecoderActivationTrace) -> None:
        self.match_local_rope(activation_trace)
        self.match_global_rope(activation_trace)
        self.match_embedding(activation_trace)

        for i, (hf_layer, layer_result) in enumerate(
            zip(self.hf_model.model.layers, activation_trace.layer_results, strict=True),
        ):
            self.match_layer(layer_result, hf_layer, i)


def load_hf_tracer(model_repo: str, torch_dtype: torch.dtype | None = None) -> HFDecoderTracer:
    result = AutoModelForCausalLM.from_pretrained(model_repo, torch_dtype=torch_dtype)
    return HFDecoderTracer(result)
