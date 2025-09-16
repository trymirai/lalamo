from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import jax
import torch
from jaxtyping import Array
from torch import Tensor, nn
from transformers import AutoModelForCausalLM
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer
from transformers.processing_utils import Unpack

from lalamo.modules import DecoderActivationTrace, DecoderLayerResult, PositionalEmbeddings
from lalamo.modules.decoder import DecoderResult
from lalamo.utils import jax_to_torch, torch_to_jax
from tests.common import assert_close

FRACTION_OF_ALLOWED_VIOLATIONS = 0.03


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
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: Tensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tensor: ...


class HFTextModel(Protocol):
    layers: Sequence[HFDecoderLayer | Gemma3DecoderLayer]
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


@dataclass(frozen=True)
class HFDecoderTracer:
    hf_model: HFModelForCausalLM
    device: torch.device

    def match_embedding(self, activation_trace: DecoderActivationTrace) -> None:
        first_layer_results, *_ = activation_trace.layer_results
        assert first_layer_results.activation_trace is not None
        llm_results = first_layer_results.activation_trace.inputs
        hf_embedding = self.hf_model.model.embed_tokens

        ref_input = jax_to_torch(activation_trace.token_ids)[None, ...].to(self.device)
        torch_embedding = hf_embedding.forward(ref_input)
        ref_embedding = torch_to_jax(torch_embedding).squeeze(0)
        assert_close(
            result=llm_results,
            reference=ref_embedding,
            operation_name="Embedding",
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
        )

    def match_readout(self, result: DecoderResult) -> None:
        assert result.activation_trace is not None

        llm_logits = result.logits

        ref_normalized_outputs = jax_to_torch(result.activation_trace.output_norm)[None, ...].to(self.device)
        hf_logits = self.hf_model.lm_head(ref_normalized_outputs)

        ref_logits = torch_to_jax(hf_logits).squeeze(0)

        assert_close(
            result=llm_logits,
            reference=ref_logits,
            operation_name="Readout (lm_head)",
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
        )

    def match_layer(
        self,
        layer_result: DecoderLayerResult,
        hf_layer: HFDecoderLayer | Gemma3DecoderLayer,
        layer_index: int,
    ) -> None:
        activation_trace = layer_result.activation_trace
        assert activation_trace is not None

        # Gemma and Llama/Qwen confusingly have very different naming conventions.
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
            hf_layer.self_attn,  # type: ignore
            activation_trace.positional_embeddings,
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

        self.match_mlp(
            activation_trace.pre_mlp_norm,
            activation_trace.mlp,
            hf_layer.mlp,
            f"Layer {layer_index} MLP",
        )

        if hf_post_mlp_norm is not None:
            assert activation_trace.post_mlp_norm is not None
            self.match_rmsnorm(
                activation_trace.mlp,
                activation_trace.post_mlp_norm,
                hf_post_mlp_norm,
                f"Layer {layer_index} Post MLP RMSNorm",
            )

        # Test full decoder layer
        ref_inputs = jax_to_torch(activation_trace.inputs).to(self.device)
        cosines = jax_to_torch(activation_trace.positional_embeddings.cosines).to(self.device)
        sines = jax_to_torch(activation_trace.positional_embeddings.sines).to(self.device)

        if isinstance(hf_layer, Gemma3DecoderLayer):
            torch_outputs, *_ = hf_layer.forward(
                hidden_states=ref_inputs,
                position_embeddings_global=(cosines, sines),  # type: ignore
                position_embeddings_local=(cosines, sines),  # type: ignore
            )
        else:
            torch_outputs, *_ = hf_layer.forward(
                hidden_states=ref_inputs,
                position_embeddings=(cosines, sines),
            )

        ref_outputs = torch_to_jax(torch_outputs)
        assert_close(
            result=layer_result.outputs,
            reference=ref_outputs,
            operation_name=f"Layer {layer_index} Full Output",
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
        )

    def match_rmsnorm(self, llm_inputs: Array, llm_outputs: Array, hf_layer: HFRMSNorm, name: str) -> None:
        ref_inputs = jax_to_torch(llm_inputs).to(self.device)
        torch_outputs = hf_layer.forward(ref_inputs)
        ref_outputs = torch_to_jax(torch_outputs)
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
        hf_attention: HFAttention,
        position_embeddings: PositionalEmbeddings,
        name: str,
    ) -> None:
        ref_inputs = jax_to_torch(llm_inputs).to(self.device)
        cosines = jax_to_torch(position_embeddings.cosines).to(self.device)
        sines = jax_to_torch(position_embeddings.sines).to(self.device)

        torch_outputs, _ = hf_attention.forward(
            hidden_states=ref_inputs,
            position_embeddings=(cosines, sines),
            attention_mask=None,
        )
        ref_outputs = torch_to_jax(torch_outputs)
        assert_close(
            result=llm_outputs,
            reference=ref_outputs,
            operation_name=name,
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
        )

    def match_mlp(self, llm_inputs: Array, llm_outputs: Array, hf_mlp: HFMLP, name: str) -> None:
        ref_inputs = jax_to_torch(llm_inputs).to(self.device)
        torch_outputs = hf_mlp.forward(ref_inputs)
        ref_outputs = torch_to_jax(torch_outputs)
        assert_close(
            result=llm_outputs,
            reference=ref_outputs,
            operation_name=name,
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
        )

    def match_local_rope(self, activation_trace: DecoderActivationTrace) -> None:
        llm_results = activation_trace.local_positional_embeddings
        hf_global_rope = getattr(self.hf_model.model, "rotary_emb_local", self.hf_model.model.rotary_emb)

        dummy_input = torch.zeros((), dtype=torch.float32).to(self.device)
        ref_input = jax_to_torch(activation_trace.token_positions).to(self.device)
        torch_cosines, torch_sines = hf_global_rope.forward(dummy_input, ref_input)
        ref_cosines = torch_to_jax(torch_cosines)
        ref_sines = torch_to_jax(torch_sines)
        assert_close(
            result=llm_results.cosines,
            reference=ref_cosines,
            operation_name="Local RoPE Cosines",
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
        )
        assert_close(result=llm_results.sines, reference=ref_sines, operation_name="Local RoPE Sines")

    def match_global_rope(self, activation_trace: DecoderActivationTrace) -> None:
        llm_results = activation_trace.global_positional_embeddings
        hf_global_rope = self.hf_model.model.rotary_emb

        dummy_input = torch.zeros((), dtype=torch.float32).to(self.device)
        ref_input = jax_to_torch(activation_trace.token_positions).to(self.device)
        torch_cosines, torch_sines = hf_global_rope.forward(dummy_input, ref_input)
        ref_cosines = torch_to_jax(torch_cosines)
        ref_sines = torch_to_jax(torch_sines)
        assert_close(
            result=llm_results.cosines,
            reference=ref_cosines,
            operation_name="Global RoPE Cosines",
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
        )
        assert_close(result=llm_results.sines, reference=ref_sines, operation_name="Global RoPE Sines")

    @torch.no_grad()
    def match_activations(self, result: DecoderResult) -> None:
        assert result.activation_trace is not None
        self.match_local_rope(result.activation_trace)
        self.match_global_rope(result.activation_trace)
        self.match_embedding(result.activation_trace)

        for i, (hf_layer, layer_result) in enumerate(
            zip(self.hf_model.model.layers, result.activation_trace.layer_results, strict=True),
        ):
            self.match_layer(layer_result, hf_layer, i)

        self.match_rmsnorm(
            result.activation_trace.layer_results[-1].outputs,
            result.activation_trace.output_norm,
            self.hf_model.model.norm,
            "Output RMSNorm",
        )

        self.match_readout(result)

        hf_input_ids = jax_to_torch(result.activation_trace.token_ids).to(self.device)
        hf_token_positions = jax_to_torch(result.activation_trace.token_positions).to(self.device)
        hf_outputs = self.hf_model.forward(
            input_ids=hf_input_ids,
            position_ids=hf_token_positions,
            output_hidden_states=True,
        )
        assert hf_outputs.hidden_states is not None
        *hf_hidden_states, hf_last_norm_output = hf_outputs.hidden_states

        for i, (hf_layer_inputs, layer_result) in enumerate(
            zip(hf_hidden_states, result.activation_trace.layer_results, strict=False),
        ):
            layer_activation_trace = layer_result.activation_trace
            assert layer_activation_trace is not None
            ref_layer_inputs = torch_to_jax(hf_layer_inputs)
            assert_close(
                result=layer_activation_trace.inputs,
                reference=ref_layer_inputs,
                fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
                operation_name=f"End2End Layer {i} inputs",
            )

        ref_last_norm_output = torch_to_jax(hf_last_norm_output)
        assert_close(
            result=result.activation_trace.output_norm,
            reference=ref_last_norm_output,
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
            operation_name="End2End Output RMSNorm",
        )

        assert hf_outputs.logits is not None
        ref_probas = jax.nn.softmax(torch_to_jax(hf_outputs.logits), axis=-1)
        llm_probas = jax.nn.softmax(result.logits, axis=-1)
        assert_close(
            result=llm_probas,
            reference=ref_probas,
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
            operation_name="End2End Token Probabilities",
        )


def load_hf_tracer(model_repo: str, torch_dtype: torch.dtype) -> HFDecoderTracer:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_repo,
        torch_dtype=torch_dtype,
        device_map=device,
    )

    # Correct the bug in the HF Gemma implementation
    # See https://github.com/huggingface/transformers/issues/38702
    if hasattr(hf_model.model.embed_tokens, "embed_scale"):
        wrong_scale = hf_model.model.embed_tokens.embed_scale
        correct_scale = wrong_scale.to(torch.bfloat16).to(wrong_scale.dtype)
        hf_model.model.embed_tokens.embed_scale = correct_scale

    return HFDecoderTracer(hf_model, device)
