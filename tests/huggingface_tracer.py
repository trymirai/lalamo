from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import jax.nn
import jax.numpy as jnp
import torch
from jaxtyping import Array
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssAttention
from transformers.processing_utils import Unpack

from lalamo.modules import (
    DecoderActivationTrace,
    PositionalEmbeddings,
    TransformerLayerResult,
)
from lalamo.modules.classifier import ClassifierActivationTrace, ClassifierResult
from lalamo.modules.decoder import DecoderResult
from lalamo.modules.torch_interop import jax_to_torch, torch_to_jax
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
        position_embeddings: tuple[Tensor, Tensor] | None = None,
        attention_mask: Tensor | None = None,
        past_key_value: Cache | None = None,
        cache_position: Tensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tensor: ...


class HFPredictionHead(Protocol):
    def forward(
        self,
        hidden_states: Tensor,
    ) -> Tensor: ...


class HFTransformerLayer(Protocol):
    self_attn: HFAttention
    mlp: HFMLP
    input_layernorm: HFRMSNorm
    post_attention_layernorm: HFRMSNorm

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor] | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: Tensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tensor: ...


class HFTextModel(Protocol):
    layers: Sequence[HFTransformerLayer | Gemma3DecoderLayer]
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


class HFClassificationModel(Protocol):
    embeddings: HFWordEmbedding
    layers: Sequence[HFTransformerLayer]
    final_norm: HFRMSNorm

    head: HFPredictionHead
    classifier: nn.Linear

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        sliding_window_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        labels: Tensor | None = None,
        indices: Tensor | None = None,
        cu_seqlens: Tensor | None = None,
        max_seqlen: int | None = None,
        batch_size: int | None = None,
        seq_len: int | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
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


class HFModelForSequenceClassification(Protocol):
    model: HFClassificationModel

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
    ) -> SequenceClassifierOutputWithPast: ...


def _build_hf_attention_mask(
    hidden_states: Tensor, hf_attention: HFAttention
) -> Tensor:
    batch, seqlen, _ = hidden_states.shape
    q_len = seqlen
    k_len = seqlen
    device = hidden_states.device
    dtype = hidden_states.dtype

    # Causal mask: mask j > i
    causal = torch.triu(
        torch.ones((q_len, k_len), device=device, dtype=torch.bool), diagonal=1
    )

    # Sliding window mask for GPT-OSS when enabled on this layer
    if (
        isinstance(hf_attention, GptOssAttention)
        and hf_attention.sliding_window is not None
    ):
        sliding_window: int = int(hf_attention.sliding_window)
        if sliding_window < 1:
            raise ValueError(f"Invalid sliding_window={sliding_window}")
        # Mask keys strictly outside the last sliding_window positions: j < i - (sliding_window - 1)
        # Equivalent integer condition: j - i <= -sliding_window
        too_old = torch.tril(
            torch.ones((q_len, k_len), device=device, dtype=torch.bool),
            diagonal=-sliding_window,
        )
        bool_mask = causal | too_old
    else:
        bool_mask = causal

    neg_inf = torch.finfo(dtype).min
    attention_mask = (
        torch.where(
            bool_mask,
            torch.tensor(neg_inf, dtype=dtype, device=device),
            torch.tensor(0, dtype=dtype, device=device),
        )
        .unsqueeze(0)
        .unsqueeze(1)
        .expand(batch, 1, q_len, k_len)
    )
    return attention_mask


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

        ref_normalized_outputs = jax_to_torch(result.activation_trace.output_norm)[
            None, ...
        ].to(self.device)
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
        layer_result: TransformerLayerResult,
        hf_layer: HFTransformerLayer | Gemma3DecoderLayer,
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
        assert ref_inputs.ndim == 3
        cosines = jax_to_torch(activation_trace.positional_embeddings.cosines).to(
            self.device
        )
        sines = jax_to_torch(activation_trace.positional_embeddings.sines).to(
            self.device
        )
        assert cosines.ndim == 3
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
        if ref_outputs.ndim != 3:
            ref_outputs = ref_outputs[None, ...]
        assert ref_outputs.ndim == 3
        assert_close(
            result=layer_result.outputs,
            reference=ref_outputs,
            operation_name=f"Layer {layer_index} Full Output",
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
        )

    def match_rmsnorm(
        self, llm_inputs: Array, llm_outputs: Array, hf_layer: HFRMSNorm, name: str
    ) -> None:
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

        head_dim = cosines.shape[-1] // 2
        if isinstance(hf_attention, GptOssAttention):
            cosines = cosines[:, :, :head_dim]
            sines = sines[:, :, :head_dim]

        attention_mask = _build_hf_attention_mask(ref_inputs, hf_attention)

        torch_outputs, _ = hf_attention.forward(
            hidden_states=ref_inputs,
            position_embeddings=(cosines, sines),
            attention_mask=attention_mask,
        )
        ref_outputs = torch_to_jax(torch_outputs)
        assert_close(
            result=llm_outputs,
            reference=ref_outputs,
            operation_name=name,
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
        )

    def match_mlp(
        self, llm_inputs: Array, llm_outputs: Array, hf_mlp: HFMLP, name: str
    ) -> None:
        ref_inputs = jax_to_torch(llm_inputs).to(self.device)
        torch_outputs = hf_mlp.forward(ref_inputs)
        if isinstance(torch_outputs, tuple):
            torch_outputs, _ = torch_outputs
        ref_outputs = torch_to_jax(torch_outputs)
        assert_close(
            result=llm_outputs,
            reference=ref_outputs,
            operation_name=name,
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
        )

    def match_local_rope(self, activation_trace: DecoderActivationTrace) -> None:
        llm_results = activation_trace.local_positional_embeddings
        hf_global_rope = getattr(
            self.hf_model.model, "rotary_emb_local", self.hf_model.model.rotary_emb
        )

        dummy_input = torch.zeros((), dtype=torch.float32).to(self.device)
        ref_input = jax_to_torch(activation_trace.token_positions).to(self.device)
        torch_cosines, torch_sines = hf_global_rope.forward(dummy_input, ref_input)
        ref_cosines = torch_to_jax(torch_cosines)
        ref_sines = torch_to_jax(torch_sines)

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

    def match_global_rope(self, activation_trace: DecoderActivationTrace) -> None:
        llm_results = activation_trace.global_positional_embeddings
        hf_global_rope = self.hf_model.model.rotary_emb

        dummy_input = torch.zeros((), dtype=torch.float32).to(self.device)
        ref_input = jax_to_torch(activation_trace.token_positions).to(self.device)
        torch_cosines, torch_sines = hf_global_rope.forward(dummy_input, ref_input)
        ref_cosines = torch_to_jax(torch_cosines)
        ref_sines = torch_to_jax(torch_sines)

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

    @torch.no_grad()
    def match_activations(self, result: DecoderResult) -> None:
        assert result.activation_trace is not None
        self.match_local_rope(result.activation_trace)
        self.match_global_rope(result.activation_trace)
        self.match_embedding(result.activation_trace)

        for i, (hf_layer, layer_result) in enumerate(
            zip(
                self.hf_model.model.layers,
                result.activation_trace.layer_results,
                strict=True,
            ),
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
        hf_token_positions = jax_to_torch(result.activation_trace.token_positions).to(
            self.device
        )
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


@dataclass(frozen=True)
class HFClassifierTracer:
    hf_model: HFModelForSequenceClassification
    device: torch.device

    def _update_attention_mask(
        self, attention_mask: torch.Tensor, dtype: torch.dtype, local_attention: int
    ) -> tuple[torch.Tensor, torch.Tensor]:

        def _expand_mask(mask: torch.Tensor, dtype: torch.dtype):
            """
            Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
            """
            bsz, src_len = mask.size()

            expanded_mask = (
                mask[:, None, None, :].expand(bsz, 1, src_len, src_len).to(dtype)
            )

            inverted_mask = torch.tensor(1.0, dtype=dtype) - expanded_mask

            return inverted_mask.masked_fill(
                inverted_mask.to(torch.bool), torch.finfo(dtype).min
            )

        global_attention_mask = _expand_mask(attention_mask, dtype)

        # Create position indices
        rows = torch.arange(global_attention_mask.shape[2]).unsqueeze(0)
        # Calculate distance between positions
        distance = torch.abs(rows - rows.T)

        # Create sliding window mask (1 for positions within window, 0 outside)
        window_mask = (
            (distance <= local_attention // 2)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(attention_mask.device)
        )
        # Combine with existing mask
        sliding_window_mask = global_attention_mask.masked_fill(
            window_mask.logical_not(), torch.finfo(dtype).min
        )

        return global_attention_mask, sliding_window_mask

    def match_embedding(self, activation_trace: ClassifierActivationTrace) -> None:
        lalamo_embeddings = activation_trace.embedding_norm_output
        hf_embedding = self.hf_model.model.embeddings

        ref_input = jax_to_torch(activation_trace.token_ids)[None, ...].to(self.device)
        torch_embedding = hf_embedding.forward(ref_input)
        ref_embedding = torch_to_jax(torch_embedding).squeeze(0)
        assert_close(
            result=lalamo_embeddings,
            reference=ref_embedding,
            operation_name="Embedding",
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
        )

    def match_readout(self, result: ClassifierResult) -> None:
        assert result.activation_trace is not None

        llm_logits = result.logits

        ref_normalized_outputs = jax_to_torch(
            result.activation_trace.output_prediction_head
        )[None, ...].to(self.device)
        hf_logits = self.hf_model.classifier(ref_normalized_outputs)  # type: ignore

        ref_logits = torch_to_jax(hf_logits).squeeze(0)

        assert_close(
            result=llm_logits,
            reference=ref_logits,
            operation_name="Readout (final linear)",
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
        )

    def match_layer(
        self,
        layer_result: TransformerLayerResult,
        position_ids: Array,
        hf_layer: HFTransformerLayer,
        layer_index: int,
    ) -> None:
        activation_trace = layer_result.activation_trace
        assert activation_trace is not None

        hf_pre_attention_norm = hf_layer.attn_norm
        hf_pre_mlp_norm = hf_layer.mlp_norm

        if layer_index > 0:
            self.match_rmsnorm(
                activation_trace.inputs,
                activation_trace.pre_attention_norm,
                hf_pre_attention_norm,
                f"Layer {layer_index} Pre Attention RMSNorm",
            )

        self.match_attention(
            activation_trace.pre_attention_norm,
            position_ids,
            activation_trace.attention,
            hf_layer.attn,  # type: ignore
            activation_trace.positional_embeddings,
            f"Layer {layer_index} Attention",
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

        # Test full decoder layer
        ref_inputs = jax_to_torch(activation_trace.inputs).to(self.device)
        assert ref_inputs.ndim == 3
        cosines = jax_to_torch(activation_trace.positional_embeddings.cosines).to(
            self.device
        )
        sines = jax_to_torch(activation_trace.positional_embeddings.sines).to(
            self.device
        )
        assert cosines.ndim == 3
        torch_outputs, *_ = hf_layer.forward(
            hidden_states=ref_inputs,
            position_ids=jax_to_torch(position_ids),
        )

        ref_outputs = torch_to_jax(torch_outputs)
        if ref_outputs.ndim != 3:
            ref_outputs = ref_outputs[None, ...]
        assert ref_outputs.ndim == 3
        assert_close(
            result=layer_result.outputs,
            reference=ref_outputs,
            operation_name=f"Layer {layer_index} Full Output",
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
        )

    def match_rmsnorm(
        self, llm_inputs: Array, llm_outputs: Array, hf_layer: HFRMSNorm, name: str
    ) -> None:
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
        position_ids: Array,
        llm_outputs: Array,
        hf_attention: HFAttention,
        position_embeddings: PositionalEmbeddings,
        name: str,
    ) -> None:
        ref_inputs = jax_to_torch(llm_inputs).to(self.device)
        attention_mask = torch.ones(llm_inputs.shape[0:2], dtype=torch.bool)

        attention_mask, sliding_window = self._update_attention_mask(
            attention_mask, torch.float32, self.hf_model.config.local_attention
        )

        (torch_outputs,) = hf_attention.forward(
            hidden_states=ref_inputs,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window,
            position_ids=torch.Tensor(position_ids.tolist()),
        )
        ref_outputs = torch_to_jax(torch_outputs)
        assert_close(
            result=llm_outputs,
            reference=ref_outputs,
            operation_name=name,
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
        )

    def match_mlp(
        self, llm_inputs: Array, llm_outputs: Array, hf_mlp: HFMLP, name: str
    ) -> None:
        ref_inputs = jax_to_torch(llm_inputs).to(self.device)
        torch_outputs = hf_mlp.forward(ref_inputs)
        if isinstance(torch_outputs, tuple):
            torch_outputs, _ = torch_outputs
        ref_outputs = torch_to_jax(torch_outputs)
        assert_close(
            result=llm_outputs,
            reference=ref_outputs,
            operation_name=name,
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
        )

    def match_prediction_head(
        self,
        llm_inputs: Array,
        llm_outputs: Array,
        hf_head: HFPredictionHead,
        name: str,
    ) -> None:
        ref_inputs = jax_to_torch(llm_inputs).to(self.device)
        torch_outputs = hf_head.forward(ref_inputs)
        if isinstance(torch_outputs, tuple):
            torch_outputs, _ = torch_outputs
        ref_outputs = torch_to_jax(torch_outputs)
        assert_close(
            result=llm_outputs,
            reference=ref_outputs,
            operation_name=name,
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
        )

    @torch.no_grad()
    def match_activations(self, result: ClassifierResult) -> None:
        assert result.activation_trace is not None
        self.match_embedding(result.activation_trace)

        for i, (hf_layer, layer_result) in enumerate(
            zip(
                self.hf_model.model.layers,
                result.activation_trace.layer_results,
                strict=True,
            ),
        ):
            self.match_layer(
                layer_result, result.activation_trace.token_positions, hf_layer, i
            )

        self.match_rmsnorm(
            result.activation_trace.layer_results[-1].outputs,
            result.activation_trace.output_norm,
            self.hf_model.model.final_norm,
            "Output RMSNorm",
        )

        self.match_prediction_head(
            result.activation_trace.output_pooling,
            result.activation_trace.output_prediction_head,
            self.hf_model.head,
            "Prediction Head",
        )

        self.match_readout(result)

        hf_input_ids = jax_to_torch(result.activation_trace.token_ids).to(self.device)
        hf_token_positions = jax_to_torch(result.activation_trace.token_positions).to(
            self.device
        )
        hf_outputs = self.hf_model.forward(
            input_ids=hf_input_ids,
            position_ids=hf_token_positions,
            output_hidden_states=True,
        )
        assert hf_outputs.hidden_states is not None
        *hf_hidden_states, hf_last_non_norm_output = hf_outputs.hidden_states

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

        # NOTE: Because of how layers outputs are saved in ModernBert we will only see
        # non-normalized per-layer outputs in the 'hidden_states'
        last_norm_output = result.activation_trace.output_norm
        ref_last_norm_output = torch_to_jax(
            self.hf_model.model.final_norm.forward(hf_last_non_norm_output)
        )
        assert_close(
            result=last_norm_output,
            reference=ref_last_norm_output,
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
            operation_name="End2End Output Normalized",
        )

        assert hf_outputs.logits is not None
        assert_close(
            result=result.logits,
            reference=torch_to_jax(hf_outputs.logits),
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
            operation_name="End2End Logits",
        )
        ref_probas = jax.nn.softmax(torch_to_jax(hf_outputs.logits), axis=-1)
        llm_probas = jax.nn.softmax(result.logits, axis=-1)
        assert_close(
            result=llm_probas,
            reference=ref_probas,
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
            operation_name="End2End Token Probabilities",
        )


def load_hf_tracer(model_repo: str, dtype: torch.dtype) -> HFDecoderTracer:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_repo,
        dtype=dtype,
        device_map=device,
    )

    # Correct the bug in the HF Gemma implementation
    # See https://github.com/huggingface/transformers/issues/38702
    if hasattr(hf_model.model.embed_tokens, "embed_scale"):
        wrong_scale = hf_model.model.embed_tokens.embed_scale
        correct_scale = wrong_scale.to(torch.bfloat16).to(wrong_scale.dtype)
        hf_model.model.embed_tokens.embed_scale = correct_scale

    return HFDecoderTracer(hf_model, device)


def load_hf_classifier_tracer(
    model_repo: str, dtype: torch.dtype
) -> HFClassifierTracer:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    hf_model = AutoModelForSequenceClassification.from_pretrained(
        model_repo,
        dtype=dtype,
        device_map=device,
    )

    return HFClassifierTracer(hf_model, device)
