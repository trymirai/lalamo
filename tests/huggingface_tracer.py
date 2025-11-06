from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol, Self

import torch
from jaxtyping import Array, Bool
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3Attention,
    Gemma3DecoderLayer,
)
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssAttention
from transformers.processing_utils import Unpack

from lalamo.modules.decoder import DecoderResult
from lalamo.modules.torch_interop import jax_to_torch, torch_to_jax
from tests.test_models import ModelTracer, DType

# TODO: ModernBERT per-layer tracing, remove once refactoring finished
from lalamo.modules.classifier import ClassifierActivationTrace, ClassifierResult
from lalamo.modules import (
    DecoderActivationTrace,
    PositionalEmbeddings,
    TransformerLayerResult,
)
from tests.common import assert_close
import jax

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
    hidden_states: Tensor, hf_attention: HFAttention | Gemma3Attention
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
class HFDecoderTracer(
    ModelTracer[
        torch.Tensor,
        HFTransformerLayer | Gemma3DecoderLayer,
        HFRMSNorm,
        HFAttention | Gemma3Attention,
        HFMLP,
    ],
):
    hf_model: HFModelForCausalLM
    device: torch.device

    def from_jax(self, array: Array) -> torch.Tensor:
        return jax_to_torch(array).to(self.device)

    def to_jax(self, array: torch.Tensor) -> Array:
        return torch_to_jax(array)

    def embedding(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.hf_model.model.embed_tokens.forward(token_ids)

    def global_rope(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.hf_model.model.rotary_emb.forward(x, position_ids)

    def local_rope(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hf_rope = getattr(
            self.hf_model.model, "rotary_emb_local", self.hf_model.model.rotary_emb
        )
        return hf_rope.forward(x, position_ids)

    def rmsnorm(self, rmsnorm: HFRMSNorm, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm.forward(x)

    def attention(
        self,
        attention: HFAttention | Gemma3Attention,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        attention_mask = _build_hf_attention_mask(hidden_states, attention)

        attention_output, _ = attention.forward(  # type: ignore
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,  # type: ignore
            attention_mask=attention_mask,
        )
        return attention_output

    def mlp(self, mlp: HFMLP, x: torch.Tensor) -> torch.Tensor:
        forward_outputs = mlp.forward(x)
        if isinstance(forward_outputs, tuple):
            forward_outputs, _ = forward_outputs
        return forward_outputs

    def layer(
        self,
        layer: HFTransformerLayer | Gemma3DecoderLayer,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if isinstance(layer, Gemma3DecoderLayer):
            torch_outputs, *_ = layer.forward(
                hidden_states=hidden_states,
                position_embeddings_global=position_embeddings,  # type: ignore
                position_embeddings_local=position_embeddings,  # type: ignore
            )
        else:
            torch_outputs, *_ = layer.forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
            )

        return torch_outputs

    def layer_pre_attention_norm(
        self, layer: HFTransformerLayer | Gemma3DecoderLayer
    ) -> HFRMSNorm:
        return layer.input_layernorm

    # Gemma and Llama/Qwen confusingly have very different naming conventions.

    def layer_post_attention_norm(
        self, layer: HFTransformerLayer | Gemma3DecoderLayer
    ) -> HFRMSNorm | None:
        if hasattr(layer, "post_feedforward_layernorm"):
            return layer.post_attention_layernorm

        return None

    def layer_pre_mlp_norm(
        self, layer: HFTransformerLayer | Gemma3DecoderLayer
    ) -> HFRMSNorm:
        if hasattr(layer, "post_feedforward_layernorm"):
            return layer.pre_feedforward_layernorm  # type: ignore

        return layer.post_attention_layernorm

    def layer_attention(
        self, layer: HFTransformerLayer | Gemma3DecoderLayer
    ) -> HFAttention | Gemma3Attention:
        return layer.self_attn

    def layer_mlp(self, layer: HFTransformerLayer | Gemma3DecoderLayer) -> HFMLP:
        return layer.mlp

    def layer_post_mlp_norm(
        self, layer: HFTransformerLayer | Gemma3DecoderLayer
    ) -> HFRMSNorm | None:
        return getattr(layer, "post_feedforward_layernorm", None)

    def iterate_layers(self) -> Iterable[HFTransformerLayer | Gemma3DecoderLayer]:
        return self.hf_model.model.layers

    def output_norm(self) -> HFRMSNorm:
        return self.hf_model.model.norm

    def readout(self, x: torch.Tensor) -> torch.Tensor:
        return self.hf_model.lm_head(x)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        hf_outputs = self.hf_model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            output_hidden_states=True,
        )

        assert hf_outputs.hidden_states is not None
        assert hf_outputs.logits is not None

        *hf_hidden_states, hf_last_norm_output = hf_outputs.hidden_states

        return (tuple(hf_hidden_states), hf_last_norm_output, hf_outputs.logits)

    @torch.no_grad()
    def match_activations(self, result: DecoderResult) -> None:
        return super().match_activations(result)

    @classmethod
    def load(cls, model_repo: str, dtype: DType | None) -> Self:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        hf_model = AutoModelForCausalLM.from_pretrained(
            model_repo,
            dtype=dtype.torch_dtype if dtype is not None else None,
            device_map=device,
        )

        # Correct the bug in the HF Gemma implementation
        # See https://github.com/huggingface/transformers/issues/38702
        if hasattr(hf_model.model.embed_tokens, "embed_scale"):
            wrong_scale = hf_model.model.embed_tokens.embed_scale
            correct_scale = wrong_scale.to(torch.bfloat16).to(wrong_scale.dtype)
            hf_model.model.embed_tokens.embed_scale = correct_scale

        return cls(hf_model, device)


@dataclass(frozen=True)
class HFClassifierTracer(ModelTracer):

    hf_model: HFModelForSequenceClassification
    device: torch.device

    def _hf_sliding_window_to_lalamo_format(
        self,
        sliding_window: torch.Tensor,
    ) -> Bool[Array, "tokens tokens"]:
        result = torch_to_jax(sliding_window)
        return (result == 0).astype(jax.numpy.bool)

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

        use_global_attention = (
            layer_index % hf_layer.config.global_attn_every_n_layers == 0
        )

        hf_pre_attention_norm = hf_layer.attn_norm
        hf_pre_mlp_norm = hf_layer.mlp_norm

        attention_mask = torch.ones(
            activation_trace.pre_attention_norm.shape[0:2], dtype=torch.bool
        )

        attention_mask, sliding_window = self.hf_model.model._update_attention_mask(
            attention_mask, output_attentions=False
        )

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
            attention_mask,
            sliding_window,
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

        cosines = activation_trace.positional_embeddings.cosines
        sines = activation_trace.positional_embeddings.sines
        assert cosines.ndim == 3
        torch_pos_ids = jax_to_torch(position_ids)
        # NOTE: first argument to 'rotary_emb' should be qkv tensor but it is only used
        # for its dtype and device, actual values are irrelevant.
        fake_qkv = torch.ones(1, dtype=torch.float32, device=self.device)
        ref_cosines, ref_sines = hf_layer.attn.rotary_emb(fake_qkv, torch_pos_ids)
        assert_close(
            result=sines,
            reference=torch_to_jax(ref_sines),
            operation_name=f"Rottary Embedding Cosines (global={use_global_attention})",
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
        )
        assert_close(
            result=cosines,
            reference=torch_to_jax(ref_cosines),
            operation_name=f"Rottary Embedding Cosines (global={use_global_attention})",
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
        )

        # Test full decoder layer
        ref_inputs = jax_to_torch(activation_trace.inputs).to(self.device)
        assert ref_inputs.ndim == 3

        if not use_global_attention:
            hf_layer.attn.local_attention = (1, 1)

        torch_outputs, *_ = hf_layer.forward(
            hidden_states=ref_inputs,
            position_ids=jax_to_torch(position_ids),
            sliding_window_mask=sliding_window,
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
        attention_mask: torch.Tensor,
        sliding_window_mask: torch.Tensor,
        name: str,
    ) -> None:
        ref_inputs = jax_to_torch(llm_inputs).to(self.device)
        (torch_outputs,) = hf_attention.forward(
            hidden_states=ref_inputs,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
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

    @classmethod
    def load(cls, model_repo: str, dtype: DType | None) -> Self:

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        hf_model = AutoModelForSequenceClassification.from_pretrained(
            model_repo,
            dtype=dtype.torch_dtype if dtype is not None else None,
            device_map=device,
        )

        return cls(hf_model, device)
