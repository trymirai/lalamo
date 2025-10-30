from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol, Self

import torch
from jaxtyping import Array
from torch import Tensor, nn
from transformers import AutoModelForCausalLM
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention, Gemma3DecoderLayer
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssAttention
from transformers.processing_utils import Unpack

from lalamo.modules.decoder import DecoderResult
from lalamo.modules.torch_interop import jax_to_torch, torch_to_jax
from tests.test_models import DecoderTracer, DType

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


def _build_hf_attention_mask(hidden_states: Tensor, hf_attention: HFAttention | Gemma3Attention) -> Tensor:
    batch, seqlen, _ = hidden_states.shape
    q_len = seqlen
    k_len = seqlen
    device = hidden_states.device
    dtype = hidden_states.dtype

    # Causal mask: mask j > i
    causal = torch.triu(torch.ones((q_len, k_len), device=device, dtype=torch.bool), diagonal=1)

    # Sliding window mask for GPT-OSS when enabled on this layer
    if isinstance(hf_attention, GptOssAttention) and hf_attention.sliding_window is not None:
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
    DecoderTracer[
        torch.Tensor,
        HFDecoderLayer | Gemma3DecoderLayer,
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

    def global_rope(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.hf_model.model.rotary_emb.forward(x, position_ids)

    def local_rope(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hf_rope = getattr(self.hf_model.model, "rotary_emb_local", self.hf_model.model.rotary_emb)
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
        layer: HFDecoderLayer | Gemma3DecoderLayer,
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

    def layer_pre_attention_norm(self, layer: HFDecoderLayer | Gemma3DecoderLayer) -> HFRMSNorm:
        return layer.input_layernorm

    # Gemma and Llama/Qwen confusingly have very different naming conventions.

    def layer_post_attention_norm(self, layer: HFDecoderLayer | Gemma3DecoderLayer) -> HFRMSNorm | None:
        if hasattr(layer, "post_feedforward_layernorm"):
            return layer.post_attention_layernorm

        return None

    def layer_pre_mlp_norm(self, layer: HFDecoderLayer | Gemma3DecoderLayer) -> HFRMSNorm:
        if hasattr(layer, "post_feedforward_layernorm"):
            return layer.pre_feedforward_layernorm  # type: ignore

        return layer.post_attention_layernorm

    def layer_attention(self, layer: HFDecoderLayer | Gemma3DecoderLayer) -> HFAttention | Gemma3Attention:
        return layer.self_attn

    def layer_mlp(self, layer: HFDecoderLayer | Gemma3DecoderLayer) -> HFMLP:
        return layer.mlp

    def layer_post_mlp_norm(self, layer: HFDecoderLayer | Gemma3DecoderLayer) -> HFRMSNorm | None:
        return getattr(layer, "post_feedforward_layernorm", None)

    def iterate_layers(self) -> Iterable[HFDecoderLayer | Gemma3DecoderLayer]:
        return self.hf_model.model.layers

    def output_norm(self) -> HFRMSNorm:
        return self.hf_model.model.norm

    def readout(self, x: torch.Tensor) -> torch.Tensor:
        return self.hf_model.lm_head(x)

    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor,
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
