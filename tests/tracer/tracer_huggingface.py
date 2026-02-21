import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from inspect import signature
from typing import Any, Protocol, Self

import jax
import torch
from jaxtyping import Array
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
from transformers.cache_utils import Cache
from transformers.masking_utils import create_bidirectional_mask, create_bidirectional_sliding_window_mask
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
from transformers.models.modernbert.modeling_modernbert import ModernBertEncoderLayer
from transformers.models.modernbert.modular_modernbert import ModernBertAttention, ModernBertMLP
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextDecoderLayer, Qwen3NextGatedDeltaNet
from transformers.processing_utils import Unpack

from lalamo.modules import DecoderResult
from lalamo.modules.classifier import ClassifierResult
from lalamo.modules.torch_interop import jax_to_torch, torch_to_jax
from tests.common import assert_close
from tests.tracer.tracer import ActivationTrace, DType, InferenceResult, ModelTracer

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


class HFDeltaNetAttention(Protocol):
    def forward(
        self,
        hidden_states: Tensor,
        cache_params: Any | None = None,
        cache_position: Tensor | None = None,
        attention_mask: Tensor | None = None,
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
    layers: Sequence[ModernBertEncoderLayer]
    final_norm: nn.LayerNorm
    head: HFPredictionHead
    config: Any


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
    classifier: nn.Linear

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


def _load_hf_model(
    model_type: type[AutoModelForCausalLM | AutoModelForSequenceClassification],
    model_repo: str,
    dtype: DType | None,
) -> tuple[Any, torch.device]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    device_map: str | torch.device = device
    max_memory: dict[str, str] | None = None
    device_map_env = os.getenv("LALAMO_HF_DEVICE_MAP")
    is_qwen3_next = "qwen3_next" in model_repo.lower().replace("-", "_")

    if device_map_env is not None:
        device_map = device_map_env
    if is_qwen3_next:
        device_map = "auto"

    if device_map == "auto":
        gpu_devices = [d for d in jax.devices() if d.platform == "gpu"]
        if gpu_devices:
            gib = 1024**3
            try:
                bytes_limit = gpu_devices[0].memory_stats().get("bytes_limit")
            except:
                bytes_limit = 60 * gib
            if bytes_limit is not None:
                safe_bytes = int(bytes_limit * 0.9)
                safe_gib = max(1, safe_bytes // gib)
                max_memory = {0: f"{safe_gib}GiB", "cpu": "1000GiB"}

    model_kwargs: dict[str, Any] = {
        "dtype": dtype.torch_dtype if dtype is not None else None,
        "device_map": device_map,
    }
    if max_memory is not None:
        model_kwargs["max_memory"] = max_memory
    model = model_type.from_pretrained(model_repo, **model_kwargs)

    return model, device


def _build_hf_attention_mask(
    hidden_states: Tensor,
    hf_attention: HFAttention | Gemma3Attention | HFDeltaNetAttention,
) -> Tensor:
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


def _module_device(module: nn.Module, fallback_device: torch.device) -> torch.device:
    for param in module.parameters():
        if param.device.type == "meta":
            continue
        return param.device
    for buffer in module.buffers():
        if buffer.device.type == "meta":
            continue
        return buffer.device
    return fallback_device


def _rope_forward(
    rope: HFRotaryEmbedding,
    x: torch.Tensor,
    position_ids: torch.Tensor,
    layer_type: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rope_device = _module_device(rope, device)  # type: ignore[arg-type]
    rope_kwargs: dict[str, Any] = {}
    if "layer_type" in signature(rope.forward).parameters:
        rope_kwargs["layer_type"] = layer_type
    return rope.forward(x.to(rope_device), position_ids.to(rope_device), **rope_kwargs)  # type: ignore[misc]


@dataclass(frozen=True)
class HFDecoderTracer(
    ModelTracer[
        torch.Tensor,
        HFTransformerLayer | Gemma3DecoderLayer,
        HFRMSNorm,
        HFAttention | Gemma3Attention | HFDeltaNetAttention,
        HFMLP,
    ],
):
    hf_model: HFModelForCausalLM
    device: torch.device

    def from_jax(self, array: Array) -> torch.Tensor:
        return jax_to_torch(array)

    def to_jax(self, array: torch.Tensor) -> Array:
        return torch_to_jax(array)

    def embedding(self, token_ids: torch.Tensor) -> torch.Tensor:
        embed_device = _module_device(self.hf_model.model.embed_tokens, self.device)
        return self.hf_model.model.embed_tokens.forward(token_ids.to(embed_device))

    def global_rope(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _rope_forward(self.hf_model.model.rotary_emb, x, position_ids, "full_attention", self.device)

    def local_rope(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hf_rope = getattr(self.hf_model.model, "rotary_emb_local", self.hf_model.model.rotary_emb)
        return _rope_forward(hf_rope, x, position_ids, "sliding_attention", self.device)

    def rmsnorm(self, rmsnorm: HFRMSNorm, x: torch.Tensor) -> torch.Tensor:
        rmsnorm_device = _module_device(rmsnorm, self.device)  # type: ignore[arg-type]
        return rmsnorm.forward(x.to(rmsnorm_device))

    def attention(
        self,
        attention: HFAttention | Gemma3Attention | HFDeltaNetAttention,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        attention_device = _module_device(attention, self.device)  # type: ignore[arg-type]
        if Qwen3NextGatedDeltaNet is not None and isinstance(attention, Qwen3NextGatedDeltaNet):
            # DeltaNet does not have attention
            return attention.forward(
                hidden_states=hidden_states.to(attention_device),
                attention_mask=None,
            )
        hidden_states = hidden_states.to(attention_device)
        if position_embeddings is not None:
            position_embeddings = (
                position_embeddings[0].to(attention_device),
                position_embeddings[1].to(attention_device),
            )
        attention_mask = _build_hf_attention_mask(hidden_states, attention)

        forward_kwargs: dict[str, Any] = {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
        }
        if "position_embeddings" in signature(attention.forward).parameters:
            forward_kwargs["position_embeddings"] = position_embeddings

        attention_output, _ = attention.forward(**forward_kwargs)  # type: ignore
        return attention_output

    def mlp(self, mlp: HFMLP, x: torch.Tensor) -> torch.Tensor:
        mlp_device = _module_device(mlp, self.device)  # type: ignore[arg-type]
        forward_outputs = mlp.forward(x.to(mlp_device))
        if isinstance(forward_outputs, tuple):
            forward_outputs, _ = forward_outputs
        return forward_outputs

    def layer(
        self,
        layer: HFTransformerLayer | Gemma3DecoderLayer,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        layer_device = _module_device(layer, self.device)  # type: ignore[arg-type]
        hidden_states = hidden_states.to(layer_device)
        if position_embeddings is not None:
            position_embeddings = (
                position_embeddings[0].to(layer_device),
                position_embeddings[1].to(layer_device),
            )
        torch_outputs, *_ = layer.forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
        )

        return torch_outputs

    def layer_pre_attention_norm(self, layer: HFTransformerLayer | Gemma3DecoderLayer) -> HFRMSNorm:
        return layer.input_layernorm

    # Gemma and Llama/Qwen confusingly have very different naming conventions.

    def layer_post_attention_norm(self, layer: HFTransformerLayer | Gemma3DecoderLayer) -> HFRMSNorm | None:
        if hasattr(layer, "post_feedforward_layernorm"):
            return layer.post_attention_layernorm

        return None

    def layer_pre_mlp_norm(self, layer: HFTransformerLayer | Gemma3DecoderLayer) -> HFRMSNorm:
        if hasattr(layer, "post_feedforward_layernorm"):
            return layer.pre_feedforward_layernorm  # type: ignore

        return layer.post_attention_layernorm

    def layer_attention(
        self,
        layer: HFTransformerLayer | Gemma3DecoderLayer | Qwen3NextDecoderLayer,
    ) -> HFAttention | Gemma3Attention | HFDeltaNetAttention:
        if getattr(layer, "layer_type", None) == "linear_attention" and hasattr(layer, "linear_attn"):
            return layer.linear_attn
        return layer.self_attn

    def layer_mlp(self, layer: HFTransformerLayer | Gemma3DecoderLayer) -> HFMLP:
        return layer.mlp

    def layer_post_mlp_norm(self, layer: HFTransformerLayer | Gemma3DecoderLayer) -> HFRMSNorm | None:
        return getattr(layer, "post_feedforward_layernorm", None)

    def iterate_layers(self) -> Iterable[HFTransformerLayer | Gemma3DecoderLayer]:
        return self.hf_model.model.layers

    def output_norm(self) -> HFRMSNorm:
        return self.hf_model.model.norm

    def readout(self, x: torch.Tensor) -> torch.Tensor:
        head_device = _module_device(self.hf_model.lm_head, self.device)
        return self.hf_model.lm_head(x.to(head_device))

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        embed_device = _module_device(self.hf_model.model.embed_tokens, self.device)
        input_ids = input_ids.to(embed_device)
        position_ids = position_ids.to(embed_device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=embed_device)
        hf_outputs = self.hf_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
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

        # Correct the bug in the HF Gemma implementation
        # See https://github.com/huggingface/transformers/issues/38702
        if hasattr(hf_model.model.embed_tokens, "embed_scale"):
            wrong_scale = hf_model.model.embed_tokens.embed_scale
            correct_scale = wrong_scale.to(torch.bfloat16).to(wrong_scale.dtype)
            hf_model.model.embed_tokens.embed_scale = correct_scale

        return cls(hf_model, device)


@dataclass(frozen=True)
class ModernBertTracer(
    ModelTracer[torch.Tensor, ModernBertEncoderLayer, nn.LayerNorm, ModernBertAttention, ModernBertMLP],
):
    hf_model: HFModelForSequenceClassification
    device: torch.device

    def rmsnorm(self, rmsnorm: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm.forward(x)

    def mlp(self, mlp: ModernBertMLP, x: torch.Tensor) -> torch.Tensor:
        forward_outputs = mlp.forward(x)
        if isinstance(forward_outputs, tuple):
            forward_outputs, _ = forward_outputs
        return forward_outputs

    def embedding(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.hf_model.model.embeddings.forward(token_ids)

    def from_jax(self, array: Array) -> torch.Tensor:
        return jax_to_torch(array).to(self.device)

    def to_jax(self, array: torch.Tensor) -> Array:
        return torch_to_jax(array)

    def readout(self, x: torch.Tensor) -> torch.Tensor:
        return self.hf_model.classifier(x)

    def normalized_output(self, result: InferenceResult) -> Tensor:
        assert result.activation_trace is not None
        assert isinstance(result, ClassifierResult)
        return self.from_jax(result.activation_trace.output_pooling[None, ...])

    def _update_attention_mask(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        attention_mask = torch.ones(hidden_states.shape[0:2], dtype=torch.bool, device=hidden_states.device)
        full_attention_mask = create_bidirectional_mask(
            config=self.hf_model.model.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
        )
        sliding_window_mask = create_bidirectional_sliding_window_mask(
            config=self.hf_model.model.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
        )
        return full_attention_mask, sliding_window_mask

    def match_layer(
        self,
        ref_layer: ModernBertEncoderLayer,
        layer_index: int,
        full_activation_trace: ActivationTrace,
    ) -> None:
        layer_result = full_activation_trace.layer_results[layer_index]
        position_ids = full_activation_trace.token_positions
        activation_trace = layer_result.activation_trace
        assert activation_trace is not None

        use_global_attention = layer_index % ref_layer.config.global_attn_every_n_layers == 0

        hf_pre_attention_norm = ref_layer.attn_norm
        hf_pre_mlp_norm = ref_layer.mlp_norm

        layer_hidden_states = jax_to_torch(activation_trace.pre_mixer_norm).to(self.device)
        attention_mask, sliding_window = self._update_attention_mask(layer_hidden_states)
        layer_attention_mask = attention_mask if use_global_attention else sliding_window

        if layer_index > 0:
            assert isinstance(hf_pre_attention_norm, nn.LayerNorm)
            self.match_rmsnorm(
                activation_trace.inputs,
                activation_trace.pre_mixer_norm,
                hf_pre_attention_norm,
                f"Layer {layer_index} Pre Attention RMSNorm",
            )

        assert activation_trace.positional_embeddings is not None
        cosines = activation_trace.positional_embeddings.cosines
        sines = activation_trace.positional_embeddings.sines
        assert cosines.ndim == 3
        layer_type = ref_layer.attention_type
        rotary_emb = self.hf_model.model.rotary_emb  # type: ignore[attr-defined]
        # NOTE: first argument to 'rotary_emb' should be qkv tensor but it is only used
        # for its dtype and device, actual values are irrelevant.
        fake_qkv = torch.ones(1, dtype=torch.float32)
        torch_pos_ids = jax_to_torch(position_ids)
        ref_cosines, ref_sines = _rope_forward(rotary_emb, fake_qkv, torch_pos_ids, layer_type, self.device)
        assert_close(
            result=sines,
            reference=torch_to_jax(ref_sines),
            operation_name=f"Rottary Embedding Cosines (global={use_global_attention})",
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
        )

        self.match_attention_custom(
            activation_trace.pre_mixer_norm,
            activation_trace.mixer,
            ref_layer.attn,  # type: ignore
            layer_attention_mask,
            (ref_cosines, ref_sines),
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
            ref_layer.mlp,  # type: ignore
            f"Layer {layer_index} MLP",
        )
        assert_close(
            result=cosines,
            reference=torch_to_jax(ref_cosines),
            operation_name=f"Rottary Embedding Cosines (global={use_global_attention})",
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
        )

        ref_inputs = jax_to_torch(activation_trace.inputs).to(self.device)
        assert ref_inputs.ndim == 3

        forward_outputs = ref_layer.forward(
            hidden_states=ref_inputs,
            attention_mask=layer_attention_mask.to(ref_inputs.device) if layer_attention_mask is not None else None,
            position_embeddings=(ref_cosines.to(ref_inputs.device), ref_sines.to(ref_inputs.device)),
        )
        if isinstance(forward_outputs, tuple):
            torch_outputs, *_ = forward_outputs
        else:
            torch_outputs = forward_outputs

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

    def match_attention_custom(
        self,
        llm_inputs: Array,
        llm_outputs: Array,
        ref_attention: ModernBertAttention,
        attention_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        name: str,
    ) -> None:
        ref_inputs = jax_to_torch(llm_inputs).to(self.device)
        forward_outputs = ref_attention.forward(
            hidden_states=ref_inputs,
            attention_mask=attention_mask.to(ref_inputs.device) if attention_mask is not None else None,
            position_embeddings=(
                position_embeddings[0].to(ref_inputs.device),
                position_embeddings[1].to(ref_inputs.device),
            ),
        )
        if isinstance(forward_outputs, tuple):
            torch_outputs, *_ = forward_outputs
        else:
            torch_outputs = forward_outputs
        ref_outputs = torch_to_jax(torch_outputs)
        assert_close(
            result=llm_outputs,
            reference=ref_outputs,
            operation_name=name,
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
        )

    def match_global_rope(self, activation_trace: ActivationTrace) -> None:
        # NOTE: currently in ModernBERT rope's are compared in per-layer tracing function
        pass

    def match_local_rope(self, activation_trace: ActivationTrace) -> None:
        # NOTE: currently in ModernBERT rope's are compared in per-layer tracing function
        pass

    def iterate_layers(self) -> Iterable[ModernBertEncoderLayer]:
        return self.hf_model.model.layers

    def output_norm(self) -> nn.LayerNorm:
        return self.hf_model.model.final_norm

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

        *hf_hidden_states, hf_last_norm_output = hf_outputs.hidden_states
        assert hf_outputs.logits is not None
        return (tuple(hf_hidden_states), hf_last_norm_output, hf_outputs.logits)

    @torch.no_grad()
    def match_activations(self, result: InferenceResult) -> None:
        super().match_activations(result)

    @classmethod
    def load(cls, model_repo: str, dtype: DType | None) -> Self:
        hf_model, device = _load_hf_model(AutoModelForSequenceClassification, model_repo, dtype)
        return cls(hf_model, device)
