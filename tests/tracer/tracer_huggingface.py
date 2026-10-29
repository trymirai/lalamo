import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from inspect import signature
from typing import Any, Protocol, Self, cast

import jax
import torch
from jaxtyping import Array
from torch import Tensor, nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSequenceClassification
from transformers.masking_utils import create_bidirectional_mask, create_bidirectional_sliding_window_mask
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssAttention
from transformers.models.modernbert.modeling_modernbert import ModernBertEncoderLayer
from transformers.models.modernbert.modular_modernbert import ModernBertAttention
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextGatedDeltaNet
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM

from lalamo.model_import.loaders.binding import Binding
from lalamo.modules.decoder import DecoderResult
from lalamo.modules.flow import Sequence
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.torch_interop import jax_to_torch, torch_to_jax
from tests.common import assert_close
from tests.tracer.tracer import DType, InferenceResult, ModelTracer

FRACTION_OF_ALLOWED_VIOLATIONS = 0.03


class TorchModuleWithDevice(Protocol):
    def parameters(self) -> Iterable[nn.Parameter]: ...

    def buffers(self) -> Iterable[Tensor]: ...


class HFRotaryEmbedding(TorchModuleWithDevice, Protocol):
    def forward(self, x: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]: ...


class HFWordEmbeddingWithScale(Protocol):
    embed_scale: Tensor


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
    max_memory: dict[int | str, str] | None = None
    device_map_env = os.getenv("LALAMO_HF_DEVICE_MAP")
    config = AutoConfig.from_pretrained(model_repo)
    if config.get_text_config().model_type in {"qwen3_next", "qwen4_exp_text"}:
        device_map = "auto"
    if device_map_env is not None:
        device_map = device_map_env

    if device_map == "auto":
        gpu_devices = [d for d in jax.devices() if d.platform == "gpu"]
        if gpu_devices:
            gib = 1024**3
            try:
                bytes_limit = gpu_devices[0].memory_stats().get("bytes_limit")
            except (AttributeError, NotImplementedError, RuntimeError):
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
    model = model_type.from_pretrained(model_repo, config=config, **model_kwargs)

    return model, device


def _build_hf_attention_mask(hidden_states: Tensor, hf_attention: nn.Module) -> Tensor:
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
    return (
        torch.where(
            bool_mask,
            torch.tensor(neg_inf, dtype=dtype, device=device),
            torch.tensor(0, dtype=dtype, device=device),
        )
        .unsqueeze(0)
        .unsqueeze(1)
        .expand(batch, 1, q_len, k_len)
    )


def _module_device(module: TorchModuleWithDevice, fallback_device: torch.device) -> torch.device:
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


def _named_modules(model: nn.Module) -> dict[str, nn.Module]:
    return dict(model.named_modules())


@dataclass(frozen=True)
class HFDecoderTracer(ModelTracer[Tensor, nn.Module]):
    hf_model: Any
    device: torch.device
    modules: Mapping[str, nn.Module]

    @property
    def text_model(self) -> Any:  # noqa: ANN401
        model = self.hf_model.model
        return getattr(model, "language_model", model)

    def from_jax(self, array: Array) -> Tensor:
        return jax_to_torch(array)

    def to_jax(self, array: Tensor) -> Array:
        return torch_to_jax(array)

    def module_at(self, path: ParameterPath) -> nn.Module | None:
        # Bound paths are canonical; multimodal checkpoints nest the language model one level deeper.
        for name in (str(path), str(path).replace("model.", "model.language_model.", 1)):
            if name in self.modules:
                return self.modules[name]
        return None

    def embedding(self, token_ids: Tensor) -> Tensor:
        embed_device = _module_device(self.text_model.embed_tokens, self.device)
        return self.text_model.embed_tokens.forward(token_ids.to(embed_device))

    def norm(self, norm: nn.Module, x: Tensor) -> Tensor:
        return norm.forward(x.to(_module_device(norm, self.device)))

    def mixer(
        self,
        mixer: nn.Module,
        x: Tensor,
        positional_embeddings: tuple[Tensor, Tensor] | None,
        path: ParameterPath,  # noqa: ARG002
    ) -> Tensor:
        mixer_device = _module_device(mixer, self.device)
        x = x.to(mixer_device)
        if isinstance(mixer, (Qwen3NextGatedDeltaNet, Qwen3_5GatedDeltaNet)):
            return mixer.forward(hidden_states=x, attention_mask=None)
        if positional_embeddings is not None:
            cosines, sines = (embedding.to(mixer_device) for embedding in positional_embeddings)
            if isinstance(mixer, GptOssAttention):
                # GPT-OSS rotates interleaved halves and takes only the first half of the tables.
                head_dim = cosines.shape[-1] // 2
                cosines, sines = cosines[..., :head_dim], sines[..., :head_dim]
            positional_embeddings = (cosines, sines)
        forward_kwargs: dict[str, Any] = {
            "hidden_states": x,
            "attention_mask": _build_hf_attention_mask(x, mixer),
        }
        if "position_embeddings" in signature(mixer.forward).parameters:
            forward_kwargs["position_embeddings"] = positional_embeddings
        attention_output, _ = mixer.forward(**forward_kwargs)
        return attention_output

    def mlp(self, mlp: nn.Module, x: Tensor) -> Tensor:
        forward_outputs = mlp.forward(x.to(_module_device(mlp, self.device)))
        if isinstance(forward_outputs, tuple):
            forward_outputs, _ = forward_outputs
        return forward_outputs

    def rope(self, mixer: nn.Module, path: ParameterPath, positions: Tensor) -> tuple[Tensor, Tensor] | None:  # noqa: ARG002
        layer_type = "sliding_attention" if getattr(mixer, "is_sliding", False) else "full_attention"
        return _rope_forward(
            self.text_model.rotary_emb,
            torch.zeros((), dtype=torch.float32),
            positions,
            layer_type,
            self.device,
        )

    def readout(self, x: Tensor) -> Tensor:
        head_device = _module_device(self.hf_model.lm_head, self.device)
        return self.hf_model.lm_head(x.to(head_device))

    def forward(self, input_ids: Tensor, position_ids: Tensor) -> tuple[tuple[Tensor, ...], Tensor, Tensor]:
        embed_device = _module_device(self.text_model.embed_tokens, self.device)
        input_ids = input_ids.to(embed_device)
        hf_outputs = self.hf_model.forward(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids, dtype=torch.bool, device=embed_device),
            position_ids=position_ids.to(embed_device),
            output_hidden_states=True,
        )
        assert hf_outputs.hidden_states is not None
        assert hf_outputs.logits is not None
        *hf_hidden_states, hf_last_norm_output = hf_outputs.hidden_states
        return (tuple(hf_hidden_states), hf_last_norm_output, hf_outputs.logits)

    @torch.no_grad()
    def match_activations(
        self,
        result: InferenceResult,
        program: Sequence,
        sources: tuple[Binding | None, ...],
    ) -> None:
        super().match_activations(result, program, sources)

    @classmethod
    def load(cls, model_repo: str, dtype: DType | None) -> Self:
        hf_model, device = _load_hf_model(AutoModelForCausalLM, model_repo, dtype)

        # Correct the bug in the HF Gemma implementation
        # See https://github.com/huggingface/transformers/issues/38702
        text_model: Any = getattr(hf_model.model, "language_model", hf_model.model)
        embed_scale = getattr(text_model.embed_tokens, "embed_scale", None)
        if isinstance(embed_scale, torch.Tensor):
            embed_tokens = cast("HFWordEmbeddingWithScale", text_model.embed_tokens)
            embed_tokens.embed_scale = embed_scale.to(torch.bfloat16).to(embed_scale.dtype)

        return cls(hf_model, device, _named_modules(hf_model))


@dataclass(frozen=True)
class HFQwen38Tracer(HFDecoderTracer):
    @torch.no_grad()
    def match_activations(
        self,
        result: InferenceResult,
        program: Sequence,  # noqa: ARG002
        sources: tuple[Binding | None, ...],  # noqa: ARG002
    ) -> None:
        assert isinstance(result, DecoderResult) and result.activation_trace is not None
        layers: list[Tensor] = []
        readout: list[Tensor] = []

        def capture_layer(module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor) -> None:  # noqa: ARG001
            layers.append(output.detach())

        def capture_readout(module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor) -> None:  # noqa: ARG001
            readout.append(output.detach())

        handles = [layer.register_forward_hook(capture_layer) for layer in self.hf_model.model.layers]
        handles.append(self.hf_model.model.hyper_connection_mixer.register_forward_hook(capture_readout))
        try:
            device = _module_device(self.hf_model.model.embed_tokens, self.device)
            outputs = self.hf_model(
                input_ids=jax_to_torch(result.activation_trace.token_ids).to(device),
                position_ids=jax_to_torch(result.activation_trace.token_positions).to(device),
                use_cache=False,
            )
        finally:
            for handle in handles:
                handle.remove()
        for index, (reference, tap) in enumerate(zip(layers, result.activation_trace.taps, strict=True)):
            assert_close(
                result=tap,
                reference=torch_to_jax(reference),
                operation_name=f"Qwen3.8 layer {index}",
                fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
            )
        (normalized,) = readout
        assert_close(
            result=result.activation_trace.outputs,
            reference=torch_to_jax(normalized),
            operation_name="Qwen3.8 gated readout",
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
        )
        assert outputs.logits is not None
        assert_close(
            result=jax.nn.softmax(result.logits, axis=-1),
            reference=jax.nn.softmax(torch_to_jax(outputs.logits.float()), axis=-1),
            operation_name="Qwen3.8 token probabilities",
            fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
        )

    @classmethod
    def load(cls, model_repo: str, dtype: DType | None) -> Self:
        model, device = _load_hf_model(AutoModelForCausalLM, model_repo, dtype)
        assert isinstance(model, Qwen4ExpForCausalLM)
        # Torch grouped GEMM requires 16-byte strides; tracing uses the reference expert implementation.
        model.set_experts_implementation("eager")
        return cls(model, device, _named_modules(model))


@dataclass(frozen=True)
class ModernBertTracer(HFDecoderTracer):
    def embedding(self, token_ids: Tensor) -> Tensor:
        # The embedding norm is a separate program step, matched at its own checkpoint path.
        return self.hf_model.model.embeddings.tok_embeddings(token_ids.to(self.device))

    def _layer(self, path: ParameterPath) -> ModernBertEncoderLayer:
        layer = self.module_at(ParameterPath(path.rpartition(".")[0]))
        assert isinstance(layer, ModernBertEncoderLayer)
        return layer

    def _attention_mask(self, hidden_states: Tensor, layer: ModernBertEncoderLayer) -> Tensor | None:
        attention_mask = torch.ones(hidden_states.shape[0:2], dtype=torch.bool, device=hidden_states.device)
        if layer.attention_type == "sliding_attention":
            mask = create_bidirectional_sliding_window_mask(
                config=self.hf_model.model.config,
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
            )
        else:
            mask = create_bidirectional_mask(
                config=self.hf_model.model.config,
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
            )
        assert mask is None or isinstance(mask, Tensor)
        return mask

    def mixer(
        self,
        mixer: nn.Module,
        x: Tensor,
        positional_embeddings: tuple[Tensor, Tensor] | None,
        path: ParameterPath,
    ) -> Tensor:
        assert isinstance(mixer, ModernBertAttention) and positional_embeddings is not None
        x = x.to(self.device)
        cosines, sines = positional_embeddings
        forward_outputs = mixer.forward(
            hidden_states=x,
            attention_mask=self._attention_mask(x, self._layer(path)),
            position_embeddings=(cosines.to(self.device), sines.to(self.device)),
        )
        if isinstance(forward_outputs, tuple):
            forward_outputs, *_ = forward_outputs
        return forward_outputs

    def rope(self, mixer: nn.Module, path: ParameterPath, positions: Tensor) -> tuple[Tensor, Tensor] | None:  # noqa: ARG002
        return _rope_forward(
            self.hf_model.model.rotary_emb,
            torch.ones(1, dtype=torch.float32),
            positions,
            self._layer(path).attention_type,
            self.device,
        )

    def readout(self, x: Tensor) -> Tensor:
        x = x.to(self.device)
        return self.hf_model.classifier(self.hf_model.head(x))

    def forward(self, input_ids: Tensor, position_ids: Tensor) -> tuple[tuple[Tensor, ...], Tensor, Tensor]:
        hf_outputs = self.hf_model.forward(
            input_ids=input_ids.to(self.device),
            position_ids=position_ids.to(self.device),
            output_hidden_states=True,
        )
        assert hf_outputs.hidden_states is not None
        assert hf_outputs.logits is not None
        *hf_hidden_states, hf_last_norm_output = hf_outputs.hidden_states
        return (tuple(hf_hidden_states), hf_last_norm_output, hf_outputs.logits)

    @classmethod
    def load(cls, model_repo: str, dtype: DType | None) -> Self:
        hf_model, device = _load_hf_model(AutoModelForSequenceClassification, model_repo, dtype)
        return cls(hf_model, device, _named_modules(hf_model))
