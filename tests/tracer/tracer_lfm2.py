from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM
from transformers.models.lfm2.modeling_lfm2 import Lfm2Attention

from lalamo.utils.parameter_path import ParameterPath
from tests.tracer.tracer import DType
from tests.tracer.tracer_huggingface import HFDecoderTracer, _build_hf_attention_mask, _load_hf_model, _named_modules


@dataclass(frozen=True)
class LFM2DecoderTracer(HFDecoderTracer):
    def mixer(
        self,
        mixer: nn.Module,
        x: Tensor,
        positional_embeddings: tuple[Tensor, Tensor] | None,
        path: ParameterPath,  # noqa: ARG002
    ) -> Tensor:
        x = x.to(self.device)
        attention_mask = _build_hf_attention_mask(x, mixer)
        if isinstance(mixer, Lfm2Attention):
            assert positional_embeddings is not None
            cosines, sines = positional_embeddings
            attention_output, _ = mixer.forward(
                hidden_states=x,
                position_embeddings=(cosines.to(self.device), sines.to(self.device)),
                attention_mask=attention_mask,
            )
            return attention_output
        return mixer.forward(hidden_states=x, attention_mask=attention_mask)

    def rope(self, mixer: nn.Module, path: ParameterPath, positions: Tensor) -> tuple[Tensor, Tensor] | None:  # noqa: ARG002
        return self.hf_model.model.rotary_emb.forward(torch.zeros((), dtype=torch.float32), positions)

    def forward(self, input_ids: Tensor, position_ids: Tensor) -> tuple[tuple[Tensor, ...], Tensor, Tensor]:
        return super().forward(input_ids.long(), position_ids.long())

    @classmethod
    def load(cls, model_repo: str, dtype: DType | None) -> Self:
        hf_model, device = _load_hf_model(AutoModelForCausalLM, model_repo, dtype)
        return cls(hf_model, device, _named_modules(hf_model))
