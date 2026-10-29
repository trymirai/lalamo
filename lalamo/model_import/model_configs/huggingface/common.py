from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, Literal

import cattrs
import equinox as eqx
from cattrs.strategies import configure_tagged_union
from jaxtyping import Array

from lalamo.model import Model
from lalamo.model_import.loaders import (
    huggingface_loader,
    load_huggingface_classifier,
    load_huggingface_decoder,
)
from lalamo.model_import.loaders.binding import Binder
from lalamo.model_import.model_configs import ForeignClassifierConfig, ForeignLMConfig
from lalamo.models import ClassifierModel, LanguageModel
from lalamo.modules.decoder import DecoderConfig
from lalamo.modules.embedding import EmbeddingConfig
from lalamo.modules.flow import EmitConfig, SequenceConfig, StepConfig, residual
from lalamo.modules.normalization import NormalizationConfig
from lalamo.utils.parameter_path import ParameterPath
from lalamo.weight_matrix import CompressionImplementation

__all__ = [
    "AWQQuantizationConfig",
    "GPTQMetaConfig",
    "GPTQQuantizationConfig",
    "HuggingFaceClassifierConfig",
    "HuggingFaceLMConfig",
    "residual_decoder",
]


def residual_decoder(
    at: Binder,
    *,
    embedding: EmbeddingConfig,
    layers: Sequence[tuple[StepConfig, StepConfig]],
    norm: NormalizationConfig,
    model_dim: int,
    vocab_size: int,
    base: str = "model",
    mixer_norm: str = "input_layernorm",
    mlp_norm: str = "post_attention_layernorm",
    post_mixer_norm: str | None = None,
    post_mlp_norm: str | None = None,
    output_norm: str = "norm",
    output_norm_config: NormalizationConfig | None = None,
) -> DecoderConfig:
    """The common checkpoint layout `x += post_norm(mixer(norm(x))); x += post_norm(mlp(norm(x)))` with a final norm.

    `layers` holds the already bound mixer and MLP of every checkpoint layer; a tap marks each layer's output.
    Post norms are only present when their checkpoint names are given; the final norm defaults to the block norm.
    """
    steps: list[StepConfig] = []
    taps: list[EmitConfig] = []
    for index, (mixer, mlp) in enumerate(layers):
        path = ParameterPath(base) / "layers" / index
        tap = EmitConfig(f"hidden_states.{index}")
        mixer_branch = [at(path / mixer_norm, norm), mixer]
        if post_mixer_norm is not None:
            mixer_branch.append(at(path / post_mixer_norm, norm))
        mlp_branch = [at(path / mlp_norm, norm), mlp]
        if post_mlp_norm is not None:
            mlp_branch.append(at(path / post_mlp_norm, norm))
        steps += [residual(*mixer_branch), residual(*mlp_branch), tap]
        taps.append(tap)
    steps.append(at(ParameterPath(base) / output_norm, output_norm_config or norm))
    return DecoderConfig(embedding, SequenceConfig(tuple(steps)), model_dim, vocab_size, tuple(taps))


@dataclass(frozen=True)
class AWQQuantizationConfig:
    backend: Literal["autoawq"] = "autoawq"
    bits: Literal[4, 8] = 4
    do_fuse: Literal[False] = False
    exllama_config: None = None
    fuse_max_seq_len: None = None
    group_size: int = 128
    modules_to_fuse: None = None
    modules_to_not_convert: None = None
    quant_method: Literal["awq"] = "awq"
    version: Literal["gemm"] = "gemm"
    zero_point: bool = True


@dataclass(frozen=True)
class GPTQMetaConfig:
    damp_auto_increment: float
    damp_percent: float
    mse: float
    quantizer: list[str]
    static_groups: bool
    true_sequential: bool
    uri: str


@dataclass(frozen=True)
class GPTQQuantizationConfig:
    bits: int
    checkpoint_format: str
    desc_act: bool
    group_size: int
    lm_head: bool
    meta: GPTQMetaConfig
    pack_dtype: str
    quant_method: Literal["gptq"]
    sym: bool


@dataclass(frozen=True)
class MLXQuantizationConfig:
    group_size: int
    bits: int


type QuantizationConfig = AWQQuantizationConfig | GPTQQuantizationConfig | MLXQuantizationConfig
type QuantizationConfigType = QuantizationConfig | None


def _quantization_tag(config_type: type) -> str:
    return {
        AWQQuantizationConfig: "awq",
        GPTQQuantizationConfig: "gptq",
        MLXQuantizationConfig: "mlx",
    }[config_type]


@dataclass(frozen=True)
class HuggingFaceLMConfig(ForeignLMConfig):
    _converter: ClassVar[cattrs.Converter] = cattrs.Converter()
    _converter.register_structure_hook(int | list[int], lambda v, _: v)
    configure_tagged_union(
        QuantizationConfig,
        _converter,
        tag_name="quant_method",
        tag_generator=_quantization_tag,
        default=MLXQuantizationConfig,
    )

    @property
    def eos_token_ids(self) -> list[int]:
        result = getattr(self, "eos_token_id", None)
        if result is None:
            raise RuntimeError("model doesn't have eos_token_id, override eos_token_ids in model config")

        if isinstance(result, int):
            result = [result]

        return result

    def _load_weights(
        self,
        model: Model,
        weights_dict: Mapping[str, Array],
        *,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
    ) -> Model:
        assert isinstance(model, LanguageModel)
        binder = Binder(huggingface_loader)
        config = self.build_decoder(binder)
        decoder = load_huggingface_decoder(model.decoder, config, binder, weights_dict, implementation=implementation)
        return eqx.tree_at(lambda m: (m.decoder,), model, (decoder,))


@dataclass(frozen=True)
class HuggingFaceClassifierConfig(ForeignClassifierConfig):
    def _load_weights(
        self,
        model: Model,
        weights_dict: Mapping[str, Array],
        *,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
    ) -> Model:
        assert isinstance(model, ClassifierModel)
        binder = Binder(huggingface_loader)
        config = self.build_classifier(binder)
        classifier = load_huggingface_classifier(
            model.classifier,
            config,
            binder,
            weights_dict,
            implementation=implementation,
        )
        return eqx.tree_at(lambda m: (m.classifier,), model, (classifier,))
