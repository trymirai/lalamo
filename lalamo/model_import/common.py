import importlib.metadata
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import huggingface_hub
import jax.numpy as jnp
from jaxtyping import DTypeLike
from tokenizers import Tokenizer

from lalamo.language_model import GenerationConfig, LanguageModel, LanguageModelConfig
from lalamo.message_processor import MessageProcessor, MessageProcessorConfig
from lalamo.quantization import QuantizationMode

from .huggingface_generation_config import HFGenerationConfig
from .huggingface_tokenizer_config import HFTokenizerConfig
from .model_specs import REPO_TO_MODEL, FileSpec, ModelSpec, UseCase

__all__ = [
    "REPO_TO_MODEL",
    "ModelMetadata",
    "ModelSpec",
    "import_model",
]


LALAMO_VERSION = importlib.metadata.version("lalamo")


@dataclass(frozen=True)
class ModelMetadata:
    toolchain_version: str
    vendor: str
    family: str
    name: str
    size: str
    quantization: QuantizationMode | None
    repo: str
    use_cases: tuple[UseCase, ...]
    model_config: LanguageModelConfig


def download_file(file_spec: FileSpec, model_repo: str, output_dir: Path | str | None = None) -> Path:
    result = huggingface_hub.hf_hub_download(
        repo_id=file_spec.repo or model_repo,
        local_dir=output_dir,
        filename=file_spec.filename,
    )
    return Path(result)


def list_weight_files(model_repo: str) -> list[FileSpec]:
    all_files = huggingface_hub.list_repo_files(model_repo)
    return [FileSpec(filename) for filename in all_files if filename.endswith(".safetensors")]


def download_weights(model_spec: ModelSpec, output_dir: Path | str | None = None) -> list[Path]:
    return [download_file(file_spec, model_spec.repo, output_dir) for file_spec in list_weight_files(model_spec.repo)]


def download_config_file(model_spec: ModelSpec, output_dir: Path | str | None = None) -> Path:
    return download_file(model_spec.configs.model_config, model_spec.repo, output_dir)


class ImportResults(NamedTuple):
    model: LanguageModel
    metadata: ModelMetadata


def import_message_processor(model_spec: ModelSpec, output_dir: Path | str | None = None) -> MessageProcessor:
    tokenizer_file = download_file(model_spec.configs.tokenizer, model_spec.repo, output_dir)
    tokenizer_config_file = download_file(model_spec.configs.tokenizer_config, model_spec.repo, output_dir)
    tokenizer_config = HFTokenizerConfig.from_json(tokenizer_config_file)
    if tokenizer_config.chat_template is None:
        if model_spec.configs.chat_template is None:
            raise ValueError("Missiing chat template.")
        chat_template_file = download_file(model_spec.configs.chat_template, model_spec.repo, output_dir)
        prompt_template = chat_template_file.read_text()
    else:
        if model_spec.configs.chat_template is not None:
            raise ValueError("Conflicting chat template specifications.")
        prompt_template = tokenizer_config.chat_template
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    tokenizer.add_special_tokens(tokenizer_config.added_tokens())
    message_processor_config = MessageProcessorConfig(
        prompt_template=prompt_template,
        output_parser_regex=model_spec.output_parser_regex,
        system_role_name=model_spec.system_role_name,
        user_role_name=model_spec.user_role_name,
        assistant_role_name=model_spec.assistant_role_name,
        bos_token=tokenizer_config.bos_token,
    )
    return MessageProcessor(config=message_processor_config, tokenizer=tokenizer)


def import_model(
    model_spec: ModelSpec,
    *,
    context_length: int | None = None,
    precision: DTypeLike | None = None,
    accumulation_precision: DTypeLike = jnp.float32,
) -> ImportResults:
    foreign_decoder_config_file = download_config_file(model_spec)
    foreign_decoder_config = model_spec.config_type.from_json(foreign_decoder_config_file)

    if precision is None:
        precision = foreign_decoder_config.default_precision

    weights_paths = download_weights(model_spec)
    weights_dict = {}
    for weights_path in weights_paths:
        weights_dict.update(model_spec.weights_type.load(weights_path, precision))

    decoder = foreign_decoder_config.load_decoder(context_length, precision, accumulation_precision, weights_dict)
    message_processor = import_message_processor(model_spec)

    stop_token_ids = tuple(foreign_decoder_config.eos_token_ids)

    if model_spec.configs.generation_config is not None:
        hf_generation_config_file = download_file(model_spec.configs.generation_config, model_spec.repo)
        hf_generation_config = HFGenerationConfig.from_json(hf_generation_config_file)
        generation_config = GenerationConfig(
            stop_token_ids=stop_token_ids,
            temperature=hf_generation_config.temperature,
            top_p=hf_generation_config.top_p,
            top_k=hf_generation_config.top_k,
            banned_tokens=None,
        )
    else:
        generation_config = GenerationConfig(
            stop_token_ids=stop_token_ids,
            temperature=None,
            top_p=None,
            top_k=None,
            banned_tokens=None,
        )

    language_model_config = LanguageModelConfig(
        decoder_config=decoder.config,
        message_processor_config=message_processor.config,
        generation_config=generation_config,
    )

    language_model = LanguageModel(language_model_config, decoder, message_processor)
    metadata = ModelMetadata(
        toolchain_version=LALAMO_VERSION,
        vendor=model_spec.vendor,
        family=model_spec.family,
        name=model_spec.name,
        size=model_spec.size,
        quantization=model_spec.quantization,
        repo=model_spec.repo,
        use_cases=model_spec.use_cases,
        model_config=language_model_config,
    )
    return ImportResults(language_model, metadata)
