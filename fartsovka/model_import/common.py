import importlib.metadata
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import huggingface_hub
import jax.numpy as jnp
from jaxtyping import DTypeLike

from fartsovka.modules import Decoder, DecoderConfig

from .model_specs import REPO_TO_MODEL, ModelSpec

__all__ = [
    "REPO_TO_MODEL",
    "ModelMetadata",
    "ModelSpec",
    "import_model",
]


FARTSOVKA_VERSION = importlib.metadata.version("fartsovka")


@dataclass(frozen=True)
class ModelMetadata:
    fartsovka_version: str
    vendor: str
    name: str
    model_config: DecoderConfig
    tokenizer_file_names: tuple[str, ...]


def download_weights(model_spec: ModelSpec, output_dir: Path | str | None = None) -> list[Path]:
    result = [
        huggingface_hub.hf_hub_download(
            repo_id=model_spec.repo,
            local_dir=output_dir,
            filename=filename,
        )
        for filename in model_spec.weights_file_names
    ]
    return [Path(path) for path in result]


def download_config_file(model_spec: ModelSpec, output_dir: Path | str | None = None) -> Path:
    result = huggingface_hub.hf_hub_download(
        repo_id=model_spec.repo,
        local_dir=output_dir,
        filename=model_spec.config_file_name,
    )
    return Path(result)


def download_tokenizer_files(model_spec: ModelSpec, output_dir: Path | str | None = None) -> tuple[Path, ...]:
    result = [
        huggingface_hub.hf_hub_download(
            repo_id=model_spec.repo,
            local_dir=output_dir,
            filename=metadata_filename,
        )
        for metadata_filename in model_spec.tokenizer_file_names
    ]
    return tuple(Path(path) for path in result)


class ImportResults(NamedTuple):
    model: Decoder
    metadata: ModelMetadata
    tokenizer_file_paths: tuple[Path, ...]


def import_model(
    model_spec: ModelSpec,
    *,
    context_length: int = 8192,
    precision: DTypeLike | None = None,
    accumulation_precision: DTypeLike = jnp.float32,
) -> ImportResults:
    foreign_config_file = download_config_file(model_spec)
    foreign_config = model_spec.config_type.from_json(foreign_config_file)

    tokenizer_file_paths = download_tokenizer_files(model_spec)
    if precision is None:
        precision = foreign_config.default_precision

    weights_paths = download_weights(model_spec)
    weights_dict = {}
    for weights_path in weights_paths:
        weights_dict.update(model_spec.weights_type.load(weights_path, precision))

    model = foreign_config.load_model(context_length, precision, accumulation_precision, weights_dict)
    metadata = ModelMetadata(
        fartsovka_version=FARTSOVKA_VERSION,
        name=model_spec.name,
        vendor=model_spec.vendor,
        model_config=model.config,
        tokenizer_file_names=tuple(p.name for p in tokenizer_file_paths),
    )
    return ImportResults(model, metadata, tokenizer_file_paths)
