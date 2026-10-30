import json
from collections import ChainMap
from collections.abc import Callable, Mapping, MutableMapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Self, cast

import huggingface_hub
from cattrs import Converter
from jaxtyping import Array, DTypeLike

from lalamo.initializer import EmptyInitializer
from lalamo.modules.vision import ImagePreprocessingConfig, VisionEncoder
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.sharding import ShardingConfig
from lalamo.weight_matrix import CompressionImplementation

from .loaders.vision import load_qwen3_5_vision_encoder
from .model_configs.huggingface.qwen3_5_vision import HFQwen35ImageProcessorConfig, HFQwen35VisionConfig
from .origins import FileSpec, StatusEvent, hf_resolve_file, load_safetensors_weights

__all__ = [
    "VISION_WEIGHTS_PATH",
    "ImportedVisionEncoder",
    "import_qwen3_5_vision_encoder",
    "resolve_vision_weight_files",
]

VISION_WEIGHTS_PATH = ParameterPath("model.visual")


class ImportedVisionEncoder(NamedTuple):
    encoder: VisionEncoder
    preprocessing_config: ImagePreprocessingConfig


@dataclass(frozen=True)
class SafetensorsIndex:
    weight_map: dict[str, str]

    @classmethod
    def from_json(cls, json_object: object) -> Self:
        return Converter().structure(json_object, cls)

    def shards_with_prefix(self, prefix: ParameterPath) -> tuple[str, ...]:
        return tuple(sorted({shard for key, shard in self.weight_map.items() if key.startswith(f"{prefix}.")}))


def resolve_vision_weight_files(
    repo: str,
    progress_callback: Callable[[StatusEvent], None] | None = None,
) -> tuple[Path, ...]:
    """Download (or find in the HF cache) only the safetensors files that hold vision weights."""
    single_file = "model.safetensors"
    index_file = f"{single_file}.index.json"
    if index_file in huggingface_hub.list_repo_files(repo):
        with hf_resolve_file(repo, FileSpec(index_file), progress_callback).open() as index:
            filenames = SafetensorsIndex.from_json(json.load(index)).shards_with_prefix(VISION_WEIGHTS_PATH)
        if not filenames:
            raise ValueError(f"{repo} has no weights under {VISION_WEIGHTS_PATH}.")
    else:
        filenames = (single_file,)
    return tuple(hf_resolve_file(repo, FileSpec(filename), progress_callback) for filename in filenames)


def import_qwen3_5_vision_encoder(
    repo: str,
    *,
    sharding_config: ShardingConfig,
    dtype: DTypeLike | None = None,
    progress_callback: Callable[[StatusEvent], None] | None = None,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> ImportedVisionEncoder:
    """Import only the vision tower of a Qwen3.5 checkpoint from the HuggingFace Hub.

    Only the safetensors shards that contain ``model.visual.*`` are downloaded. For Qwen3.5-27B
    that is the last of eleven shards, so the 50+ GB text weights are never fetched.
    """
    vision_config = HFQwen35VisionConfig.from_model_config_json(
        hf_resolve_file(repo, FileSpec("config.json"), progress_callback),
    )
    preprocessing_config = HFQwen35ImageProcessorConfig.from_json(
        hf_resolve_file(repo, FileSpec("preprocessor_config.json"), progress_callback),
    ).to_image_preprocessing_config()

    encoder_config = vision_config.to_vision_encoder_config()
    if preprocessing_config.spatial_merge_size != encoder_config.spatial_merge_size:
        raise ValueError("Preprocessor and vision tower disagree on the spatial merge size.")
    if preprocessing_config.patch_size != encoder_config.patch_embedding_config.patch_size:
        raise ValueError("Preprocessor and vision tower disagree on the patch size.")

    with ExitStack() as stack:
        shards = [
            stack.enter_context(load_safetensors_weights(path))
            for path in resolve_vision_weight_files(repo, progress_callback)
        ]
        # Shards stay lazy: only the vision tensors are ever read from disk.
        weights_dict: Mapping[str, Array] = ChainMap(
            *(cast("MutableMapping[str, Array]", weights) for weights, _metadata in shards),
        )
        template = encoder_config.init(EmptyInitializer(default_dtype=dtype, sharding_config=sharding_config))
        encoder = load_qwen3_5_vision_encoder(
            template,
            weights_dict,
            VISION_WEIGHTS_PATH,
            implementation=implementation,
        )
    return ImportedVisionEncoder(encoder=encoder, preprocessing_config=preprocessing_config)
