import json
import warnings
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import jax.numpy as jnp
from jaxtyping import DTypeLike
from tokenizers import Tokenizer

from lalamo.exportable import Exportable, ExportResults
from lalamo.initializer import EmptyInitializer, Initializer
from lalamo.module import LalamoConfig, LalamoModule, field
from lalamo.safetensors import safe_read, safe_write
from lalamo.token_codec import TokenCodec, TokenCodecConfig
from lalamo.utils.registry_abc import RegistryABC
from lalamo.utils.sharding import ShardingConfig

__all__ = [
    "BaseModel",
    "BaseModelConfig",
    "Model",
    "ModelConfig",
]


@dataclass(frozen=True)
class BaseModelConfig(LalamoConfig, RegistryABC):
    @abstractmethod
    def init_from_directory(self, directory: Path, initializer: Initializer) -> "BaseModel": ...


class BaseModel[ConfigT: BaseModelConfig](LalamoModule[ConfigT]):
    def save(self, directory: Path | str) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        exported_model = Exportable.export(self)
        metadata = None
        if exported_model.metadata:
            metadata = {key: json.dumps(value) for key, value in exported_model.metadata.items()}

        with (directory / "model.safetensors").open("wb") as weights_file:
            safe_write(
                weights_file,
                exported_model.arrays,
                metadata=metadata,
            )

        with (directory / "config.json").open("w") as config_file:
            json.dump(self.config.to_json(), config_file, indent=4)

    @classmethod
    def load(cls, directory: Path | str, sharding_config: ShardingConfig, dtype: DTypeLike | None = None) -> Self:
        from lalamo.model_import.loaders import packed_checkpoint  # noqa: PLC0415

        directory = Path(directory)
        with (directory / "config.json").open() as config_file:
            config_json = json.load(config_file)

        with (directory / "model.safetensors").open("rb") as weights_file:
            metadata, arrays, tensor_infos = safe_read(weights_file)
            decoded_metadata = {}
            if metadata is not None:
                decoded_metadata = {key: json.loads(value) for key, value in metadata.items()}
            is_packed = packed_checkpoint.is_packed_checkpoint(decoded_metadata, arrays.keys())
            if is_packed:
                config_json = packed_checkpoint.native_config(config_json)
            config = BaseModelConfig.from_json(config_json)
            fp32_scales = sum(
                tensor_infos[name].dtype == jnp.float32
                for name in tensor_infos
                if name.endswith(".scales") and name.removesuffix(".scales") + ".spec" in decoded_metadata
            )
            template = config.init_from_directory(directory, EmptyInitializer(dtype, sharding_config))
            exported = ExportResults(arrays, decoded_metadata)
            if is_packed:
                result = packed_checkpoint.load_packed_checkpoint(template, exported, dtype)
            else:
                result = Exportable.load_exported(template, exported)

        assert isinstance(result, cls)
        if fp32_scales:
            warnings.warn(
                f"Loaded {fp32_scales} quantization scale tensors stored in FP32; use BF16 scales.",
                UserWarning,
                stacklevel=2,
            )
        return result


@dataclass(frozen=True)
class ModelConfig[TokenCodecConfigT: TokenCodecConfig](BaseModelConfig):
    token_codec_config: TokenCodecConfigT

    @abstractmethod
    def init(self, tokenizer: Tokenizer, initializer: Initializer) -> "Model": ...

    def init_from_directory(self, directory: Path, initializer: Initializer) -> "Model":
        tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        return self.init(tokenizer, initializer)


class Model[
    ConfigT: ModelConfig,
    TokenCodecT: TokenCodec,
](BaseModel[ConfigT]):
    token_codec: TokenCodecT = field(static=True)

    def save(self, directory: Path | str) -> None:
        super().save(directory)
        self.token_codec.tokenizer.save(str(Path(directory) / "tokenizer.json"))
