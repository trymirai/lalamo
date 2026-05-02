import json
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

__all__ = [
    "Model",
    "ModelConfig",
]


@dataclass(frozen=True)
class ModelConfig[TokenCodecConfigT: TokenCodecConfig](LalamoConfig, RegistryABC):
    token_codec_config: TokenCodecConfigT

    @abstractmethod
    def init(self, tokenizer: Tokenizer, initializer: Initializer) -> "Model": ...


class Model[
    TokenCodecConfigT: TokenCodecConfig,
    ConfigT: ModelConfig,
    TokenCodecT: TokenCodec,
](LalamoModule[ConfigT]):
    token_codec: TokenCodecT = field(static=True)

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

        self.token_codec.tokenizer.save(str(directory / "tokenizer.json"))

    @classmethod
    def load(cls, directory: Path | str, dtype: DTypeLike = jnp.bfloat16) -> Self:
        directory = Path(directory)
        with (directory / "config.json").open() as config_file:
            config = ModelConfig.from_json(json.load(config_file))
        tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))

        with (directory / "model.safetensors").open("rb") as weights_file:
            metadata, arrays = safe_read(weights_file)
            decoded_metadata = {}
            if metadata is not None:
                decoded_metadata = {key: json.loads(value) for key, value in metadata.items()}

            template = config.init(tokenizer, EmptyInitializer(dtype))
            result = Exportable.load_exported(
                template,
                ExportResults(
                    arrays=arrays,
                    metadata=decoded_metadata,
                ),
                allow_dtype_cast=True,
            )

        assert isinstance(result, cls)
        return result
