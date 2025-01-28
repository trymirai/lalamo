from dataclasses import dataclass

from jaxtyping import PRNGKeyArray

from fartsovka.modules.common import ModuleConfig
from fartsovka.modules.decoder import Decoder

__all__ = ["AbstractModelConfig"]


@dataclass
class AbstractModelConfig[DecoderType: Decoder](ModuleConfig[DecoderType]):
    def __call__(self, key: PRNGKeyArray) -> DecoderType:
        raise NotImplementedError
