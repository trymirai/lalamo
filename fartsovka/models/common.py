from dataclasses import dataclass

from jaxtyping import PRNGKeyArray

from fartsovka.modules.decoder import Decoder

__all__ = ["AbstractModelConfig"]


@dataclass
class AbstractModelConfig[DecoderType: Decoder]:
    def __call__(self, key: PRNGKeyArray) -> DecoderType:
        raise NotImplementedError
