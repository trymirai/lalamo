from typing import Union

from fartsovka.language_model import (
    LanguageModelConfig,
    MessageFormatSpec,
    MessageFormatType,
)
from fartsovka.modules.common import config_converter, register_config_union
from fartsovka.tokenizer import HFTokenizerConfig, TokenizerConfig


# Register the tokenizer configs union for de/serialization
TokenizerConfigUnion = Union[TokenizerConfig, HFTokenizerConfig]
register_config_union(TokenizerConfigUnion)

# Register message format type enum for de/serialization
config_converter.register_unstructure_hook(
    MessageFormatType,
    lambda o: o.name,
)
config_converter.register_structure_hook(
    MessageFormatType,
    lambda s, _: MessageFormatType[s],
)

# Register message format spec for de/serialization
config_converter.register_unstructure_hook(
    MessageFormatSpec,
    lambda o: {
        "system_template": o.system_template,
        "user_template": o.user_template,
        "assistant_template": o.assistant_template,
        "system_token_ids": o.system_token_ids,
        "user_token_ids": o.user_token_ids,
        "assistant_token_ids": o.assistant_token_ids,
    },
)
config_converter.register_structure_hook(
    MessageFormatSpec,
    lambda d, _: MessageFormatSpec(**d),
)