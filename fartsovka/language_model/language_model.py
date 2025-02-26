from dataclasses import dataclass
from enum import Enum, auto
from typing import NamedTuple, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from fartsovka.common import ParameterDict
from fartsovka.modules import Decoder, DecoderConfig, DecoderOutput
from fartsovka.modules.common import FartsovkaModule
from fartsovka.tokenizer import Tokenizer, TokenizerConfig


class MessageFormatType(Enum):
    """Type of message format for the language model."""
    PLAIN = auto()
    LLAMA = auto()
    GEMMA = auto()
    QWEN = auto()
    CUSTOM = auto()


class MessageFormatSpec(NamedTuple):
    """Specification for message formatting."""
    system_template: str | None = None
    user_template: str = "{content}"
    assistant_template: str = "{content}"
    system_token_ids: Sequence[int] | None = None
    user_token_ids: Sequence[int] | None = None
    assistant_token_ids: Sequence[int] | None = None


@dataclass
class LanguageModelConfig:
    """Configuration for a language model."""
    decoder_config: DecoderConfig
    tokenizer_config: TokenizerConfig
    message_format_type: MessageFormatType
    message_format_spec: MessageFormatSpec
    model_name: str
    
    @classmethod
    def get_default_message_format_spec(cls, format_type: MessageFormatType) -> MessageFormatSpec:
        """Get default message format specification for a format type."""
        if format_type == MessageFormatType.PLAIN:
            return MessageFormatSpec()
        
        elif format_type == MessageFormatType.LLAMA:
            return MessageFormatSpec(
                system_template="<|system|>\n{content}</s>",
                user_template="<|user|>\n{content}</s>",
                assistant_template="<|assistant|>\n{content}</s>",
            )
        
        elif format_type == MessageFormatType.GEMMA:
            return MessageFormatSpec(
                system_template=None,  # Gemma doesn't support system messages by default
                user_template="<start_of_turn>user\n{content}<end_of_turn>\n",
                assistant_template="<start_of_turn>model\n{content}<end_of_turn>\n",
            )
        
        elif format_type == MessageFormatType.QWEN:
            return MessageFormatSpec(
                system_template="<|im_start|>system\n{content}<|im_end|>\n",
                user_template="<|im_start|>user\n{content}<|im_end|>\n",
                assistant_template="<|im_start|>assistant\n{content}<|im_end|>\n",
            )
        
        else:
            # Custom format requires explicit specification
            raise ValueError(f"Custom message format requires explicit MessageFormatSpec")


class LanguageModel(FartsovkaModule[LanguageModelConfig]):
    """Language model that combines a decoder and tokenizer."""
    decoder: Decoder
    tokenizer: Tokenizer
    
    def __init__(
        self, 
        config: LanguageModelConfig,
        decoder: Decoder,
        tokenizer: Tokenizer,
    ):
        self.config = config
        self.decoder = decoder
        self.tokenizer = tokenizer
    
    def __call__(
        self,
        token_ids: Int[Array, " suffix_tokens"],
        token_positions: Int[Array, " suffix_tokens"],
        kv_cache: list[Array] | None = None,
        mask: Bool[Array, "suffix_tokens total_tokens"] | None = None,
        return_updated_kv_cache: bool = False,
    ) -> DecoderOutput:
        """Forward pass of the language model."""
        return self.decoder(
            token_ids=token_ids,
            token_positions=token_positions,
            kv_cache=kv_cache,
            mask=mask,
            return_updated_kv_cache=return_updated_kv_cache,
        )
    
    def format_messages(
        self, 
        messages: list[dict[str, str]]
    ) -> str:
        """Format a list of messages according to the model's message format."""
        result = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system" and self.config.message_format_spec.system_template:
                result.append(self.config.message_format_spec.system_template.format(content=content))
            elif role == "user":
                result.append(self.config.message_format_spec.user_template.format(content=content))
            elif role == "assistant":
                result.append(self.config.message_format_spec.assistant_template.format(content=content))
        
        return "".join(result)
    
    def export_weights(self) -> ParameterDict:
        """Export model weights."""
        return ParameterDict(
            decoder=self.decoder.export_weights(),
            tokenizer=self.tokenizer.export_weights(),
        )
    
    @classmethod
    def random_init(
        cls, 
        config: LanguageModelConfig, 
        *,
        key: PRNGKeyArray,
    ) -> "LanguageModel":
        """Initialize a language model with random weights."""
        decoder = config.decoder_config.random_init(key=key)
        
        # For tokenizer, we would typically load it from a file rather than randomly initialize
        # This is a placeholder that would need to be replaced with actual tokenizer loading
        tokenizer = Tokenizer.from_config(config.tokenizer_config)
        
        return cls(
            config=config,
            decoder=decoder,
            tokenizer=tokenizer,
        )