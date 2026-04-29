import os

# Must run before importing jax / tensorflow, this hides the XLA optimization logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from lalamo.checkpoint_manager import CheckpointManager
from lalamo.model_import import ModelSpec
from lalamo.model_import.model_spec import (
    ClassifierModelSpec,
    ConfigMap,
    FileSpec,
    JSONFieldSpec,
    LanguageModelSpec,
    TTSModelSpec,
)
from lalamo.models import ClassifierModel, LanguageModel
from lalamo.models.chat_codec import (
    AssistantMessage,
    ChatCodec,
    ChatCodecConfig,
    ContentBlock,
    Image,
    Message,
    SystemMessage,
    ToolSchema,
    UserMessage,
)

__version__ = "0.6.13"

__all__ = [
    "AssistantMessage",
    "ChatCodec",
    "ChatCodecConfig",
    "CheckpointManager",
    "ClassifierModel",
    "ClassifierModelSpec",
    "ConfigMap",
    "ContentBlock",
    "FileSpec",
    "Image",
    "JSONFieldSpec",
    "LanguageModel",
    "LanguageModelSpec",
    "Message",
    "ModelSpec",
    "SystemMessage",
    "TTSModelSpec",
    "ToolSchema",
    "UserMessage",
]
