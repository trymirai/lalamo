import os

# Must run before importing jax / tensorflow, this hides the XLA optimization logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from lalamo.checkpoint_manager import CheckpointManager
from lalamo.model_import import ModelSpec, import_model
from lalamo.model_import.model_specs.common import ConfigMap, FileSpec, JSONFieldSpec, ModelType, UseCase, WeightsType
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
    "ConfigMap",
    "ContentBlock",
    "FileSpec",
    "Image",
    "JSONFieldSpec",
    "LanguageModel",
    "Message",
    "ModelSpec",
    "ModelType",
    "SystemMessage",
    "ToolSchema",
    "UseCase",
    "UserMessage",
    "WeightsType",
    "import_model",
]
