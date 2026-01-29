import os

# Must run before importing jax / tensorflow, this hides the XLA optimization logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from lalamo.commands import (
    CollectTracesCallbacks,
    ConversionCallbacks,
    EstimateBatchsizeCallbacks,
    Precision,
    TrainCallbacks,
    collect_traces,
    convert,
    estimate_batchsize,
    train,
)
from lalamo.message_processor import (
    AssistantMessage,
    ContentBlock,
    Image,
    Message,
    SystemMessage,
    ToolSchema,
    UserMessage,
)
from lalamo.model_import import ModelSpec, import_model
from lalamo.model_import.model_specs.common import ConfigMap, FileSpec, JSONFieldSpec, ModelType, UseCase, WeightsType
from lalamo.models import ClassifierModel, LanguageModel
from lalamo.quantization import QuantizationMode
from lalamo.speculator import (
    CollectTracesEvent,
    SpeculatorTrainingEvent,
)

__version__ = "0.6.5"

__all__ = [
    "AssistantMessage",
    "ClassifierModel",
    "CollectTracesCallbacks",
    "CollectTracesEvent",
    "ConfigMap",
    "ContentBlock",
    "ConversionCallbacks",
    "EstimateBatchsizeCallbacks",
    "FileSpec",
    "Image",
    "JSONFieldSpec",
    "LanguageModel",
    "Message",
    "ModelSpec",
    "ModelType",
    "Precision",
    "QuantizationMode",
    "SpeculatorTrainingEvent",
    "SystemMessage",
    "ToolSchema",
    "TrainCallbacks",
    "UseCase",
    "UserMessage",
    "WeightsType",
    "collect_traces",
    "convert",
    "estimate_batchsize",
    "import_model",
    "train",
]
