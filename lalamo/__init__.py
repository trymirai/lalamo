import os

# Must run before importing jax / tensorflow, this hides the XLA optimization logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Persistent JAX compilation cache. Any of these can be overridden by exporting the
# matching env var before importing lalamo.
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", ".jax_cache")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0.01")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")

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
from lalamo.modules.common import ShardingConfig, pad_and_apply_data_sharding
from lalamo.quantization import QuantizationMode
from lalamo.speculator import (
    CollectTracesEvent,
    SpeculatorTrainingEvent,
)

__version__ = "0.6.12"

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
    "ShardingConfig",
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
    "pad_and_apply_data_sharding",
    "train",
]
