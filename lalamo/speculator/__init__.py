from .common import Speculator
from .inference import inference_collect_traces
from .ngram import NGramSpeculator
from .utils import train_speculator

__all__ = [
    "NGramSpeculator",
    "Speculator",
    "inference_collect_traces",
    "train_speculator",
]
