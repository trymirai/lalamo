from dataclasses import asdict
from pathlib import Path

from evals.types import InferenceConfig, InferenceEngineType

from lalamo.commands import generate_replies
from lalamo.evals.inference.engines.base import InferenceEngine
from lalamo.evals.inference.engines.callbacks import BaseEngineCallbacks

# Inference config parameters currently supported by generate_replies()
SUPPORTED_PARAMS = {"max_output_length"}


class LalamoInferenceEngine(InferenceEngine):
    def __init__(
        self,
        model_path: Path,
        inference_config: InferenceConfig,
        max_vram: int | None = None,
        batch_size: int | None = None,
    ):
        self.model_path = model_path
        self.inference_config = inference_config
        self.max_vram = max_vram
        self.batch_size = batch_size

    @property
    def engine_type(self) -> InferenceEngineType:
        return InferenceEngineType.LOCAL

    def _get_conversation_column_name(self) -> str:
        return "conversation"

    def run_inference(
        self,
        input_path: Path,
        output_path: Path,
        callbacks: BaseEngineCallbacks,
    ) -> Path:
        from lalamo.main import CliGenerateRepliesCallbacks

        # Check for unsupported inference config parameters and warn
        config_dict = asdict(self.inference_config)
        unsupported = [
            f"{k}={v}"
            for k, v in config_dict.items()
            if k not in SUPPORTED_PARAMS and v is not None
        ]

        if unsupported:
            callbacks.unsupported_inference_params(unsupported)

        # TODO(mullakhmetov): Add support for remaining inference config parameters:
        # - temperature = self.inference_config.temperature
        # - top_p = self.inference_config.top_p
        # - top_k = self.inference_config.top_k
        # - stop_tokens = self.inference_config.stop_tokens (needs tokenization)
        # - max_model_len = self.inference_config.max_model_len
        #
        # This requires updating generate_replies() signature to accept these parameters.

        max_output_length = self.inference_config.max_output_length
        generate_replies(
            model_path=self.model_path,
            dataset_path=input_path,
            output_path=output_path,
            max_vram=self.max_vram,
            max_output_length=max_output_length,
            batch_size=self.batch_size,
            callbacks_type=CliGenerateRepliesCallbacks,
        )

        return output_path
