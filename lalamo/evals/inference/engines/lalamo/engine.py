from pathlib import Path

from evals.types import InferenceConfig

from lalamo.commands import generate_replies
from lalamo.common import get_vram_limit_bytes
from lalamo.evals.inference.engines.base import InferenceEngine
from lalamo.evals.inference.engines.callbacks import BaseEngineCallbacks
from lalamo.evals.inference.engines.lalamo.config import LalamoEngineConfig
from lalamo.models.language_model import GenerationConfig

_SUPPORTED_PARAMS = {"max_output_length", "temperature", "top_k", "top_p"}


class LalamoInferenceEngine(InferenceEngine):
    def __init__(
        self,
        config: LalamoEngineConfig,
        inference_config: InferenceConfig,
    ) -> None:
        self.model_path = config.model_path
        self.batch_size = config.batch_size
        self.inference_config = inference_config

        if config.batch_size is None:
            self.max_vram = get_vram_limit_bytes(config.vram_gb)
            if self.max_vram is None:
                raise ValueError(
                    "Cannot get default device's memory stats. "
                    "Specify batch-size or vram-gb",
                )
        else:
            self.max_vram = None

    def get_conversation_column_name(self) -> str:
        return "conversation"

    def run_inference(
        self,
        input_path: Path,
        output_path: Path,
        callbacks: BaseEngineCallbacks,
    ) -> Path:
        # have to import here to avoid circular import issues, since command handlers also use generate_replies
        from lalamo.main import CliGenerateRepliesCallbacks

        unsupported = self._check_unsupported_params(self.inference_config, _SUPPORTED_PARAMS)
        if unsupported:
            callbacks.unsupported_inference_params(unsupported)

        max_output_length = self.inference_config.max_output_length
        assert max_output_length is not None, "Adapter must provide max_output_length"

        # Note: stop_token_ids must not be set - generate_replies handles this automatically
        generation_config = GenerationConfig(
            temperature=self.inference_config.temperature,
            top_k=self.inference_config.top_k,
            top_p=self.inference_config.top_p,
        )

        generate_replies(
            model_path=self.model_path,
            dataset_path=input_path,
            output_path=output_path,
            max_vram=self.max_vram,
            max_output_length=max_output_length,
            batch_size=self.batch_size,
            generation_config_override=generation_config,
            callbacks_type=CliGenerateRepliesCallbacks,
        )

        return output_path
