from pathlib import Path

import polars as pl
from evals.types import InferenceConfig, InferenceEngineType
from openai import OpenAI

from lalamo.evals.inference.engines.base import InferenceEngine
from lalamo.evals.inference.engines.callbacks import BaseEngineCallbacks
from lalamo.evals.inference.engines.custom_api.config import CustomAPIEngineConfig

_SUPPORTED_PARAMS = {"temperature", "max_output_length", "top_p", "stop_tokens"}


class CustomAPIInferenceEngine(InferenceEngine):
    def __init__(
        self,
        config: CustomAPIEngineConfig,
        inference_config: InferenceConfig,
    ) -> None:
        self.base_url = config.base_url
        self.model = config.model
        self.inference_config = inference_config
        self.api_key = config.api_key
        self.timeout = config.timeout
        self.max_retries = config.max_retries

        self._client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key or "dummy-key-for-local-server",
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

    def run_inference(
        self,
        input_path: Path,
        output_path: Path,
        callbacks: BaseEngineCallbacks,
    ) -> Path:
        unsupported = self._check_unsupported_params(self.inference_config, _SUPPORTED_PARAMS)
        if unsupported:
            callbacks.unsupported_inference_params(unsupported)

        input_df = pl.read_parquet(input_path)
        conversations = input_df["messages"].to_list()

        callbacks.generation_started(len(conversations))

        responses = []
        for i, messages in enumerate(conversations):
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.inference_config.temperature,
                max_tokens=self.inference_config.max_output_length,
                top_p=self.inference_config.top_p,
                stop=self.inference_config.stop_tokens,
                # Note: top_k not supported by OpenAI API
            )

            content = response.choices[0].message.content or ""
            responses.append(content)

            callbacks.generation_progress(i + 1)

        callbacks.finished_generation()

        output_data = pl.DataFrame({
            "response": responses,
            "chain_of_thought": [""] * len(responses),
        })

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_data.write_parquet(output_path)
        return output_path
