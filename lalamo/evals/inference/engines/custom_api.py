from pathlib import Path

import polars as pl
from evals.types import InferenceConfig, InferenceEngineType
from openai import OpenAI

from lalamo.evals.inference.engines.base import InferenceEngine
from lalamo.evals.inference.engines.callbacks import BaseEngineCallbacks


class CustomAPIInferenceEngine(InferenceEngine):
    """Custom/self-hosted OpenAI-compatible API inference engine.

    Supports self-hosted OpenAI-compatible API endpoints:
    - Ollama (http://localhost:11434/v1)
    - vLLM server
    - llama.cpp server
    - Other self-hosted inference servers

    Uses CUSTOM_API inference config (generous params for long context).
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        inference_config: InferenceConfig,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 0,
    ):
        self.base_url = base_url
        self.model = model
        self.inference_config = inference_config
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

        # Initialize OpenAI client
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key or "dummy-key-for-local-server",
            timeout=timeout,
            max_retries=max_retries,
        )

    @property
    def engine_type(self) -> InferenceEngineType:
        return InferenceEngineType.CUSTOM_API

    def run_inference(
        self,
        input_path: Path,
        output_path: Path,
        callbacks: BaseEngineCallbacks,
    ) -> Path:
        input_df = pl.read_parquet(input_path)
        conversations = input_df["messages"].to_list()

        responses = []
        for messages in conversations:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.inference_config.temperature or 0.0,
                max_tokens=self.inference_config.max_output_length or 2048,
                top_p=self.inference_config.top_p,
                stop=self.inference_config.stop_tokens,
                # Note: top_k not supported by OpenAI API
            )

            content = response.choices[0].message.content or ""
            responses.append(content)

        output_data = pl.DataFrame({
            "response": responses,
            "chain_of_thought": [""] * len(responses),  # API doesn't separate CoT
        })

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_data.write_parquet(output_path)
        return output_path
