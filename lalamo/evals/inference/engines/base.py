from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path

import polars as pl
from evals.types import EvalPrompt, InferenceConfig, InferenceEngineType, InferenceOutput, InternalEvalRecord

from lalamo.evals.inference.engines.callbacks import BaseEngineCallbacks


class InferenceEngine(ABC):
    @property
    @abstractmethod
    def engine_type(self) -> InferenceEngineType:
        """Return the engine type for inference config selection."""
        ...

    def _check_unsupported_params(
        self,
        inference_config: InferenceConfig,
        supported_params: set[str],
    ) -> list[str]:
        config_dict = asdict(inference_config)
        return [
            f"{k}={v}"
            for k, v in config_dict.items()
            if k not in supported_params and v is not None
        ]

    def prepare_input(
        self,
        prompts: list[EvalPrompt],
        records: list[InternalEvalRecord],
        output_path: Path,
    ) -> Path:
        if len(prompts) != len(records):
            raise ValueError(
                f"Prompts/records length mismatch: {len(prompts)} prompts != {len(records)} records",
            )

        conversations = []
        ids = []
        questions = []
        answers = []
        metadatas = []

        for prompt, record in zip(prompts, records, strict=True):
            if prompt.id != record.id:
                raise ValueError(
                    f"Order mismatch: prompt.id={prompt.id!r} != record.id={record.id!r}",
                )

            messages = [{"role": msg.role, "content": msg.content} for msg in prompt.messages]

            conversations.append(messages)
            ids.append(prompt.id)
            questions.append(record.question)
            answers.append(record.answer)
            metadatas.append(record.metadata)

        input_data = pl.DataFrame({
            self._get_conversation_column_name(): conversations,
            # following columns are not used for inference but included for output parsing and metadata preservation
            "id": ids,
            "question": questions,
            "answer": answers,
            "metadata": metadatas,
        })

        output_path.parent.mkdir(parents=True, exist_ok=True)
        input_data.write_parquet(output_path)
        return output_path

    def _get_conversation_column_name(self) -> str:
        """Return the column name for storing conversations. """
        return "messages"

    @abstractmethod
    def run_inference(
        self,
        input_path: Path,
        output_path: Path,
        callbacks: BaseEngineCallbacks,
    ) -> Path:
        """Run inference and save outputs."""
        ...

    def parse_output(
        self,
        output_path: Path,
        input_path: Path,
    ) -> list[InferenceOutput]:
        """Parse engine output to standardized format.

        Default implementation assumes output parquet has:
        - response column with model outputs
        - chain_of_thought column (can be empty)

        And input parquet has:
        - id, question, answer, metadata columns
        """
        input_df = pl.read_parquet(input_path)
        output_df = pl.read_parquet(output_path)

        if len(input_df) != len(output_df):
            raise ValueError(
                f"Input/output length mismatch: {len(input_df)} inputs, {len(output_df)} outputs",
            )

        return [
            InferenceOutput(
                id=input_df["id"][i],
                response=output_df["response"][i],
                chain_of_thought=output_df["chain_of_thought"][i],
                question=input_df["question"][i],
                answer=input_df["answer"][i],
                metadata=input_df["metadata"][i],
            )
            for i in range(len(output_df))
        ]
