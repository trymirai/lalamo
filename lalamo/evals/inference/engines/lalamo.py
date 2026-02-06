from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
from evals.types import EvalPrompt, InferenceOutput, InternalEvalRecord

from lalamo.commands import generate_replies
from lalamo.evals.inference.engines.base import InferenceEngine


@dataclass(frozen=True)
class LalamoInferenceEngine(InferenceEngine):
    model_path: Path
    max_vram: int | None = None
    batch_size: int | None = None
    max_output_length: int = 8192

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
            # verify ID alignment
            if prompt.id != record.id:
                raise ValueError(
                    f"Order mismatch: prompt.id={prompt.id!r} != record.id={record.id!r}. "
                    "format_prompts must maintain dataset order.",
                )

            conversation = [
                {"role": msg.role, "content": msg.content}
                for msg in prompt.messages
            ]

            conversations.append(conversation)
            ids.append(prompt.id)
            questions.append(record.question)
            answers.append(record.answer)
            metadatas.append(record.metadata)

        input_data = pl.DataFrame({
            "conversation": conversations,
            # following columns are not used for inference but included for output parsing and metadata preservation
            "id": ids,
            "question": questions,
            "answer": answers,
            "metadata": metadatas,
        })

        output_path.parent.mkdir(parents=True, exist_ok=True)
        input_data.write_parquet(output_path)
        return output_path

    def run_inference(
        self,
        input_path: Path,
        output_path: Path,
        **engine_params: Any,  # noqa: ANN401
    ) -> Path:
        from lalamo.main import CliGenerateRepliesCallbacks

        max_vram = engine_params.get("max_vram", self.max_vram)
        batch_size = engine_params.get("batch_size", self.batch_size)
        max_output_length = engine_params.get("max_output_length", self.max_output_length)

        generate_replies(
            model_path=self.model_path,
            dataset_path=input_path,
            output_path=output_path,
            max_vram=max_vram,
            max_output_length=max_output_length,
            batch_size=batch_size,
            callbacks_type=CliGenerateRepliesCallbacks,
        )

        return output_path

    def parse_output(
        self,
        output_path: Path,
        input_path: Path,
    ) -> list[InferenceOutput]:
        input_df = pl.read_parquet(input_path)
        output_df = pl.read_parquet(output_path)

        if len(input_df) != len(output_df):
            raise ValueError(
                f"Input/output length mismatch: {len(input_df)} inputs, {len(output_df)} outputs",
            )

        # match by position - we assume input and output are in the same order
        outputs = [
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
        return outputs
