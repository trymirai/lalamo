from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
from evals.types import EvalPrompt, InferenceOutput

from lalamo.commands import generate_replies
from lalamo.evals.inference.callbacks import BaseRunInferenceCallbacks
from lalamo.evals.inference.engines.base import InferenceEngine
from lalamo.main import CliGenerateRepliesCallbacks


@dataclass(frozen=True)
class LalamoInferenceEngine(InferenceEngine):
    model_path: Path
    max_vram: int | None = None
    batch_size: int | None = None
    max_output_length: int = 8192

    def prepare_input(
        self,
        prompts: list[EvalPrompt],
        output_path: Path,
    ) -> Path:
        conversations = []
        ids = []
        for prompt in prompts:
            # Convert PromptMessage to HF format
            conversation = [
                {"role": msg.role, "content": msg.content}
                for msg in prompt.messages
            ]
            conversations.append(conversation)
            ids.append(prompt.id)

        # Note: id is stored for later matching by order
        # TODO(mullakhmetov): confirm if we need id it
        input_data = pl.DataFrame({
            "id": ids,
            "conversation": conversations,
        })

        output_path.parent.mkdir(parents=True, exist_ok=True)
        input_data.write_parquet(output_path)
        return output_path

    def run_inference(
        self,
        input_path: Path,
        output_path: Path,
        callbacks: BaseRunInferenceCallbacks,
        **engine_params: Any,  # noqa: ANN401
    ) -> Path:
        max_vram = engine_params.get("max_vram", self.max_vram)
        batch_size = engine_params.get("batch_size", self.batch_size)
        max_output_length = engine_params.get("max_output_length", self.max_output_length)

        # Count rows for callback initialization
        total_rows = pl.scan_parquet(input_path).select(pl.len()).collect().item()

        # Use CliGenerateRepliesCallbacks for the batch generation
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
        ids = input_df["id"].to_list()

        output_df = pl.read_parquet(output_path)

        if len(ids) != len(output_df):
            raise ValueError(
                f"Input/output length mismatch: {len(ids)} inputs, {len(output_df)} outputs",
            )

        outputs = [
            InferenceOutput(
                id=ids[i],
                response=output_df["response"][i],
                chain_of_thought=output_df["chain_of_thought"][i],
            )
            for i in range(len(output_df))
        ]
        return outputs
