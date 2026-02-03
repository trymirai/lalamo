from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from lalamo.evals.inference.engines.base import InferenceEngine
from evals.types import EvalPrompt, InferenceOutput


@dataclass(frozen=True)
class LalamoInferenceEngine(InferenceEngine):
    """Inference engine using lalamo's generate_replies()."""

    model_path: Path
    max_vram: int | None = None
    batch_size: int | None = None
    max_output_length: int = 8192

    def prepare_input(
        self,
        prompts: list[EvalPrompt],
        output_path: Path,
    ) -> Path:
        """Convert EvalPrompt list to HF conversation format parquet.

        lalamo's generate_replies() expects a parquet file with:
        - 'id' column: string IDs for tracking
        - 'conversation' column: list of dicts with 'role' and 'content'
        """

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

        # Create DataFrame with id + conversation
        # Note: id is stored for later matching by order
        df = pl.DataFrame({
            "id": ids,
            "conversation": conversations,
        })

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(output_path)
        return output_path

    def run_inference(
        self,
        input_path: Path,
        output_path: Path,
        **engine_params,
    ) -> Path:
        """Run generate_replies() on the input with progress reporting."""

        max_vram = engine_params.get("max_vram", self.max_vram)
        batch_size = engine_params.get("batch_size", self.batch_size)
        max_output_length = engine_params.get("max_output_length", self.max_output_length)

        from lalamo.evals.inference.runner import generate_replies, GenerateRepliesCallbacks

        # Custom callbacks for progress reporting
        console = Console()

        @dataclass
        class InferenceProgressCallbacks(GenerateRepliesCallbacks):
            last_reported: int = 0

            def loading_model(self) -> None:
                console.print("      Loading model...")

            def finished_loading_model(self) -> None:
                console.print("      ✓ Model loaded")

            def loading_dataset(self) -> None:
                pass  # Already loaded

            def batch_sizes_estimated(self) -> None:
                console.print("      Estimating optimal batch sizes...")

            def batch_sizes_computed(self, event) -> None:
                console.print("      ✓ Batch sizes computed")
                console.print("      Warming up model...")

            def generation_progress(self, rows_processed: int) -> None:
                # rows_processed is 0-indexed from enumerate, so actual count is +1
                actual_processed = rows_processed + 1

                # Report every 10% or on first/last
                report_threshold = max(1, self.total_rows // 10)
                if actual_processed == 1 or actual_processed == self.total_rows or actual_processed - self.last_reported >= report_threshold:
                    percentage = (actual_processed / self.total_rows * 100) if self.total_rows > 0 else 0
                    console.print(f"      Generating replies... {actual_processed}/{self.total_rows} ({percentage:.0f}%)")
                    self.last_reported = actual_processed

            def finished_generation(self) -> None:
                console.print("      ✓ Generation complete")

        generate_replies(
            model_path=self.model_path,
            dataset_path=input_path,
            output_path=output_path,
            max_vram=max_vram,
            max_output_length=max_output_length,
            batch_size=batch_size,
            callbacks_type=InferenceProgressCallbacks,
        )

        return output_path

    def parse_output(
        self,
        output_path: Path,
        input_path: Path,
    ) -> list[InferenceOutput]:
        """Parse generate_replies output parquet.

        NOTE: Assumes output order matches input order since generate_replies()
        doesn't preserve the 'id' column. This is guaranteed by the current
        implementation which processes rows sequentially and sorts by index.
        """

        # Read input to get IDs
        input_df = pl.read_parquet(input_path)
        ids = input_df["id"].to_list()

        # Read output
        output_df = pl.read_parquet(output_path)

        if len(ids) != len(output_df):
            raise ValueError(
                f"Input/output length mismatch: {len(ids)} inputs, {len(output_df)} outputs"
            )

        outputs = []
        for i in range(len(output_df)):
            outputs.append(
                InferenceOutput(
                    id=ids[i],
                    response=output_df["response"][i],
                    chain_of_thought=output_df["chain_of_thought"][i],
                )
            )

        return outputs
