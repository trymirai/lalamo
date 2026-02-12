import json
import warnings

import cattrs
import polars as pl
from evals.types import EvalPrompt, InferenceOutput

# Suppress warning for map_elements - necessary for dynamic JSON structures
warnings.filterwarnings("ignore", category=pl.exceptions.PolarsInefficientMapWarning)


def build_inference_dataframe(
    prompts: list[EvalPrompt],
    conversation_column: str = "messages",
) -> pl.DataFrame:
    conversations = []
    ids = []
    questions = []
    answers = []
    metadatas = []

    for prompt in prompts:
        messages = [{"role": msg.role, "content": msg.content} for msg in prompt.messages]

        conversations.append(messages)
        ids.append(prompt.id)
        questions.append(prompt.question)
        answers.append(prompt.answer)
        # metadata is a nested object, so we serialize it as JSON string for storage in DataFrame
        metadatas.append(json.dumps(prompt.metadata or {}))

    return pl.DataFrame({
        conversation_column: conversations,
        # following columns are not used for inference but included for output parsing and metadata preservation
        "id": ids,
        "question": questions,
        "answer": answers,
        "metadata": metadatas,
    })


def parse_inference_outputs(
    output_df: pl.DataFrame,
    input_df: pl.DataFrame,
) -> list[InferenceOutput]:
    if len(input_df) != len(output_df):
        raise ValueError(
            f"Input/output length mismatch: {len(input_df)} inputs, {len(output_df)} outputs",
        )

    # Use map_elements for truly dynamic JSON metadata
    # Cannot use str.json_decode() because:
    # 1. lm_eval_doc fields vary by benchmark (MMLU-Pro vs IFEval have different schemas)
    # 2. Polars doesn't support pl.Object inside Struct types in json_decode
    input_df = input_df.with_columns(
        pl.col("metadata").map_elements(json.loads, return_dtype=pl.Object),
    )

    combined_df = pl.concat(
        [
            input_df.select(["id", "question", "answer", "metadata"]),
            output_df.select(["response", "chain_of_thought"]),
        ],
        how="horizontal",
    )

    return cattrs.structure(combined_df.to_dicts(), list[InferenceOutput])
