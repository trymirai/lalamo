import polars as pl
from evals.types import EvalPrompt, InferenceOutput, InternalEvalRecord


def build_inference_dataframe(
    prompts: list[EvalPrompt],
    records: list[InternalEvalRecord],
    conversation_column: str = "messages",
) -> pl.DataFrame:
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
