import logging
import os
import re
from pathlib import Path

import polars as pl
import pytest
from jax.errors import JaxRuntimeError
from tokenizers import Tokenizer
from typer.testing import CliRunner

from lalamo.main import app
from lalamo.model_import.model_specs.common import ModelSpec
from tests.conftest import ConvertModel

from .common import DEFAULT_JUDGE_MODEL, TASK_PROMPT, judge

log = logging.getLogger(__name__)

_runner = CliRunner()

MAX_TOKENS = 1024

# Models that cannot answer factual questions correctly (e.g. function-calling models).
# For these, only EOS behavior is verified; QA adequacy assertions are skipped.
BAD_MODELS: list[str] = ["google/functiongemma-270m-it", "amd/PARD-Qwen3-0.6B"]

# Models with extended thinking/reasoning that exhaust the default token budget.
# Token limit is multiplied by 4 to give them room to finish reasoning before answering.
THINKING_MODELS: list[str] = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "trymirai/DeepSeek-R1-Distill-Qwen-1.5B-AWQ",
    "HuggingFaceTB/SmolLM3-3B",
    "mlx-community/SmolLM3-3B-8bit",
    "mlx-community/SmolLM3-3B-4bit",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-8B-AWQ",
    "Qwen/Qwen3-8B-MLX-4bit",
    "Qwen/Qwen3-0.6B-MLX-4bit",
    "Qwen/Qwen3-0.6B-MLX-8bit",
    "Qwen/Qwen3-0.6B",
    "RekaAI/reka-flash-3.1",
    "Nanbeige/Nanbeige4.1-3B",
]

SIMPLE_QA: list[tuple[str, re.Pattern[str]]] = [
    ("What is 2+2?", re.compile(r"\b4\b")),
    ("What is the capital of France?", re.compile(r"\bparis\b", re.IGNORECASE)),
    ("What is H2O?", re.compile(r"\bwater\b", re.IGNORECASE)),
]


@pytest.fixture
def converted_model_path(llm_spec: ModelSpec, convert_model: ConvertModel) -> Path:
    return convert_model(llm_spec.repo)


def _generate_replies(
    converted_model_path: Path,
    dataset_path: Path,
    output_path: Path,
    *,
    batch_size: int,
    max_tokens: int = MAX_TOKENS,
) -> None:
    result = _runner.invoke(
        app,
        [
            "generate-replies",
            str(converted_model_path),
            str(dataset_path),
            "--output-path",
            str(output_path),
            "--batch-size",
            str(batch_size),
            "--max-output-length",
            str(max_tokens),
        ],
        terminal_width=240,
    )
    assert result.exit_code == 0, (
        f"generate-replies failed (exit {result.exit_code}).\n"
        f"--- output ---\n{result.output}\n"
        f"--- exception ---\n{result.exception!r}"
    )


def test_model_coherent_and_stops(
    llm_spec: ModelSpec,
    converted_model_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    assert api_key is not None

    judge_model = os.getenv("COHERENCE_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)

    is_bad = llm_spec.repo in BAD_MODELS
    max_tokens = MAX_TOKENS * 4 if llm_spec.repo in THINKING_MODELS else MAX_TOKENS

    work_dir = tmp_path_factory.mktemp("coherence")
    dataset_path = work_dir / "dataset.parquet"
    output_path = work_dir / "replies.parquet"

    n_repeats = 2
    pl.DataFrame(
        {
            "conversation": [
                [{"role": "user", "content": TASK_PROMPT}],
                *[[{"role": "user", "content": q}] for q, _ in SIMPLE_QA for _ in range(n_repeats)],
            ],
        },
    ).write_parquet(dataset_path)

    batch_size = 1 + len(SIMPLE_QA) * n_repeats
    _generate_replies(converted_model_path, dataset_path, output_path, batch_size=batch_size, max_tokens=max_tokens)

    responses = pl.read_parquet(output_path).get_column("response").to_list()
    assert len(responses) == batch_size

    coherence_output = responses[0]
    simple_outputs = responses[1:]

    tokenizer = Tokenizer.from_file(str(converted_model_path / "tokenizer.json"))

    # --- coherence check on hash table prompt ---
    assert coherence_output, "Model produced empty output for coherence prompt"
    log.info("Coherence output:\n%s", coherence_output)

    verdict = judge(
        api_key=api_key,
        model=judge_model,
        candidate_output=coherence_output,
        timeout=60,
    )

    log.info(
        "Judge verdict: coherent=%s, score=%.2f, issues=%s, summary=%s",
        verdict.coherent,
        verdict.score,
        verdict.issues,
        verdict.summary,
    )

    issues = ", ".join(verdict.issues) or "none"
    assert verdict.coherent, (
        f"The model output ({coherence_output}) was found to be incoherent "
        f"(score={verdict.score:.2f}, issues={issues}, summary={verdict.summary!r})"
    )

    # --- checks on simple factual QA prompts ---
    # Each question is repeated n_repeats times to get diverse completions.
    # Allow at most 2 failures across all QA items to tolerate occasional
    # flakiness (e.g. thinking models spending their whole token budget on
    # reasoning before outputting an answer).
    qa_failures: list[str] = []
    qa_items = [(q, p) for q, p in SIMPLE_QA for _ in range(n_repeats)]
    for (question, pattern), response in zip(qa_items, simple_outputs, strict=True):
        assert response, f"Model produced empty output for: {question!r}"

        num_tokens = len(tokenizer.encode(response).ids)
        if num_tokens >= max_tokens - 10:
            qa_failures.append(
                f"Model did not stop cleanly for {question!r} — produced {num_tokens} tokens (limit {max_tokens}). "
                f"Output: {response[:200]!r}...",
            )
            continue

        if is_bad:
            continue

        if pattern.search(response) is None:
            qa_failures.append(
                f"Expected pattern {pattern.pattern!r} not found in response to {question!r}: {response!r}",
            )
            continue

        qa_verdict = judge(
            api_key=api_key,
            model=judge_model,
            candidate_output=response,
            task_prompt=question,
            timeout=60,
        )
        log.info(
            "QA judge for %r: coherent=%s, score=%.2f, issues=%s",
            question,
            qa_verdict.coherent,
            qa_verdict.score,
            qa_verdict.issues,
        )

    if not is_bad:
        total_qa = len(SIMPLE_QA) * n_repeats
        assert len(qa_failures) <= 2, (
            f"{len(qa_failures)}/{total_qa} QA checks failed (at most 2 allowed):\n"
            + "\n".join(f"  - {f}" for f in qa_failures)
        )
