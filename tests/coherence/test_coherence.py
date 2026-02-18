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
from lalamo.model_import.model_specs.common import ModelType
from lalamo.model_registry import ModelRegistry
from tests.conftest import ConvertModel

from .common import DEFAULT_JUDGE_MODEL, TASK_PROMPT, judge

log = logging.getLogger(__name__)

_runner = CliRunner()

MODEL_REPOS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen3-0.6B",
    "mlx-community/gemma-3-1b-it-8bit",
    "google/gemma-2-2b-it",
    # "google/functiongemma-270m-it",  # output is weird by default, can't verify coherence
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "LiquidAI/LFM2.5-1.2B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "cartesia-ai/Llamba-1B-4bit-mlx",
]

MAX_TOKENS = 256

SIMPLE_QA: list[tuple[str, re.Pattern[str]]] = [
    ("What is 2+2?", re.compile(r"\b4\b")),
    ("What is the capital of France?", re.compile(r"\bparis\b", re.IGNORECASE)),
    ("What is H2O?", re.compile(r"\bwater\b", re.IGNORECASE)),
]


def _coherence_model_repos() -> list[str]:
    # When LALAMO_COHERENCE_FULL_COVERAGE=1, uses all registry LMs; GPU OOM → skip at runtime.
    if not os.getenv("LALAMO_COHERENCE_FULL_COVERAGE"):
        return MODEL_REPOS
    registry = ModelRegistry.build()
    return [spec.repo for spec in registry.models if spec.model_type == ModelType.LANGUAGE_MODEL]


@pytest.fixture(params=_coherence_model_repos(), ids=lambda repo: repo.split("/")[-1])
def converted_model_path(request: pytest.FixtureRequest, convert_model: ConvertModel) -> Path:
    try:
        return convert_model(request.param)
    except JaxRuntimeError as e:
        pytest.skip(f"Model too large to fit in GPU memory during conversion: {e}")


def _generate_replies(
    converted_model_path: Path,
    dataset_path: Path,
    output_path: Path,
    *,
    batch_size: int,
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
            str(MAX_TOKENS),
        ],
        terminal_width=240,
    )
    if isinstance(result.exception, JaxRuntimeError):
        pytest.skip(f"Model too large to fit in GPU memory during generation: {result.exception}")
    assert result.exit_code == 0, (
        f"generate-replies failed (exit {result.exit_code}).\n"
        f"--- output ---\n{result.output}\n"
        f"--- exception ---\n{result.exception!r}"
    )


def test_model_coherent_and_stops(
    converted_model_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("Set OPENROUTER_API_KEY to run coherence tests.")

    judge_model = os.getenv("COHERENCE_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)

    work_dir = tmp_path_factory.mktemp("coherence")
    dataset_path = work_dir / "dataset.parquet"
    output_path = work_dir / "replies.parquet"

    pl.DataFrame(
        {
            "conversation": [
                [{"role": "user", "content": TASK_PROMPT}],
                *[[{"role": "user", "content": q}] for q, _ in SIMPLE_QA],
            ],
        },
    ).write_parquet(dataset_path)

    batch_size = 1 + len(SIMPLE_QA)
    _generate_replies(converted_model_path, dataset_path, output_path, batch_size=batch_size)

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
    # Allow at most one failure across all QA items to tolerate occasional
    # flakiness (e.g. thinking models spending their whole token budget on
    # reasoning before outputting an answer).
    qa_failures: list[str] = []
    for (question, pattern), response in zip(SIMPLE_QA, simple_outputs, strict=True):
        assert response, f"Model produced empty output for: {question!r}"

        if pattern.search(response) is None:
            qa_failures.append(
                f"Expected pattern {pattern.pattern!r} not found in response to {question!r}: {response!r}"
            )
            continue

        num_tokens = len(tokenizer.encode(response).ids)
        if num_tokens >= MAX_TOKENS - 10:
            qa_failures.append(
                f"Model did not stop cleanly for {question!r} — produced {num_tokens} tokens (limit {MAX_TOKENS}). "
                f"Output: {response[:200]!r}..."
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

    assert len(qa_failures) <= 1, (
        f"{len(qa_failures)}/{len(SIMPLE_QA)} QA checks failed (majority required):\n"
        + "\n".join(f"  - {f}" for f in qa_failures)
    )
