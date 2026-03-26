import logging
import os
import re
import time
from pathlib import Path

import polars as pl
import pytest
from tokenizers import Tokenizer
from typer.testing import CliRunner

from lalamo.main import app
from lalamo.model_import.model_specs.common import ModelSpec
from tests.conftest import ConvertModel

from .common import DEFAULT_JUDGE_MODEL, TASK_PROMPT, judge

log = logging.getLogger(__name__)

_runner = CliRunner()

COHERENCE_MAX_TOKENS = 128
QA_MAX_TOKENS = 128

QA_QUESTION = "What color is beetroot?"
QA_PATTERN = re.compile(r"\b(purple|red|dark\s*red|deep\s*red|maroon|crimson|magenta)\b", re.IGNORECASE)


def _generate_single(
    converted_model_path: Path,
    work_dir: Path,
    prompt: str,
    *,
    max_tokens: int,
    tag: str,
) -> str:
    dataset_path = work_dir / f"{tag}_dataset.parquet"
    output_path = work_dir / f"{tag}_replies.parquet"
    pl.DataFrame({"conversation": [[{"role": "user", "content": prompt}]]}).write_parquet(dataset_path)
    result = _runner.invoke(
        app,
        [
            "generate-replies",
            str(converted_model_path),
            str(dataset_path),
            "--output-path",
            str(output_path),
            "--batch-size",
            "1",
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
    return pl.read_parquet(output_path).get_column("response").to_list()[0]


def test_model_coherent_and_stops(
    standard_llm_spec: ModelSpec,
    convert_model: ConvertModel,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    t0 = time.monotonic()
    converted_model_path = convert_model(standard_llm_spec.repo)
    log.info("Model conversion took %.1fs for %s", time.monotonic() - t0, standard_llm_spec.repo)

    api_key = os.getenv("OPENROUTER_API_KEY")
    assert api_key is not None
    judge_model = os.getenv("COHERENCE_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)

    work_dir = tmp_path_factory.mktemp("coherence")

    t0 = time.monotonic()
    coherence_output = _generate_single(
        converted_model_path,
        work_dir,
        TASK_PROMPT,
        max_tokens=COHERENCE_MAX_TOKENS,
        tag="coherence",
    )
    log.info("Coherence generation took %.1fs for %s", time.monotonic() - t0, standard_llm_spec.repo)

    assert coherence_output, "Model produced empty output for coherence prompt"
    log.info("Coherence output:\n%s", coherence_output)

    verdict = judge(api_key=api_key, model=judge_model, candidate_output=coherence_output, timeout=60)
    log.info(
        "Judge verdict: coherent=%s, score=%.2f, issues=%s, summary=%s",
        verdict.coherent,
        verdict.score,
        verdict.issues,
        verdict.summary,
    )
    assert verdict.coherent, (
        f"Output incoherent (score={verdict.score:.2f}, "
        f"issues={', '.join(verdict.issues) or 'none'}, summary={verdict.summary!r}):\n{coherence_output}"
    )

    tokenizer = Tokenizer.from_file(str(converted_model_path / "tokenizer.json"))

    for attempt in range(2):
        t0 = time.monotonic()
        qa_response = _generate_single(
            converted_model_path,
            work_dir,
            QA_QUESTION,
            max_tokens=QA_MAX_TOKENS,
            tag=f"qa_{attempt}",
        )
        log.info(
            "QA generation (attempt %d) took %.1fs for %s", attempt, time.monotonic() - t0, standard_llm_spec.repo
        )

        num_tokens = len(tokenizer.encode(qa_response).ids)
        stopped = num_tokens < QA_MAX_TOKENS - 10
        log.info("QA response (attempt %d, %d tokens, stopped=%s):\n%s", attempt, num_tokens, stopped, qa_response)

        if stopped:
            break
    else:
        pytest.fail(
            f"Model did not stop within {QA_MAX_TOKENS} tokens after 2 attempts for {QA_QUESTION!r}. "
            f"Last output ({num_tokens} tokens): {qa_response[:300]!r}"
        )

    assert QA_PATTERN.search(qa_response), (
        f"Expected answer about beetroot color not found in response to {QA_QUESTION!r}: {qa_response!r}"
    )

    qa_verdict = judge(
        api_key=api_key,
        model=judge_model,
        candidate_output=qa_response,
        task_prompt=QA_QUESTION,
        timeout=60,
    )
    log.info(
        "QA judge: coherent=%s, score=%.2f, issues=%s",
        qa_verdict.coherent,
        qa_verdict.score,
        qa_verdict.issues,
    )
    assert qa_verdict.coherent, (
        f"QA response incoherent (score={qa_verdict.score:.2f}, "
        f"issues={', '.join(qa_verdict.issues) or 'none'}):\n{qa_response}"
    )
