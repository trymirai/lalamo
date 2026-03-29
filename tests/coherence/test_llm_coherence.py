import logging
import os
import time
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from lalamo.main import app
from lalamo.model_import.model_specs.common import ModelSpec, ModelType
from tests.conftest import ConvertModel, filter_specs, mark_by_size
from tests.model_test_tiers import ModelTier

from .common import DEFAULT_JUDGE_MODEL, TASK_PROMPT, judge

standard_llm_specs = filter_specs(model_type=ModelType.LANGUAGE_MODEL, max_tier=ModelTier.STANDARD)

log = logging.getLogger(__name__)

_runner = CliRunner()

COHERENCE_MAX_TOKENS = 128


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


@pytest.mark.parametrize("spec", mark_by_size(standard_llm_specs), ids=[s.repo for s in standard_llm_specs])
def test_model_coherent_and_stops(
    spec: ModelSpec,
    convert_model: ConvertModel,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    start_time = time.monotonic()
    converted_model_path = convert_model(spec.repo)
    log.info("Model conversion took %.1fs for %s", time.monotonic() - start_time, spec.repo)

    api_key = os.getenv("OPENROUTER_API_KEY")
    assert api_key is not None
    judge_model = os.getenv("COHERENCE_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)

    work_dir = tmp_path_factory.mktemp("coherence")

    start_time = time.monotonic()
    coherence_output = _generate_single(
        converted_model_path,
        work_dir,
        TASK_PROMPT,
        max_tokens=COHERENCE_MAX_TOKENS,
        tag="coherence",
    )
    log.info("Coherence generation took %.1fs for %s", time.monotonic() - start_time, spec.repo)

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
