import logging
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lalamo.main import app

from .common import DEFAULT_JUDGE_MODEL, TASK_PROMPT, extract_output, judge

log = logging.getLogger(__name__)

# Representative language models < 10B: smallest per family + one MLX 4bit each.
MODEL_REPOS = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "EssentialAI/rnj-1-instruct",
    "google/gemma-2-2b-it",
    "google/gemma-3-1b-it",
    # "google/functiongemma-270m-it",  # output is weird by default, can't verify coherence
    "mlx-community/gemma-3-1b-it-4bit",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "LiquidAI/LFM2-350M",
    "LiquidAI/LFM2.5-1.2B-Instruct",
    "mlx-community/LFM2-350M-4bit",
    "meta-llama/Llama-3.2-1B-Instruct",
    "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "cartesia-ai/Llamba-1B",
    "cartesia-ai/Llamba-1B-4bit-mlx",
    "POLARIS-Project/Polaris-4B-Preview",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-0.6B-MLX-4bit",
]


@pytest.fixture(params=MODEL_REPOS, ids=lambda repo: repo.split("/")[-1])
def converted_model_path(request: pytest.FixtureRequest, tmp_path: Path) -> Path:
    repo = request.param
    output_dir = tmp_path / "model"

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["convert", repo, "--output-dir", str(output_dir), "--overwrite"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, f"lalamo convert {repo} failed (exit {result.exit_code}): {result.output.strip()!r}"
    return output_dir


def test_model_coherent(converted_model_path: Path) -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("Set OPENROUTER_API_KEY to run coherence tests.")

    judge_model = os.getenv("COHERENCE_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["chat", str(converted_model_path), "--message", TASK_PROMPT, "--max-tokens", "256"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, f"lalamo chat failed (exit {result.exit_code}): {result.output.strip()!r}"

    output = extract_output(result.stdout)
    assert output, "Model produced empty output"

    log.info("Model output:\n%s", output)

    verdict = judge(
        api_key=api_key,
        model=judge_model,
        candidate_output=output,
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
        f"The model output ({output}) was found to be incoherent "
        f"(score={verdict.score:.2f}, issues={issues}, summary={verdict.summary!r})"
    )
