import logging
import os
from pathlib import Path

import polars as pl
import pytest
from tokenizers import Tokenizer

from .common import DEFAULT_JUDGE_MODEL, TASK_PROMPT, judge

from tests.conftest import ConvertModel, RunLalamo

log = logging.getLogger(__name__)

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
TRIVIAL_PROMPT = "What is 2+2?"


@pytest.fixture(params=MODEL_REPOS, ids=lambda repo: repo.split("/")[-1])
def converted_model_path(request: pytest.FixtureRequest, convert_model: ConvertModel) -> Path:
    return convert_model(request.param)


def test_model_coherent_and_stops(
    converted_model_path: Path,
    run_lalamo: RunLalamo,
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
                [{"role": "user", "content": TRIVIAL_PROMPT}],
            ],
        },
    ).write_parquet(dataset_path)

    run_lalamo(
        "generate-replies",
        str(converted_model_path),
        str(dataset_path),
        "--output-path",
        str(output_path),
        "--batch-size",
        "2",
        "--max-output-length",
        str(MAX_TOKENS),
    )

    responses = pl.read_parquet(output_path).get_column("response").to_list()
    assert len(responses) == 2

    coherence_output = responses[0]
    trivial_output = responses[1]

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

    # --- EOS check on trivial prompt ---
    assert trivial_output, "Model produced empty output for trivial prompt"
    assert "4" in trivial_output, f"Expected '4' in the answer, got: {trivial_output!r}"

    tokenizer = Tokenizer.from_file(str(converted_model_path / "tokenizer.json"))
    num_tokens = len(tokenizer.encode(trivial_output).ids)
    assert num_tokens < MAX_TOKENS - 10, (
        f"Model did not stop cleanly — produced {num_tokens} tokens (limit {MAX_TOKENS}). "
        f"Output: {trivial_output[:200]!r}..."
    )
