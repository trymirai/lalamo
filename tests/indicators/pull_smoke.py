import re
from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest
from tokenizers import Tokenizer
from typer.testing import CliRunner

from lalamo.main import app

RunLalamo = Callable[..., str]

PULL_MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct"
MATH_PROMPT = "What is 2 + 2? Reply with a single number, nothing else."
YES_NO_PROMPT = "Are apples fruits? Answer with one word: yes or no."
MAX_RESPONSE_TOKENS = 30


def _has_model_weights(model_dir: Path) -> bool:
    return (model_dir / "model.safetensors").exists() or any(model_dir.glob("model*.safetensors"))


@pytest.fixture(scope="module")
def run_lalamo() -> RunLalamo:
    runner = CliRunner()

    def _run(*args: str) -> str:
        result = runner.invoke(app, list(args), terminal_width=240)
        assert result.exit_code == 0, (
            f"lalamo {' '.join(args)} failed (exit {result.exit_code}).\n"
            f"--- output ---\n{result.output}\n"
            f"--- exception ---\n{result.exception!r}"
        )
        return result.output

    return _run


@pytest.fixture(scope="module")
def pulled_model_dir(
    tmp_path_factory: pytest.TempPathFactory,
    run_lalamo: RunLalamo,
) -> Path:
    output_dir = tmp_path_factory.mktemp("pulled_models") / PULL_MODEL_REPO.replace("/", "__")
    run_lalamo("pull", PULL_MODEL_REPO, "--output-dir", str(output_dir))

    assert (output_dir / "config.json").exists(), f"Missing config.json in {output_dir}"
    assert (output_dir / "tokenizer.json").exists(), f"Missing tokenizer.json in {output_dir}"
    assert _has_model_weights(output_dir), f"Missing model weights in {output_dir}"
    return output_dir


def test_pulled_model_generates_adequate_output(
    pulled_model_dir: Path,
    tmp_path_factory: pytest.TempPathFactory,
    run_lalamo: RunLalamo,
) -> None:
    work_dir = tmp_path_factory.mktemp("pull_smoke")
    dataset_path = work_dir / "dataset.parquet"
    output_path = work_dir / "responses.parquet"

    pl.DataFrame(
        {
            "conversation": [
                [{"role": "user", "content": MATH_PROMPT}],
                [{"role": "user", "content": YES_NO_PROMPT}],
            ],
        },
    ).write_parquet(dataset_path)

    run_lalamo(
        "generate-replies",
        str(pulled_model_dir),
        str(dataset_path),
        "--output-path",
        str(output_path),
        "--batch-size",
        "2",
        "--max-output-length",
        "64",
    )

    responses = [str(response) for response in pl.read_parquet(output_path).get_column("response").to_list()]
    assert len(responses) == 2

    tokenizer = Tokenizer.from_file(str(pulled_model_dir / "tokenizer.json"))
    token_counts = [len(tokenizer.encode(response).ids) for response in responses]

    math_response = responses[0].lower()
    yes_no_response = responses[1].lower()

    assert re.search(r"\b4\b", math_response), f"Expected a '4' answer, got: {responses[0]!r}"
    assert re.search(r"\byes\b", yes_no_response), f"Expected a 'yes' answer, got: {responses[1]!r}"
    assert token_counts[0] < MAX_RESPONSE_TOKENS, (
        f"Math response is too long ({token_counts[0]} tokens): {responses[0]!r}"
    )
    assert token_counts[1] < MAX_RESPONSE_TOKENS, (
        f"Yes/no response is too long ({token_counts[1]} tokens): {responses[1]!r}"
    )
