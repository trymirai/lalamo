from pathlib import Path

import polars as pl
import pytest

from lalamo.data.lalamo_completions import LalamoCompletion
from lalamo.model_import import REPO_TO_MODEL
from tests.conftest import ConvertModel, RunLalamo, strip_ansi_escape

MODELS = ["google/gemma-3-1b-it"]

CAPITAL_PROMPT = "What's the capital of the United Kingdom? No thinking, answer right away."
APPLES_PROMPT = "Are apples fruits? Answer only yes or no, without thinking, answer right away."


def _assert_has_london_and_yes(texts: list[str]) -> None:
    joined = " ".join(texts).lower()
    assert "london" in joined, f"Expected 'london' in {texts!r}"
    assert "yes" in joined, f"Expected 'yes' in {texts!r}"


@pytest.fixture(scope="module", params=MODELS, ids=str)
def converted_model_dir(request: pytest.FixtureRequest, convert_model: ConvertModel) -> Path:
    return convert_model(request.param)


@pytest.fixture
def qa_dataset_path(tmp_path: Path) -> Path:
    dataset_path = tmp_path / "dataset.parquet"
    pl.DataFrame(
        {
            "conversation": [
                [{"role": "user", "content": CAPITAL_PROMPT}],
                [{"role": "user", "content": APPLES_PROMPT}],
            ],
        },
    ).write_parquet(dataset_path)
    return dataset_path


def test_convert(converted_model_dir: Path) -> None:
    assert (converted_model_dir / "model.safetensors").exists() or any(converted_model_dir.glob("model*.safetensors"))
    assert (converted_model_dir / "config.json").exists()
    assert (converted_model_dir / "tokenizer.json").exists()


def test_list_models_plain_and_no_plain(run_lalamo: RunLalamo) -> None:
    plain_output = strip_ansi_escape(run_lalamo("list-models", "--plain"))
    plain_repos = [line.strip() for line in plain_output.splitlines() if line.strip()]
    assert plain_repos
    assert all("/" in repo for repo in plain_repos)
    assert "│" not in plain_output

    fancy_output = strip_ansi_escape(run_lalamo("list-models", "--no-plain"))
    assert "│" in fancy_output
    fancy_repos = [repo for repo in plain_repos if repo in fancy_output]

    expected_repos = sorted(REPO_TO_MODEL)
    assert sorted(plain_repos) == expected_repos
    assert sorted(fancy_repos) == expected_repos


@pytest.mark.parametrize(
    "extra_args",
    [
        pytest.param(["--batch-size", "2"], id="batch-size"),
        pytest.param(["--vram-gb", "6"], id="vram"),
    ],
)
def test_generate_replies(
    converted_model_dir: Path,
    qa_dataset_path: Path,
    tmp_path: Path,
    run_lalamo: RunLalamo,
    extra_args: list[str],
) -> None:
    output_path = tmp_path / "replies.parquet"

    run_lalamo(
        "generate-replies",
        str(converted_model_dir),
        str(qa_dataset_path),
        "--output-path",
        str(output_path),
        *extra_args,
    )

    _assert_has_london_and_yes(pl.read_parquet(output_path).get_column("response").to_list())


def test_chat(
    converted_model_dir: Path,
    run_lalamo: RunLalamo,
) -> None:
    capital_output = run_lalamo("chat", str(converted_model_dir), "--message", CAPITAL_PROMPT)
    assert "london" in capital_output.lower(), f"Expected 'london' in {capital_output!r}"

    apples_output = run_lalamo("chat", str(converted_model_dir), "--message", APPLES_PROMPT)
    assert "yes" in apples_output.lower(), f"Expected 'yes' in {apples_output!r}"


def test_collect_traces_answers(
    converted_model_dir: Path,
    qa_dataset_path: Path,
    tmp_path: Path,
    run_lalamo: RunLalamo,
) -> None:
    trace_path = tmp_path / "traces.bin"

    run_lalamo(
        "speculator",
        "collect-traces",
        str(converted_model_dir),
        str(qa_dataset_path),
        "--output-path",
        str(trace_path),
        "--batch-size",
        "2",
        "--num-tokens-to-generate",
        "32",
        "--max-output-length",
        "64",
    )

    # view-traces detokenizes the completions; collect-traces shuffles so check unordered
    view_output = strip_ansi_escape(
        run_lalamo(
            "speculator",
            "view-traces",
            str(trace_path),
            str(converted_model_dir),
        ),
    )
    _assert_has_london_and_yes([view_output])
