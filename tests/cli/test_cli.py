from pathlib import Path

import polars as pl
import pytest

from lalamo.model_registry import ModelRegistry
from tests.conftest import ConvertModel, RunLalamo, strip_ansi_escape

MODELS = ["google/gemma-3-1b-it", "mlx-community/LFM2-350M-8bit"]

CAPITAL_PROMPT = "What's the capital of the United Kingdom? No thinking, answer right away."
APPLES_PROMPT = "Are apples fruits? Answer only yes or no, without thinking, answer right away."


def _assert_has_london_and_yes(texts: list[str]) -> None:
    joined = " ".join(texts).lower()
    assert "london" in joined, f"Expected 'london' in {texts!r}"
    assert "yes" in joined, f"Expected 'yes' in {texts!r}"


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


@pytest.mark.fast
@pytest.mark.parametrize("model_repo", MODELS)
def test_convert(convert_model: ConvertModel, model_repo: str) -> None:
    converted_model_dir = convert_model(model_repo, cached=True)
    assert (converted_model_dir / "model.safetensors").exists() or any(converted_model_dir.glob("model*.safetensors"))
    assert (converted_model_dir / "config.json").exists()
    assert (converted_model_dir / "tokenizer.json").exists()


def test_list_models_plain_and_no_plain(run_lalamo: RunLalamo, model_registry: ModelRegistry) -> None:
    plain_output = strip_ansi_escape(run_lalamo("list-models", "--plain"))
    plain_repos = [line.strip() for line in plain_output.splitlines() if line.strip()]
    assert plain_repos
    assert all("/" in repo for repo in plain_repos)
    assert "│" not in plain_output

    fancy_output = strip_ansi_escape(run_lalamo("list-models", "--no-plain"))
    assert "│" in fancy_output
    fancy_repos = [repo for repo in plain_repos if repo in fancy_output]

    local_repos = set(model_registry.repo_to_model)
    assert local_repos.issubset(set(plain_repos))
    assert local_repos.issubset(set(fancy_repos))


@pytest.mark.fast
@pytest.mark.parametrize("model_repo", MODELS)
def test_generate_replies(
    convert_model: ConvertModel,
    model_repo: str,
    qa_dataset_path: Path,
    tmp_path: Path,
    run_lalamo: RunLalamo,
) -> None:
    converted_model_dir = convert_model(model_repo, cached=True)
    output_path = tmp_path / "replies.parquet"

    run_lalamo(
        "generate-replies",
        str(converted_model_dir),
        str(qa_dataset_path),
        "--output-path",
        str(output_path),
        "--batch-size",
        "2",
        "--max-output-length",
        "64",
    )

    _assert_has_london_and_yes(pl.read_parquet(output_path).get_column("response").to_list())


@pytest.mark.parametrize("model_repo", MODELS)
def test_generate_replies_with_vram(
    convert_model: ConvertModel,
    model_repo: str,
    qa_dataset_path: Path,
    tmp_path: Path,
    run_lalamo: RunLalamo,
) -> None:
    converted_model_dir = convert_model(model_repo, cached=True)
    output_path = tmp_path / "replies.parquet"

    run_lalamo(
        "generate-replies",
        str(converted_model_dir),
        str(qa_dataset_path),
        "--output-path",
        str(output_path),
        "--vram-gb",
        "6",
        "--max-output-length",
        "64",
    )

    _assert_has_london_and_yes(pl.read_parquet(output_path).get_column("response").to_list())


@pytest.mark.parametrize("model_repo", MODELS)
def test_chat(
    convert_model: ConvertModel,
    model_repo: str,
    run_lalamo: RunLalamo,
) -> None:
    converted_model_dir = convert_model(model_repo, cached=True)
    capital_output = run_lalamo("chat", str(converted_model_dir), "--message", CAPITAL_PROMPT)
    assert "london" in capital_output.lower(), f"Expected 'london' in {capital_output!r}"

    apples_output = run_lalamo("chat", str(converted_model_dir), "--message", APPLES_PROMPT)
    assert "yes" in apples_output.lower(), f"Expected 'yes' in {apples_output!r}"


@pytest.mark.parametrize("model_repo", MODELS)
def test_collect_traces_answers(
    convert_model: ConvertModel,
    model_repo: str,
    qa_dataset_path: Path,
    tmp_path: Path,
    run_lalamo: RunLalamo,
) -> None:
    converted_model_dir = convert_model(model_repo, cached=True)
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
