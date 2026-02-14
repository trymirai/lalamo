import re
from pathlib import Path

import polars as pl
import pytest

from lalamo.data.lalamo_completions import LalamoCompletion
from lalamo.model_import import REPO_TO_MODEL
from tests.conftest import ConvertModel, RunLalamo

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "cartesia-ai/Llamba-1B-4bit-mlx",
    "mlx-community/LFM2-350M-4bit",
]

CAPITAL_PROMPT = "What's the capital of the United Kingdom? No thinking, answer right away."
APPLES_PROMPT = "Are apples fruits? Answer only yes or no, without thinking, answer right away."


def _assert_has_london_and_yes(texts: list[str]) -> None:
    joined = " ".join(texts).lower()
    assert "london" in joined, f"Expected 'london' in {texts!r}"
    assert "yes" in joined, f"Expected 'yes' in {texts!r}"


ANSI_ESCAPE_REGEX = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi_escape(s: str) -> str:
    return ANSI_ESCAPE_REGEX.sub("", s)


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


def test_collect_traces_max_output_length_does_not_change_logits(
    converted_model_dir: Path,
    tmp_path: Path,
    run_lalamo: RunLalamo,
) -> None:
    dataset_path = tmp_path / "dataset.parquet"
    pl.DataFrame(
        {
            "conversation": [
                [{"role": "user", "content": "Implement a B-tree data structure in Rust."}],
            ],
        },
    ).write_parquet(dataset_path)

    short_trace_path = tmp_path / "short.bin"
    run_lalamo(
        "speculator",
        "collect-traces",
        str(converted_model_dir),
        str(dataset_path),
        "--output-path",
        str(short_trace_path),
        "--batch-size",
        "1",
        "--num-logits-per-token",
        "8",
        "--num-tokens-to-generate",
        "32",
        "--max-output-length",
        "32",
    )

    long_trace_path = tmp_path / "long.bin"
    run_lalamo(
        "speculator",
        "collect-traces",
        str(converted_model_dir),
        str(dataset_path),
        "--output-path",
        str(long_trace_path),
        "--batch-size",
        "1",
        "--num-logits-per-token",
        "8",
        "--num-tokens-to-generate",
        "32",
        "--max-output-length",
        "139",
    )

    with short_trace_path.open("rb") as short_trace_fd:
        short_traces = list(LalamoCompletion.deserialize_many(short_trace_fd))
    with long_trace_path.open("rb") as long_trace_fd:
        long_traces = list(LalamoCompletion.deserialize_many(long_trace_fd))

    assert len(short_traces) == len(long_traces)
    for short_trace, long_trace in zip(short_traces, long_traces, strict=True):
        assert short_trace.prefix_token_ids == long_trace.prefix_token_ids
        assert short_trace.completion_token_ids == long_trace.completion_token_ids
        assert len(short_trace.completion_token_logits) == len(long_trace.completion_token_logits)

        for short_logits, long_logits in zip(
            short_trace.completion_token_logits,
            long_trace.completion_token_logits,
            strict=True,
        ):
            short_keys = list(short_logits)
            long_keys = list(long_logits)
            assert short_keys == long_keys

            short_values = [short_logits[key] for key in short_keys]
            long_values = [long_logits[key] for key in long_keys]

            assert short_values == pytest.approx(long_values, abs=1e-7, rel=1e-7)


@pytest.mark.parametrize(
    "extra_args",
    [
        pytest.param(["--batch-size", "2"], id="batch-size"),
        pytest.param(["--vram-gb", "3"], id="vram"),
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
