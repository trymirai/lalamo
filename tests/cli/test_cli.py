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

B_TREE_PROMPT = "Implement a B-tree data structure in Rust concisely"
CAPITAL_PROMPT = "What's the capital of the United Kingdom? No thinking, answer right away."
APPLES_PROMPT = "Are apples fruits? Answer only yes or no, without thinking, answer right away."


QA_CONVERSATIONS = [
    [{"role": "user", "content": CAPITAL_PROMPT}],
    [{"role": "user", "content": APPLES_PROMPT}],
]


def _write_qa_dataset(path: Path) -> None:
    pl.DataFrame({"conversation": QA_CONVERSATIONS}).write_parquet(path)


def _read_responses(path: Path) -> list[str]:
    return pl.read_parquet(path).get_column("response").to_list()


def _assert_has_london_and_yes(texts: list[str]) -> None:
    joined = " ".join(texts).lower()
    assert "london" in joined, f"Expected 'london' in {texts!r}"
    assert "yes" in joined, f"Expected 'yes' in {texts!r}"


def strip_ansi_escape(s: str) -> str:
    escape_regex = re.compile(r"\x1b\[[0-9;]*m")
    return escape_regex.sub("", s)


def _read_traces(trace_path: Path) -> list[LalamoCompletion]:
    with open(trace_path, "rb") as trace_fd:
        return list(LalamoCompletion.deserialize_many(trace_fd))


@pytest.fixture(scope="module", params=MODELS, ids=str)
def converted_model_dir(request: pytest.FixtureRequest, convert_model: ConvertModel) -> Path:
    return convert_model(request.param)


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
    tmp_path_factory: pytest.TempPathFactory,
    run_lalamo: RunLalamo,
) -> None:
    work_dir = tmp_path_factory.mktemp("max_output_length_consistency")

    dataset_path = work_dir / "dataset.parquet"
    prompts = [B_TREE_PROMPT]
    pl.DataFrame({"conversation": [[{"role": "user", "content": prompt}] for prompt in prompts]}).write_parquet(
        dataset_path,
    )

    short_trace_path = work_dir / "short.bin"
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

    long_trace_path = work_dir / "long.bin"
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

    short_traces = _read_traces(short_trace_path)
    long_traces = _read_traces(long_trace_path)

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


def test_generate_replies_batch_size(
    converted_model_dir: Path,
    tmp_path_factory: pytest.TempPathFactory,
    run_lalamo: RunLalamo,
) -> None:
    work_dir = tmp_path_factory.mktemp("generate_replies_batch_size")
    dataset_path = work_dir / "dataset.parquet"
    output_path = work_dir / "replies.parquet"
    _write_qa_dataset(dataset_path)

    run_lalamo(
        "generate-replies",
        str(converted_model_dir),
        str(dataset_path),
        "--output-path",
        str(output_path),
        "--batch-size",
        "2",
    )

    _assert_has_london_and_yes(_read_responses(output_path))


def test_generate_replies_vram(
    converted_model_dir: Path,
    tmp_path_factory: pytest.TempPathFactory,
    run_lalamo: RunLalamo,
) -> None:
    work_dir = tmp_path_factory.mktemp("generate_replies_vram")
    dataset_path = work_dir / "dataset.parquet"
    output_path = work_dir / "replies.parquet"
    _write_qa_dataset(dataset_path)

    run_lalamo(
        "generate-replies",
        str(converted_model_dir),
        str(dataset_path),
        "--output-path",
        str(output_path),
        "--vram-gb",
        "3",
    )

    _assert_has_london_and_yes(_read_responses(output_path))


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
    tmp_path_factory: pytest.TempPathFactory,
    run_lalamo: RunLalamo,
) -> None:
    work_dir = tmp_path_factory.mktemp("collect_traces_answers")
    dataset_path = work_dir / "dataset.parquet"
    trace_path = work_dir / "traces.bin"
    _write_qa_dataset(dataset_path)

    run_lalamo(
        "speculator",
        "collect-traces",
        str(converted_model_dir),
        str(dataset_path),
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
