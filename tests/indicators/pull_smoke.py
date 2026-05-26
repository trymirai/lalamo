import re
from dataclasses import dataclass
from pathlib import Path

import pytest
from safetensors import safe_open
from tokenizers import Tokenizer

from tests.conftest import ConvertModel, RunLalamo, strip_ansi_escape
from tests.model_test_tiers import ModelTier, get_models_by_tier

PULL_MODEL_REPO = "google/gemma-3-1b-it"
PULL_SIGNATURE_MODEL_REPOS = get_models_by_tier(ModelTier.CANONICAL) + get_models_by_tier(ModelTier.CORE)
MATH_PROMPT = "What is 2 + 2? Reply with a single number, nothing else."
YES_NO_PROMPT = "Are apples fruits? Answer with one word: yes or no."
MAX_RESPONSE_TOKENS = 30


@dataclass(frozen=True)
class TensorSignature:
    dtype: str
    shape: tuple[int, ...]


def _pull_model_dir(
    repo: str,
    tmp_path_factory: pytest.TempPathFactory,
    run_lalamo: RunLalamo,
) -> Path:
    output_dir = tmp_path_factory.mktemp("pulled_models") / repo.replace("/", "__")
    run_lalamo("pull", repo, "--output-dir", str(output_dir))

    assert (output_dir / "config.json").exists(), f"Missing config.json in {output_dir}"
    assert (output_dir / "tokenizer.json").exists(), f"Missing tokenizer.json in {output_dir}"
    assert any(output_dir.glob("model*.safetensors")), f"Missing model weights in {output_dir}"
    return output_dir


def _tensor_signatures(model_dir: Path) -> dict[str, TensorSignature]:
    weight_paths = tuple(sorted(model_dir.glob("model*.safetensors")))
    assert weight_paths, f"Missing model weights in {model_dir}"

    signatures: dict[str, TensorSignature] = {}
    for weight_path in weight_paths:
        with safe_open(weight_path, framework="numpy") as tensors_file:
            for tensor_name in sorted(tensors_file.keys()):
                if tensor_name in signatures:
                    raise AssertionError(f"Duplicate tensor {tensor_name!r} across safetensors files in {model_dir}")

                tensor_slice = tensors_file.get_slice(tensor_name)
                signatures[tensor_name] = TensorSignature(
                    dtype=tensor_slice.get_dtype(),
                    shape=tuple(tensor_slice.get_shape()),
                )

    return signatures


def _format_tensor_signature_diffs(
    *,
    converted_signatures: dict[str, TensorSignature],
    pulled_signatures: dict[str, TensorSignature],
    limit: int = 20,
) -> str:
    converted_names = set(converted_signatures)
    pulled_names = set(pulled_signatures)
    missing_from_pull = sorted(converted_names - pulled_names)
    extra_in_pull = sorted(pulled_names - converted_names)
    mismatched = {
        tensor_name: (
            converted_signatures[tensor_name],
            pulled_signatures[tensor_name],
        )
        for tensor_name in sorted(converted_names & pulled_names)
        if converted_signatures[tensor_name] != pulled_signatures[tensor_name]
    }

    lines: list[str] = []
    if missing_from_pull:
        lines.append(f"Tensors missing from pulled artifact ({len(missing_from_pull)}): {missing_from_pull[:limit]}")
    if extra_in_pull:
        lines.append(f"Extra tensors in pulled artifact ({len(extra_in_pull)}): {extra_in_pull[:limit]}")
    if mismatched:
        lines.append(f"Tensor signature mismatches ({len(mismatched)}):")
        for tensor_name, (converted_signature, pulled_signature) in tuple(mismatched.items())[:limit]:
            lines.append(f"  {tensor_name}: converted={converted_signature}, pulled={pulled_signature}")

    return "\n".join(lines)


def test_pulled_model_generates_adequate_output(
    tmp_path_factory: pytest.TempPathFactory,
    run_lalamo: RunLalamo,
) -> None:
    pulled_model_dir = _pull_model_dir(PULL_MODEL_REPO, tmp_path_factory, run_lalamo)
    responses = [
        strip_ansi_escape(
            run_lalamo(
                "chat",
                str(pulled_model_dir),
                "--message",
                prompt,
                "--max-tokens",
                "64",
                "--temperature",
                "0",
            ),
        )
        for prompt in [MATH_PROMPT, YES_NO_PROMPT]
    ]

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


@pytest.mark.parametrize("repo", PULL_SIGNATURE_MODEL_REPOS, ids=PULL_SIGNATURE_MODEL_REPOS)
def test_pulled_canonical_and_core_model_tensor_signatures_match_fresh_conversion(
    repo: str,
    tmp_path_factory: pytest.TempPathFactory,
    run_lalamo: RunLalamo,
    convert_model: ConvertModel,
) -> None:
    pulled_model_dir = _pull_model_dir(repo, tmp_path_factory, run_lalamo)
    converted_model_dir = convert_model(repo, cached=True)
    converted_signatures = _tensor_signatures(converted_model_dir)
    pulled_signatures = _tensor_signatures(pulled_model_dir)

    diff_message = _format_tensor_signature_diffs(
        converted_signatures=converted_signatures,
        pulled_signatures=pulled_signatures,
    )

    if pulled_signatures != converted_signatures:
        raise AssertionError(
            f"pulled model tensor signatures differ from a fresh conversion for {repo}.\n{diff_message}"
        )
