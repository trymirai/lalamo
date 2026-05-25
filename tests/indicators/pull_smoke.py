import re
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from tests.conftest import RunLalamo, strip_ansi_escape

PULL_MODEL_REPO = "google/gemma-3-1b-it"
MATH_PROMPT = "What is 2 + 2? Reply with a single number, nothing else."
YES_NO_PROMPT = "Are apples fruits? Answer with one word: yes or no."
MAX_RESPONSE_TOKENS = 30


def _has_model_weights(model_dir: Path) -> bool:
    return (model_dir / "model.safetensors").exists() or any(model_dir.glob("model*.safetensors"))


@pytest.fixture(scope="session")
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
    run_lalamo: RunLalamo,
) -> None:
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
