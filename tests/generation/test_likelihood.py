import jax.numpy as jnp
import pytest

from lalamo.message_processor import UserMessage
from lalamo.model_import import REPO_TO_MODEL, import_model
from lalamo.models import LanguageModel
from tests.common import assert_close

MODEL_LIST = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "LiquidAI/LFM2-700M",
    "cartesia-ai/Llamba-1B",
]

QA_PAIRS = [
    ("Do birds fly?", "Yes"),
    ("What is 2+2?", "4"),
    ("Is water wet?", "Yes, water is wet."),
    ("What color is the sky?", "Blue"),
    ("Is the sun a star?", "Yes, the sun is a star."),
    ("What is the capital of France?", "Paris"),
    ("Do fish swim?", "Yes"),
    ("Is ice cold?", "Yes, ice is cold."),
    ("How many legs does a dog have?", "Four"),
    ("What is the opposite of hot?", "Cold"),
    ("Is 1 greater than 0?", "Yes"),
    ("What language is spoken in Japan?", "Japanese"),
    ("Do cats meow?", "Yes, cats meow."),
    ("Is the earth round?", "Yes, the earth is approximately round."),
    ("What is H2O?", "Water"),
    ("Can humans breathe underwater?", "No"),
    ("Is fire hot?", "Yes"),
    ("What comes after Monday?", "Tuesday"),
    ("Do plants need sunlight?", "Yes, plants need sunlight to grow."),
    ("Is Python a programming language?", "Yes, Python is a programming language."),
]


@pytest.fixture(params=MODEL_LIST)
def language_model(request: pytest.FixtureRequest) -> LanguageModel:
    model = import_model(REPO_TO_MODEL[request.param]).model
    assert isinstance(model, LanguageModel)
    return model


def _tokenize_qa(model: LanguageModel) -> tuple[list[list[int]], list[list[int]]]:
    inputs = [model.message_processor.tokenize_request([UserMessage(q)]) for q, _ in QA_PAIRS]
    outputs = [model.message_processor.tokenize_text(a) for _, a in QA_PAIRS]
    return inputs, outputs


def test_shapes_and_quality(language_model: LanguageModel) -> None:
    inputs, outputs = _tokenize_qa(language_model)
    results = language_model.loglikelihood(inputs, outputs, top_k=5, batch_size=4)

    assert len(results) == len(QA_PAIRS)
    for i, result in enumerate(results):
        assert len(result.logits) == len(outputs[i])
        assert len(result.tokens) == len(outputs[i])
        assert all(len(row) == 5 for row in result.logits)
        assert all(len(row) == 5 for row in result.tokens)

    # "Do birds fly?" -> "Yes" should be top-1
    assert outputs[0][0] == results[0].tokens[0][0]


def test_consistency_across_batch_sizes(language_model: LanguageModel) -> None:
    inputs, outputs = _tokenize_qa(language_model)
    results_bs1 = language_model.loglikelihood(inputs, outputs, top_k=5, batch_size=1)
    results_bs7 = language_model.loglikelihood(inputs, outputs, top_k=5, batch_size=7)

    for i in range(len(QA_PAIRS)):
        assert_close(
            result=jnp.array(results_bs1[i].logits),
            reference=jnp.array(results_bs7[i].logits),
            operation_name=f"row {i} logits bs=1 vs bs=7",
        )
