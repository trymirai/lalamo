import logging
import os
import time

import pytest

from lalamo.model_import.model_spec import LanguageModelSpec, ModelSpec
from lalamo.models import LanguageModel
from lalamo.models.chat_codec import UserMessage
from lalamo.models.language_model import GenerationConfig
from lalamo.module import Keychain
from tests.conftest import ConvertModel, filter_specs, load_converted_model, mark_by_size
from tests.model_test_tiers import ModelTier

from .common import DEFAULT_JUDGE_MODEL, TASK_PROMPT, judge

standard_llm_specs = filter_specs(model_type=LanguageModelSpec, max_tier=ModelTier.STANDARD)

log = logging.getLogger(__name__)

COHERENCE_MAX_TOKENS = 128


def _generate_single(
    model: LanguageModel,
    prompt: str,
    *,
    max_tokens: int,
) -> str:
    return model.reply(
        [UserMessage(prompt)],
        generation_config=GenerationConfig(temperature=0.0),
        max_output_length=max_tokens,
        keychain=Keychain.init(0),
    ).response


@pytest.mark.parametrize(
    "spec",
    mark_by_size(standard_llm_specs),
    ids=[s.origin.description for s in standard_llm_specs],
)
def test_model_coherent_and_stops(
    spec: ModelSpec,
    convert_model: ConvertModel,
) -> None:
    start_time = time.monotonic()
    converted_model_path = convert_model(spec.origin.description)
    log.info("Model conversion took %.1fs for %s", time.monotonic() - start_time, spec.origin.description)

    api_key = os.getenv("OPENROUTER_API_KEY")
    assert api_key is not None
    judge_model = os.getenv("COHERENCE_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)

    model = load_converted_model(converted_model_path)
    assert isinstance(model, LanguageModel)

    start_time = time.monotonic()
    coherence_output = _generate_single(
        model,
        TASK_PROMPT,
        max_tokens=COHERENCE_MAX_TOKENS,
    )
    log.info("Coherence generation took %.1fs for %s", time.monotonic() - start_time, spec.origin.description)

    assert coherence_output, "Model produced empty output for coherence prompt"
    log.info("Coherence output:\n%s", coherence_output)

    verdict = judge(api_key=api_key, model=judge_model, candidate_output=coherence_output, timeout=60)
    log.info(
        "Judge verdict: coherent=%s, score=%.2f, issues=%s, summary=%s",
        verdict.coherent,
        verdict.score,
        verdict.issues,
        verdict.summary,
    )
    assert verdict.coherent, (
        f"Output incoherent (score={verdict.score:.2f}, "
        f"issues={', '.join(verdict.issues) or 'none'}, summary={verdict.summary!r}):\n{coherence_output}"
    )
