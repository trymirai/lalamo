import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from lalamo.model_import.model_spec import LanguageModelSpec
from lalamo.model_import.model_specs.gemma import GEMMA_MODELS
from lalamo.model_import.model_specs.granite import GRANITE_MODELS
from lalamo.model_import.model_specs.output_parser_regexes import (
    GEMMA4_OUTPUT_PARSER_REGEX,
    GRANITE_THINKING_OUTPUT_PARSER_REGEX,
    OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
)
from lalamo.model_import.model_specs.qwen import QWEN_MODELS
from lalamo.models.chat_codec import AssistantMessage, ChatCodec, ChatCodecConfig, ReasoningEffort
from lalamo.models.language_model import GenerationConfig


def _chat_codec(output_parser_regex: str) -> ChatCodec:
    return ChatCodecConfig(
        prompt_template="",
        output_parser_regex=output_parser_regex,
        system_role_name="system",
        user_role_name="user",
        assistant_role_name="assistant",
        eos_token=None,
        bos_token=None,
        end_of_thinking_tag=None,
    ).init(Tokenizer(WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]")))


@pytest.mark.parametrize(
    ("model_spec", "parameter"),
    [
        (next(spec for spec in QWEN_MODELS if spec.name == "Qwen3.5-0.8B"), "enable_thinking"),
        (next(spec for spec in GEMMA_MODELS if spec.name == "gemma-4-e2b-it"), "enable_thinking"),
        (next(spec for spec in GRANITE_MODELS if spec.name == "granite-3.3-2b-instruct"), "thinking"),
    ],
)
def test_default_off_models_support_xhigh(model_spec: LanguageModelSpec, parameter: str) -> None:
    (reasoning_effort_mapping,) = model_spec.reasoning_effort_mappings
    generation_params = model_spec.configs.generation_params_overrides

    assert generation_params is not None
    assert generation_params.reasoning_effort is ReasoningEffort.NONE
    assert reasoning_effort_mapping.effort is ReasoningEffort.XHIGH
    assert reasoning_effort_mapping.parameter == parameter
    assert reasoning_effort_mapping.value is True


def test_default_off_reasoning_effort_keeps_plain_response() -> None:
    generation_config = GenerationConfig(reasoning_effort=ReasoningEffort.NONE)
    reasoning_effort = generation_config.resolve_reasoning_effort(None)

    assert reasoning_effort is ReasoningEffort.NONE
    assert _chat_codec(OPTIONAL_THINKING_OUTPUT_PARSER_REGEX).parse_response(
        "answer",
        expect_thinking=reasoning_effort.is_enabled,
    ) == AssistantMessage(chain_of_thought=None, response="answer")


@pytest.mark.parametrize(
    ("output_parser_regex", "response", "expected"),
    [
        (
            GRANITE_THINKING_OUTPUT_PARSER_REGEX,
            "<think>reasoning</think><response>answer</response>",
            AssistantMessage(chain_of_thought="reasoning", response="answer"),
        ),
        (
            GRANITE_THINKING_OUTPUT_PARSER_REGEX,
            "answer",
            AssistantMessage(chain_of_thought=None, response="answer"),
        ),
        (
            GEMMA4_OUTPUT_PARSER_REGEX,
            "<|channel>thought\nreasoning<channel|>answer<turn|>",
            AssistantMessage(chain_of_thought="reasoning", response="answer"),
        ),
        (
            GEMMA4_OUTPUT_PARSER_REGEX,
            "answer<turn|>",
            AssistantMessage(chain_of_thought=None, response="answer"),
        ),
    ],
)
def test_reasoning_response_parsing(
    output_parser_regex: str,
    response: str,
    expected: AssistantMessage,
) -> None:
    assert _chat_codec(output_parser_regex).parse_response(response, expect_thinking=True) == expected
