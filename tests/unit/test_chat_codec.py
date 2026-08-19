import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from lalamo.model_import.model_spec import LanguageModelSpec
from lalamo.model_import.model_specs.gemma import GEMMA_MODELS
from lalamo.model_import.model_specs.granite import GRANITE_MODELS
from lalamo.model_import.model_specs.output_parser_regexes import (
    GEMMA4_OUTPUT_PARSER_REGEX,
    GRANITE_THINKING_OUTPUT_PARSER_REGEX,
)
from lalamo.model_import.model_specs.qwen import QWEN_MODELS
from lalamo.models.chat_codec import AssistantMessage, ChatCodec, ChatCodecConfig, ReasoningEffort


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
    mappings = {mapping.effort: mapping for mapping in model_spec.reasoning_effort_mappings}

    assert set(mappings) == {ReasoningEffort.XHIGH, ReasoningEffort.NONE}
    assert mappings[ReasoningEffort.XHIGH].parameter == parameter
    assert mappings[ReasoningEffort.XHIGH].value is True
    assert mappings[ReasoningEffort.NONE].parameter == parameter
    assert mappings[ReasoningEffort.NONE].value is False


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
    assert _chat_codec(output_parser_regex).parse_response(response) == expected
