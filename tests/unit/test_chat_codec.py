import pytest
from frozendict import frozendict
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from lalamo.model_import.model_specs.output_parser_regexes import (
    GEMMA4_OUTPUT_PARSER_REGEX,
    GRANITE_THINKING_OUTPUT_PARSER_REGEX,
    OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
)
from lalamo.model_import.model_specs.reasoning_configs import BOOLEAN_REASONING_DEFAULT_ON_CONFIG
from lalamo.models.chat_codec import (
    AssistantMessage,
    ChatCodec,
    ChatCodecConfig,
    ReasoningConfig,
    ReasoningEffort,
    UserMessage,
)


def _chat_codec(
    *,
    prompt_template: str = "",
    output_parser_regex: str | None = None,
    reasoning_config: ReasoningConfig | None = None,
) -> ChatCodec:
    return ChatCodecConfig(
        prompt_template=prompt_template,
        output_parser_regex=output_parser_regex,
        system_role_name="system",
        user_role_name="user",
        assistant_role_name="assistant",
        eos_token=None,
        bos_token=None,
        reasoning_config=reasoning_config,
    ).init(Tokenizer(WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]")))


def test_reasoning_config_requires_a_mapping_for_its_default() -> None:
    with pytest.raises(ValueError, match="default reasoning effort"):
        ReasoningConfig(
            default_reasoning_effort=ReasoningEffort.MEDIUM,
            field_name="reasoning_effort",
            reasoning_effort_to_field_value=frozendict({ReasoningEffort.LOW: "low"}),
        )


def test_reasoning_effort_is_rendered_through_the_configured_field() -> None:
    codec = _chat_codec(
        prompt_template="{{ reasoning_effort }}",
        reasoning_config=ReasoningConfig(
            default_reasoning_effort=ReasoningEffort.MEDIUM,
            field_name="reasoning_effort",
            reasoning_effort_to_field_value=frozendict(
                {
                    ReasoningEffort.LOW: "low",
                    ReasoningEffort.MEDIUM: "medium",
                }
            ),
        ),
    )

    assert codec.render_request([UserMessage("hello")]) == "medium"
    assert codec.render_request([UserMessage("hello")], reasoning_effort=ReasoningEffort.LOW) == "low"

    with pytest.raises(ValueError, match="not supported"):
        codec.render_request([UserMessage("hello")], reasoning_effort=ReasoningEffort.HIGH)


def test_model_without_reasoning_config_rejects_an_explicit_effort() -> None:
    codec = _chat_codec(prompt_template="{{ reasoning_effort | default('unset') }}")

    assert codec.render_request([UserMessage("hello")]) == "unset"
    with pytest.raises(ValueError, match="does not support configurable reasoning effort"):
        codec.render_request([UserMessage("hello")], reasoning_effort=ReasoningEffort.MEDIUM)


def test_boolean_template_field_uses_medium_as_enabled() -> None:
    codec = _chat_codec(
        prompt_template="{% if enable_thinking %}on{% else %}off{% endif %}",
        reasoning_config=BOOLEAN_REASONING_DEFAULT_ON_CONFIG,
    )

    assert codec.render_request([UserMessage("hello")]) == "on"
    assert (
        codec.render_request(
            [UserMessage("hello")],
            reasoning_effort=ReasoningEffort.NO_REASONING,
        )
        == "off"
    )


@pytest.mark.parametrize(
    ("output_parser_regex", "full_output"),
    [
        (OPTIONAL_THINKING_OUTPUT_PARSER_REGEX, "<think>reasoning</think>answer"),
        (GRANITE_THINKING_OUTPUT_PARSER_REGEX, "<think>reasoning</think><response>answer</response>"),
        (GEMMA4_OUTPUT_PARSER_REGEX, "<|channel>thought\nreasoning<channel|>answer<turn|>"),
    ],
    ids=["optional-thinking", "granite", "gemma4"],
)
def test_generation_is_parsed_at_every_truncation_stage(output_parser_regex: str, full_output: str) -> None:
    codec = _chat_codec(output_parser_regex=output_parser_regex)
    mid_thinking = full_output[: full_output.index("reasoning") + len("reas")]
    mid_response = full_output[: full_output.index("answer") + len("answ")]

    assert codec.parse_response("answer") == AssistantMessage(chain_of_thought=None, response="answer")
    assert codec.parse_response(mid_thinking) == AssistantMessage(chain_of_thought="reas", response="")
    assert codec.parse_response(mid_response) == AssistantMessage(chain_of_thought="reasoning", response="answ")
    assert codec.parse_response(full_output) == AssistantMessage(chain_of_thought="reasoning", response="answer")


def test_optional_thinking_parses_response_without_an_opening_tag() -> None:
    codec = _chat_codec(output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX)

    expected = AssistantMessage(chain_of_thought="reasoning", response="answer")
    assert codec.parse_response("reasoning</think>answer") == expected


def test_granite_parses_a_response_missing_its_wrapper() -> None:
    codec = _chat_codec(output_parser_regex=GRANITE_THINKING_OUTPUT_PARSER_REGEX)

    expected = AssistantMessage(chain_of_thought="reasoning", response="answer")
    assert codec.parse_response("<think>reasoning</think>answer") == expected
