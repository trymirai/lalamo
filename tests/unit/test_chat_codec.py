from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from lalamo.model_import.model_specs.qwen import QWEN_THINKING_PARSER_REGEX
from lalamo.models.chat_codec import ChatCodecConfig


def _qwen_thinking_codec_config() -> ChatCodecConfig:
    return ChatCodecConfig(
        prompt_template="{{ '<|im_start|>assistant' }}",
        output_parser_regex=QWEN_THINKING_PARSER_REGEX,
        system_role_name="system",
        user_role_name="user",
        assistant_role_name="assistant",
        eos_token=None,
        bos_token=None,
    )


def _tokenizer() -> Tokenizer:
    return Tokenizer(WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]"))


def test_qwen_thinking_parser_extracts_answer_after_thinking_separator() -> None:
    codec = _qwen_thinking_codec_config().init(_tokenizer())

    message = codec.parse_response("reasoning\n</think>\n\n42")

    assert message.chain_of_thought == "reasoning"
    assert message.response == "42"


def test_qwen_thinking_parser_strips_leading_think_marker() -> None:
    codec = _qwen_thinking_codec_config().init(_tokenizer())

    message = codec.parse_response("<think>\nfoo\n</think>\nbar")

    assert message.chain_of_thought == "foo"
    assert message.response == "bar"


def test_qwen_thinking_parser_hides_unfinished_thinking() -> None:
    codec = _qwen_thinking_codec_config().init(_tokenizer())

    message = codec.parse_response("I need to reason through the options first.")

    assert message.chain_of_thought == "I need to reason through the options first."
    assert message.response == ""


def test_plain_parser_without_regex_preserves_raw_response() -> None:
    config = ChatCodecConfig(
        prompt_template="{{ messages[-1].content }}",
        output_parser_regex=None,
        system_role_name="system",
        user_role_name="user",
        assistant_role_name="assistant",
        eos_token=None,
        bos_token=None,
    )
    codec = config.init(_tokenizer())

    message = codec.parse_response("direct answer")

    assert message.chain_of_thought is None
    assert message.response == "direct answer"
