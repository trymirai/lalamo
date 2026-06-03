from collections.abc import Iterable

from lalamo.commands import count_collectable_prompt_token_ids
from lalamo.models import LanguageModel
from lalamo.models.chat_codec import AssistantMessage, Message, SystemMessage, UserMessage
from tests.conftest import RunLalamo


class FakeTokenCodec:
    def encode_request(self, request: Iterable[Message]) -> list[int]:
        *_, last_message = request
        assert isinstance(last_message, UserMessage)
        assert isinstance(last_message.content, str)
        return list(range(len(last_message.content.split())))


def test_count_collectable_prompt_token_ids_applies_prompt_policy_and_max_length() -> None:
    model = object.__new__(LanguageModel)
    object.__setattr__(model, "token_codec", FakeTokenCodec())

    assert (
        count_collectable_prompt_token_ids(
            model,
            [
                [UserMessage("one two")],
                [UserMessage("one two three")],
                [UserMessage("one"), AssistantMessage(None, "target")],
                [SystemMessage("system")],
                [UserMessage("  ")],
            ],
            max_input_length=2,
        )
        == 2
    )


def test_collect_traces_help(run_lalamo: RunLalamo) -> None:
    output = run_lalamo("collect-traces", "--help")

    assert "MODEL_PATH" in output
    assert "DATASET_PATH" in output
    assert "--output-path" in output
    assert "--num-tokens-to-generate" not in output
