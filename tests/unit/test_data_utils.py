from lalamo.data.utils import get_prompt_ending_in_user_message
from lalamo.models.chat_codec import AssistantMessage, SystemMessage, UserMessage


def test_get_prompt_ending_in_user_message_drops_final_assistant() -> None:
    conversation = [
        SystemMessage("You are helpful."),
        UserMessage("hello"),
        AssistantMessage(None, "hi"),
        UserMessage("continue"),
        AssistantMessage(None, "sure"),
    ]

    assert get_prompt_ending_in_user_message(conversation) == conversation[:-1]


def test_get_prompt_ending_in_user_message_keeps_prompt_only_row() -> None:
    conversation = [UserMessage("hello"), AssistantMessage(None, "hi"), UserMessage("continue")]

    assert get_prompt_ending_in_user_message(conversation) == conversation


def test_get_prompt_ending_in_user_message_rejects_non_user_or_empty_end() -> None:
    assert get_prompt_ending_in_user_message([SystemMessage("system")]) is None
    assert get_prompt_ending_in_user_message([UserMessage("   ")]) is None
