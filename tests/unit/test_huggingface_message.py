import pytest
from cattrs.errors import ClassValidationError

from lalamo.data.huggingface_message import HFMessage
from lalamo.models.chat_codec import AssistantMessage


def test_assistant_reasoning_is_kept() -> None:
    message = HFMessage.from_dict({"role": "assistant", "content": "answer", "reasoning_content": "thoughts"})
    assert message.as_message() == AssistantMessage(chain_of_thought="thoughts", response="answer")


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ClassValidationError):
        HFMessage.from_dict({"role": "assistant", "content": "answer", "reasoning": "thoughts"})
