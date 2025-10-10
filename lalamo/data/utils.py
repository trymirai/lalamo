from collections.abc import Iterable

from lalamo.message_processor import Message, UserMessage


def get_prefixes_ending_in_user_message(conversation: Iterable[Message]) -> list[list[Message]]:
    conversation = list(conversation)
    return [conversation[: i + 1] for i, msg in enumerate(conversation) if isinstance(msg, UserMessage)]
