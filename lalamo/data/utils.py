from collections.abc import Iterable

from lalamo.message_processor import Message, UserMessage


def is_non_empty_user_message(message: Message) -> bool:
    if type(message) is not UserMessage:
        return False

    content = message.content
    if isinstance(content, str):
        return bool(content.strip())

    return any(isinstance(block, str) and bool(block.strip()) for block in content)


def get_prefixes_ending_in_user_message(conversation: Iterable[Message]) -> list[list[Message]]:
    conversation = list(conversation)
    return [conversation[: i + 1] for i, msg in enumerate(conversation) if is_non_empty_user_message(msg)]
