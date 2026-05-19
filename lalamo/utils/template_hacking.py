import re


def fix_chat_template(template: str) -> str:
    generation_block_tag_regex = re.compile(r"{%-?\s*(?:generation|endgeneration)\s*-?%}")
    return generation_block_tag_regex.sub("", template)
