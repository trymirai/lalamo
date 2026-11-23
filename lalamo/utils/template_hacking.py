import re


def fix_chat_template(template: str) -> str:
    generation_block_tag_regex = re.compile(r"{%-?\s*(?:generation|endgeneration)\s*-?%}")
    # Muse serializes optional OpenAI fields directly; Jinja's Undefined is not JSON serializable.
    template = template.replace("fn.description | tojson", "fn.description | default('') | tojson")
    template = template.replace("fn.parameters | tojson", "fn.parameters | default({}) | tojson")
    return generation_block_tag_regex.sub("", template)
