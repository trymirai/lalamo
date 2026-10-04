import re


def fix_chat_template(template: str) -> str:
    generation_block_tag_regex = re.compile(r"{%-?\s*(?:generation|endgeneration)\s*-?%}")
    # LFM2.5-230M still quotes tool strings without escaping; use the current larger Liquid models' expression.
    template = template.replace(
        '''"'" + arg_value + "'"''',
        r""""'" + (arg_value | replace("\\", "\\\\") | replace("'", "\\'")"""
        r''' | replace("\n", "\\n") | replace("\r", "\\r")) + "'"''',
    )
    # Muse serializes optional OpenAI fields directly; Jinja's Undefined is not JSON serializable.
    template = template.replace("fn.description | tojson", "fn.description | default('') | tojson")
    template = template.replace("fn.parameters | tojson", "fn.parameters | default({}) | tojson")
    return generation_block_tag_regex.sub("", template)
