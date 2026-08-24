OPTIONAL_THINKING_OUTPUT_PARSER_REGEX = (
    r"(?s)(?=(<think>|.*?</think>)?)"
    r"(?(1)(?:<think>)?(?P<chain_of_thought>.*?)(?:</think>\s*|\Z))"
    r"(?P<response>.*)\Z"
)
GEMMA4_OUTPUT_PARSER_REGEX = (
    r"(?s)(?:<\|channel>thought\n(?P<chain_of_thought>.*?)<channel\|>)?"
    r"(?P<response>(?:(?!<turn\|>)(?!<\|tool_response>).)+)?"
    r"(?:<turn\|>|<\|tool_response>)?\Z"
)
GRANITE_THINKING_OUTPUT_PARSER_REGEX = (
    r"(?s)<think>(?P<chain_of_thought>.*?)</think>\s*"
    r"<response>(?P<response>.*?)</response>\s*\Z"
)
GPT_OSS_OUTPUT_PARSER_REGEX = (
    r"(?s)(?:<\|channel\|>analysis<\|message\|>(?P<chain_of_thought>.*?))?"
    r"(?:(?:<\|end\|><\|start\|>assistant)?<\|channel\|>final<\|message\|>(?P<response>.*?))?"
    r"(?:<\|return\|>|<\|end\|>)?\Z"
)
