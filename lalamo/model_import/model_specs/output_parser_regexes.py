OPTIONAL_THINKING_OUTPUT_PARSER_REGEX = r"(?s)(?:<think>)?(?P<chain_of_thought>.*?)(?:</think>\s*(?P<response>.*))?\Z"
GPT_OSS_OUTPUT_PARSER_REGEX = (
    r"(?s)(?:<\|channel\|>analysis<\|message\|>(?P<chain_of_thought>.*?))?"
    r"(?:(?:<\|end\|><\|start\|>assistant)?<\|channel\|>final<\|message\|>(?P<response>.*?))?"
    r"(?:<\|return\|>|<\|end\|>)?\Z"
)
