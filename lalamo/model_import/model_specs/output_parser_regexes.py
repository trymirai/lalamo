OPTIONAL_THINKING_OUTPUT_PARSER_REGEX = r"(?s)(?:<think>)?(?P<chain_of_thought>.*?)(?:</think>\s*(?P<response>.*))?\Z"
GPT_OSS_OUTPUT_PARSER_REGEX = r"(?s)analysis(?P<chain_of_thought>.*?)(?:assistantfinal(?P<response>.*))?\Z"
