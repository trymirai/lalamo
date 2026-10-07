import ast
import json
import re
from itertools import chain, product
from typing import ClassVar, cast

from lalamo.model_import.model_configs import HFLFM2Config
from lalamo.model_import.model_spec import ConfigMap, FileSpec, LanguageModelSpec
from lalamo.model_import.model_specs.output_parsers import ThinkingResponseParser
from lalamo.model_import.origins import HuggingFaceOrigin
from lalamo.models.chat_codec import FunctionCall, ResponseParser, ToolCall, ToolSchema
from lalamo.models.language_model import GenerationConfig
from lalamo.utils.json import JSON

__all__ = ["LFM2_MODELS"]


class _JsonLiterals(ast.NodeTransformer):
    """Liquid writes nested values as JSON, whose literals read as Python names."""

    def visit_Name(self, node: ast.Name) -> ast.Constant:
        literals = {"true": True, "false": False, "null": None}
        if node.id not in literals:
            raise ValueError(f"Unexpected name {node.id!r} in a tool argument.")
        return ast.Constant(literals[node.id])


class LiquidResponseParser(ResponseParser):
    tool_call_tags: ClassVar[tuple[str, str] | None] = ("<|tool_call_start|>", "<|tool_call_end|>")

    @classmethod
    def parse_tool_calls(cls, body: str, tools: tuple[ToolSchema, ...]) -> tuple[ToolCall, ...]:
        names = (cast("str", cast("dict[str, JSON]", tool["function"])["name"]) for tool in tools)
        # Liquid calls are Python expressions, which cannot contain the hyphens OpenAI allows in names. Outside string
        # literals, such names are replaced by non-ASCII identifiers, which cannot collide with ASCII OpenAI names.
        aliases = {name: f"tool_\u03b1{index}" for index, name in enumerate(names)}
        hyphenated = "|".join(re.escape(name) for name in aliases if not name.isidentifier())
        if hyphenated:
            body = re.sub(
                rf"""("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|(?<![\w.-])({hyphenated})(?=\s*\()""",
                lambda match: match[1] or aliases[match[2]],
                body,
            )
        names_by_alias = {alias: name for name, alias in aliases.items()}
        expression = ast.parse(body.strip(), mode="eval").body
        calls = []
        expressions = [expression]
        if isinstance(expression, ast.List):
            expressions = expression.elts
        if not expressions:
            raise ValueError("Malformed tool call.")
        for call in expressions:
            if (
                not isinstance(call, ast.Call)
                or not isinstance(call.func, ast.Name)
                or call.args
                or any(keyword.arg is None for keyword in call.keywords)
                or len({keyword.arg for keyword in call.keywords}) != len(call.keywords)
            ):
                raise ValueError("Malformed tool call.")
            name = call.func.id
            for node in ast.walk(call):
                if isinstance(node, (ast.Set, ast.Tuple)) or (
                    isinstance(node, ast.Dict)
                    and any(not isinstance(key, ast.Constant) or not isinstance(key.value, str) for key in node.keys)
                ):
                    raise ValueError("Tool arguments must be JSON values.")
            arguments = {
                cast("str", keyword.arg): ast.literal_eval(_JsonLiterals().visit(keyword.value))
                for keyword in call.keywords
            }
            try:
                json.dumps(arguments, allow_nan=False)
            except TypeError as error:
                raise ValueError("Tool arguments must be JSON values.") from error
            calls.append(
                ToolCall(
                    type="function", function=FunctionCall(name=names_by_alias.get(name, name), arguments=arguments)
                )
            )
        return tuple(calls)


class LiquidThinkingResponseParser(LiquidResponseParser, ThinkingResponseParser):
    pass


def _lfm_repo(family: str, size: str, variant: str | None, quantization_bits: int | None) -> tuple[str, str]:
    return (
        "LiquidAI" if quantization_bits is None else "mlx-community",
        f"{family}-{size}"
        f"{f'-{variant}' if variant is not None else ''}"
        f"{f'-{quantization_bits}bit' if quantization_bits is not None else ''}",
    )


_LFM20_MODELS = [
    LanguageModelSpec(
        vendor="LiquidAI",
        family="LFM2",
        name=_lfm_repo("LFM2", size, variant, quantization_bits)[1],
        size=size,
        origin=HuggingFaceOrigin(repo="/".join(_lfm_repo("LFM2", size, variant, quantization_bits))),
        config_type=HFLFM2Config,
        configs=ConfigMap(
            generation_config=GenerationConfig(temperature=0.3, min_p=0.15),  # , repetition_penalty=1.05
            chat_template=FileSpec("chat_template.jinja"),
        ),
    )
    for size, variant, quantization_bits in chain(
        product(["350M", "700M", "1.2B", "2.6B"], [None], [None, 4, 8]),
    )
]

_LFM25_MODEL_SPECS = (
    ("LiquidAI", "LFM2.5-350M", "350M", None, LiquidResponseParser),
    ("LiquidAI", "LFM2.5-1.2B-Instruct", "1.2B", None, LiquidResponseParser),
    ("LiquidAI", "LFM2.5-1.2B-Instruct-MLX-4bit", "1.2B", 4, LiquidResponseParser),
    ("LiquidAI", "LFM2.5-1.2B-Instruct-MLX-8bit", "1.2B", 8, LiquidResponseParser),
    ("LiquidAI", "LFM2.5-1.2B-Thinking", "1.2B", None, LiquidThinkingResponseParser),
    ("mlx-community", "LFM2.5-1.2B-Thinking-4bit", "1.2B", 4, LiquidThinkingResponseParser),
    ("mlx-community", "LFM2.5-1.2B-Thinking-8bit", "1.2B", 8, LiquidThinkingResponseParser),
)

_LFM25_MODELS = [
    LanguageModelSpec(
        vendor="LiquidAI",
        family="LFM2.5",
        name=name,
        size=size,
        origin=HuggingFaceOrigin(repo=f"{repo_owner}/{name}"),
        config_type=HFLFM2Config,
        configs=ConfigMap(
            generation_config=GenerationConfig(temperature=0.1, top_k=50, top_p=0.1),  # , repetition_penalty=1.05
            chat_template=FileSpec("chat_template.jinja"),
        ),
        response_parser=response_parser,
    )
    for repo_owner, name, size, _quantization_bits, response_parser in _LFM25_MODEL_SPECS
]

LFM2_MODELS = _LFM20_MODELS + _LFM25_MODELS
