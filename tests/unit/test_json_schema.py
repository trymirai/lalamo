import json
from collections.abc import Sequence

import pytest
import xgrammar as xgr
from jsonschema import Draft202012Validator

from lalamo.models.json_schema import check_json_schema_syntax, validate_json_schema
from lalamo.utils.json import JSON

pytestmark = pytest.mark.fast


@pytest.fixture(scope="module")
def compiler() -> xgr.GrammarCompiler:
    return xgr.GrammarCompiler(
        xgr.TokenizerInfo([bytes([index]) for index in range(256)] + [b"<eos>"], stop_token_ids=[256]),
        max_threads=1,
    )


@pytest.mark.parametrize(
    ("schema", "accepted", "rejected", "strict"),
    [
        (
            {
                "type": "object",
                "properties": {
                    "pattern": {"$ref": "#/$defs/Answer"},
                    "format": {"anyOf": [{"type": "string", "minLength": 1, "maxLength": 1}, {"type": "null"}]},
                    "anyOf": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": -2, "maximum": 3},
                        "minItems": 1,
                        "maxItems": 2,
                    },
                },
                "required": ["pattern", "format", "anyOf"],
                "additionalProperties": False,
                "$defs": {"Answer": {"type": "string", "enum": ["yes", "no"]}},
                "examples": [{"not": {"pattern": "this is example data"}}],
                "default": {"allOf": ["annotation data"]},
            },
            ['{"pattern":"yes","format":"😀","anyOf":[1]}', '{"pattern":"no","format":null,"anyOf":[-2,3]}'],
            [
                '{"pattern":"yes","anyOf":[1]}',
                '{"pattern":"yes","format":"á","anyOf":[1]}',
                '{"pattern":"yes","format":null,"anyOf":[4]}',
                '{"pattern":"yes","format":null,"anyOf":[]}',
                '{"pattern":"yes","format":null,"anyOf":[1],"extra":true}',
            ],
            True,
        ),
        (
            {
                "type": "object",
                "properties": {"value": {"type": "integer"}, "next": {"anyOf": [{"type": "null"}, {"$ref": "#"}]}},
                "required": ["value", "next"],
                "additionalProperties": False,
            },
            ['{"value":1,"next":null}', '{"value":1,"next":{"value":2,"next":null}}'],
            ['{"value":1,"next":{"value":"2","next":null}}', '{"value":1,"next":{}}'],
            True,
        ),
        (
            {
                "type": "object",
                "properties": {"node": {"$ref": "#/$defs/Node"}},
                "required": ["node"],
                "additionalProperties": False,
                "$defs": {
                    "Node": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "string"},
                            "children": {"type": "array", "items": {"$ref": "#/$defs/Node"}},
                        },
                        "required": ["value", "children"],
                        "additionalProperties": False,
                    }
                },
            },
            [
                '{"node":{"value":"one","children":[]}}',
                '{"node":{"value":"one","children":[{"value":"two","children":[]}]}}',
            ],
            ['{"node":{"value":"one","children":[{"value":null,"children":[]}]}}'],
            True,
        ),
        ({"type": "integer", "const": 3, "enum": [3, 7], "minimum": 2, "maximum": 4}, ["3"], ["2", "7"], False),
        ({"type": ["integer", "null"], "enum": [3, None], "minimum": 2}, ["3", "null"], ["1", "4"], False),
        (
            {
                "const": 10,
                "anyOf": [{"$ref": "#/$defs/Bound"}],
                "$defs": {"Bound": {"type": "integer", "minimum": 10}},
            },
            ["10"],
            ["9"],
            False,
        ),
        ({"type": "string", "const": "café", "minLength": 4, "maxLength": 4}, ['"café"'], ['"cafe"'], False),
        ({"type": "number", "minimum": -1.25, "maximum": 2.5}, ["-1.25", "2.5"], ["-1.3", "2.6"], False),
        (
            {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
                "additionalProperties": {"type": "string"},
            },
            ['{"value":1,"extra":"yes"}'],
            ['{"value":1,"extra":false}'],
            False,
        ),
        (
            {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "nested": {
                        "type": "object",
                        "properties": {"title": {"type": "integer"}, "description": {"type": "string"}},
                        "required": ["title", "description"],
                        "additionalProperties": False,
                    },
                },
                "required": ["title", "description", "nested"],
                "additionalProperties": False,
            },
            ['{"title":"book","description":"summary","nested":{"title":1,"description":"chapter"}}'],
            ['{"title":"book","description":"summary","nested":{"title":"1","description":"chapter"}}'],
            True,
        ),
        (
            {
                "type": "object",
                "properties": {'quote"slash\\': {"type": "integer"}},
                "required": ['quote"slash\\'],
                "additionalProperties": False,
            },
            [json.dumps({'quote"slash\\': 1})],
            [json.dumps({'quote"slash\\': "1"})],
            True,
        ),
        (
            {
                "type": "object",
                "properties": {
                    name: {
                        "type": "object",
                        "description": name,
                        "properties": {"title": {"type": "string", "description": name}},
                        "required": ["title"],
                        "additionalProperties": False,
                    }
                    for name in ("a", "b")
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
            ['{"a":{"title":"one"},"b":{"title":"two"}}'],
            ['{"a":{"title":"one"},"b":{"title":2}}'],
            True,
        ),
        ({}, ["null", '"free"', "[1,true]", '{"free":1}'], [], False),
    ],
)
def test_validated_schema_grammar_enforces_completed_instances(
    compiler: xgr.GrammarCompiler,
    schema: dict[str, JSON],
    accepted: Sequence[str],
    rejected: Sequence[str],
    *,
    strict: bool,
) -> None:
    validate_json_schema(schema, strict=strict)
    compiled = compiler.compile_json_schema(schema, strict_mode=False, any_order=False)
    validator = Draft202012Validator(schema)
    for instances, expected in ((accepted, True), (rejected, False)):
        for text in instances:
            assert validator.is_valid(json.loads(text)) is expected
            matcher = xgr.GrammarMatcher(compiled)
            complete = matcher.accept_string(text) and matcher.accept_token(256)
            assert complete is expected


@pytest.mark.parametrize("field_name", ["title", "description"])
@pytest.mark.parametrize("literal_kind", [None, "const", "enum"])
def test_rejects_annotation_named_property_cache_collisions(
    compiler: xgr.GrammarCompiler,
    field_name: str,
    literal_kind: str | None,
) -> None:
    schema: dict[str, JSON] = {
        "type": "object",
        "properties": {
            name: {
                "type": "object",
                "properties": {field_name: {"type": field_type}},
                "required": [field_name],
                "additionalProperties": False,
            }
            for name, field_type in (("a", "string"), ("b", "integer"))
        },
        "required": ["a", "b"],
        "additionalProperties": False,
    }
    if literal_kind is not None:
        properties = schema["properties"]
        assert isinstance(properties, dict)
        for name, value in (("a", "one"), ("b", 2)):
            node = properties[name]
            assert isinstance(node, dict)
            literal: JSON = {field_name: value}
            if literal_kind == "enum":
                literal = [literal]
            node[literal_kind] = literal
    invalid_value = "two"
    if literal_kind is not None:
        invalid_value = "one"
    invalid = json.dumps({"a": {field_name: "one"}, "b": {field_name: invalid_value}}, separators=(",", ":"))
    assert not Draft202012Validator(schema).is_valid(json.loads(invalid))
    compiled = compiler.compile_json_schema(schema, strict_mode=False, any_order=False)
    matcher = xgr.GrammarMatcher(compiled)
    assert matcher.accept_string(invalid) and matcher.accept_token(256)
    with pytest.raises(ValueError, match="parser cache"):
        validate_json_schema(schema, strict=True)


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "array", "uniqueItems": True},
        {"type": "array", "contains": {"const": 1}},
        {"type": "object", "not": {"required": ["x"]}},
        {"type": "integer", "multipleOf": 2},
        {"type": "string", "pattern": "^a$", "minLength": 3},
        {"type": "string", "format": "date"},
        {"type": "integer", "enum": [1, 3], "minimum": 2},
        {"type": "string", "const": "a", "minLength": 2},
        {"type": "integer", "const": "1"},
        {"type": "integer", "const": 3, "enum": [4]},
        {"type": "integer", "enum": [1], "anyOf": [{"type": "null"}]},
        {"const": 9, "anyOf": [{"$ref": "#/$defs/Bound"}], "$defs": {"Bound": {"type": "integer", "minimum": 10}}},
        {"const": "a", "anyOf": [{"$ref": "#"}]},
        {"type": "integer", "const": 2**64 + 1},
        {"type": "integer", "enum": [2**64 + 1]},
        {"type": "object", "const": {"value": 2**64 + 1}},
        {"type": "array", "const": [2**64 + 1]},
        {"type": "object", "enum": [{"value": [2**64 + 1]}]},
        {"$ref": "#/$defs/N", "minimum": 10, "$defs": {"N": {"type": "integer"}}},
        {"anyOf": [{"type": "integer"}, {"type": "null"}], "minimum": 10},
        {"$ref": "https://example.invalid/schema"},
        {"$ref": "#anchor"},
        {"$ref": "#/$defs/space%20name", "$defs": {"space name": {"type": "integer"}}},
        {"$ref": "#/$defs/a~1b", "$defs": {"a/b": {"type": "integer"}}},
        {"$ref": "#/$defs/Missing", "$defs": {}},
        {"minimum": 10},
        {"minLength": 2},
        {"type": "object", "properties": {}, "required": ["x"], "additionalProperties": True},
        {"type": "number", "minimum": 2**53 + 1},
        {"type": "number", "minimum": 10**400},
        {"type": "integer", "minimum": 0.5},
        {"type": "integer", "minimum": 2**63},
        {"type": "string", "maxLength": 2**31},
        {"type": "array", "maxItems": 2**31},
    ],
)
def test_rejects_valid_schemas_the_guided_backend_would_weaken(schema: dict[str, JSON]) -> None:
    Draft202012Validator.check_schema(schema)
    with pytest.raises(ValueError):
        validate_json_schema(schema)


@pytest.mark.parametrize(
    "schema",
    [
        {},
        {"type": "array", "items": {"type": "integer"}},
        {"anyOf": [{"type": "object"}, {"type": "null"}]},
        {"type": "object", "properties": {"value": {"type": "integer"}}, "additionalProperties": False},
        {"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]},
        {
            "type": "object",
            "properties": {
                "child": {"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]}
            },
            "required": ["child"],
            "additionalProperties": False,
        },
    ],
)
def test_strict_contract_requires_closed_objects_and_all_fields(schema: dict[str, JSON]) -> None:
    with pytest.raises(ValueError):
        validate_json_schema(schema, strict=True)


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "string", "minimum": float("inf")},
        {"type": "number", "const": float("nan")},
        {"examples": [{"value": float("inf")}]},
        {"type": "invalid"},
        {"type": "array", "items": 3},
    ],
)
def test_syntax_boundary_rejects_invalid_or_nonfinite_json(schema: dict[str, JSON]) -> None:
    with pytest.raises(ValueError):
        check_json_schema_syntax(schema)
