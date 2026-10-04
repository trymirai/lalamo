import json
from collections.abc import Iterator
from copy import deepcopy
from typing import cast

from jsonschema import Draft202012Validator, SchemaError

from lalamo.utils.json import JSON

__all__ = ["check_json_schema_syntax", "iter_json_schema_nodes", "validate_json_schema", "xgrammar_schema"]


def check_json_schema_syntax(schema: dict[str, JSON]) -> None:
    try:
        json.dumps(schema, allow_nan=False)
        Draft202012Validator.check_schema(schema)
    except (TypeError, ValueError, SchemaError) as error:
        raise ValueError(f"Invalid JSON schema: {error}") from error


def iter_json_schema_nodes(node: dict[str, JSON] | bool) -> Iterator[dict[str, JSON] | bool]:
    yield node
    if isinstance(node, bool):
        return
    for key in ("properties", "$defs"):
        children = node.get(key)
        if isinstance(children, dict):
            for child in children.values():
                yield from iter_json_schema_nodes(cast("dict[str, JSON] | bool", child))
    for key in ("items", "additionalProperties"):
        child = node.get(key)
        if isinstance(child, dict) or (key == "items" and isinstance(child, bool)):
            yield from iter_json_schema_nodes(child)
    alternatives = node.get("anyOf")
    if isinstance(alternatives, list):
        for child in alternatives:
            yield from iter_json_schema_nodes(cast("dict[str, JSON] | bool", child))


def validate_json_schema(schema: dict[str, JSON], *, strict: bool = False) -> None:
    check_json_schema_syntax(schema)
    annotations = {
        "title",
        "description",
        "default",
        "examples",
        "deprecated",
        "readOnly",
        "writeOnly",
        "$comment",
        "$schema",
    }
    keywords = annotations | {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "enum",
        "const",
        "anyOf",
        "$ref",
        "$defs",
    }

    if strict and (schema.get("type") != "object" or "anyOf" in schema):
        raise ValueError("Strict JSON schemas must have an object root without anyOf.")
    schema_nodes = tuple(iter_json_schema_nodes(schema))
    schema_node_ids = {id(node) for node in schema_nodes}

    def cache_keys(value: JSON) -> tuple[str, str]:
        if isinstance(value, dict):
            entries = [(key, cache_keys(child)) for key, child in sorted(value.items())]
            backend = ",".join(f'"{key}":{keys[0]}' for key, keys in entries if key not in annotations)
            meaning = ",".join(
                f"{json.dumps(key)}:{keys[1]}"
                for key, keys in entries
                if id(value) not in schema_node_ids or key not in annotations
            )
            return "{" + backend + "}", "{" + meaning + "}"
        if isinstance(value, list):
            children = [cache_keys(child) for child in value]
            return (
                "[" + ",".join(keys[0] for keys in children) + "]",
                "[" + ",".join(keys[1] for keys in children) + "]",
            )
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return encoded, encoded

    # XGrammar 0.2.8 drops annotation-named keys recursively, even in properties,
    # and concatenates keys without JSON escaping. Reproduce that cache key only here.
    cache_meanings: dict[str, str] = {}
    for node in schema_nodes:
        cache_key, meaning = cache_keys(node)
        if cache_key in cache_meanings and cache_meanings[cache_key] != meaning:
            raise ValueError("JSON schema assertions collide in the guided backend parser cache.")
        cache_meanings[cache_key] = meaning
    # Validate all references before instance validation, which can otherwise fetch remote schemas.
    for node in schema_nodes:
        if isinstance(node, bool):
            if not node:
                raise ValueError("JSON schema false cannot produce a value.")
            continue
        if unsupported := node.keys() - keywords:
            raise ValueError(f"Unsupported JSON schema keywords: {', '.join(sorted(unsupported))}.")
        if "$ref" in node:
            reference = cast("str", node["$ref"])
            if reference != "#":
                prefix = "#/$defs/"
                name = reference.removeprefix(prefix)
                definitions = schema.get("$defs")
                if (
                    not reference.startswith(prefix)
                    or not name
                    or any(character in name for character in "/~%")
                    or not isinstance(definitions, dict)
                    or name not in definitions
                ):
                    raise ValueError("JSON schema references must be # or #/$defs/<unescaped-name>.")
            if node.keys() - (annotations | {"$ref", "$defs"}):
                raise ValueError("JSON schema $ref assertion siblings are not supported.")
        elif "const" not in node and "enum" not in node and "anyOf" in node:
            if node.keys() - (annotations | {"anyOf", "$defs"}):
                raise ValueError("JSON schema anyOf assertion siblings are not supported.")
        if any(key in node for key in ("$ref", "anyOf", "const", "enum")):
            continue
        types = node.get("type", [])
        if isinstance(types, str):
            types = [types]
        types = set(cast("list[str]", types))
        if not types:
            if "properties" in node or "additionalProperties" in node:
                types = {"object"}
            elif "items" in node:
                types = {"array"}
            elif node.keys() - (annotations | {"$defs"}):
                raise ValueError("JSON schema assertions require a declared type.")
        properties = node.get("properties", {})
        required = node.get("required", [])
        if isinstance(properties, dict) and isinstance(required, list):
            if set(cast("list[str]", required)) - properties.keys():
                raise ValueError("Required JSON schema keys must be declared in properties.")
            if (
                strict
                and "object" in types
                and (
                    node.get("additionalProperties") is not False
                    or set(cast("list[str]", required)) != properties.keys()
                )
            ):
                raise ValueError("Strict JSON objects require every property and additionalProperties:false.")
        for key in ("minItems", "maxItems", "minLength", "maxLength"):
            if key in node and not 0 <= cast("int", node[key]) <= 2**31 - 1:
                raise ValueError(f"JSON schema {key} must fit a nonnegative int32.")
        for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
            if key not in node:
                continue
            bound = cast("int | float", node[key])
            if "integer" in types and (bound != int(bound) or not -(2**63) <= bound < 2**63):
                raise ValueError("Integer JSON schema bounds must be whole signed int64 values.")
            if "number" in types and isinstance(bound, int):
                try:
                    approximate = float(bound)
                except OverflowError as error:
                    raise ValueError("Number JSON schema bounds must fit finite float64.") from error
                if approximate != bound:
                    raise ValueError("Number JSON schema integer bounds must be exactly representable as float64.")

    def check_literal_integers(value: JSON) -> None:
        if isinstance(value, dict):
            for child in value.values():
                check_literal_integers(child)
        elif isinstance(value, list):
            for child in value:
                check_literal_integers(child)
        elif type(value) is int and not -(2**63) <= value < 2**63:
            raise ValueError("JSON schema enum and const integers must fit signed int64.")

    validator = Draft202012Validator(schema)
    for node in schema_nodes:
        if isinstance(node, bool):
            continue
        if "const" in node:
            values = [node["const"]]
        elif "enum" in node:
            values = cast("list[JSON]", node["enum"])
        else:
            continue
        for value in values:
            check_literal_integers(value)
            try:
                valid = validator.evolve(schema=node).is_valid(value)
            except RecursionError as error:
                raise ValueError("Recursive JSON schema literal assertions cannot be validated.") from error
            if not valid:
                raise ValueError("JSON schema enum or const values violate their sibling assertions.")


def xgrammar_schema(schema: dict[str, JSON]) -> dict[str, JSON]:
    # XGrammar 0.2.8 derives rule names from UTF-8 $ref bytes and does not decode JSON pointers.
    # Only internal references are aliased; instance keys and schema values remain untouched.
    translated = deepcopy(schema)
    definitions = cast("dict[str, JSON]", translated.get("$defs", {}))
    names = {name: f"definition_{index}" for index, name in enumerate(definitions)}
    for node in iter_json_schema_nodes(translated):
        if isinstance(node, dict):
            reference = node.get("$ref")
            if isinstance(reference, str) and reference.startswith("#/$defs/"):
                node["$ref"] = "#/$defs/" + names[reference.removeprefix("#/$defs/")]
    translated_definitions: dict[str, JSON] = {names[name]: node for name, node in definitions.items()}
    reference = translated.get("$ref")
    if isinstance(reference, str) and reference.startswith("#/properties/"):
        property_name = reference.removeprefix("#/properties/").replace("~1", "/").replace("~0", "~")
        properties = cast("dict[str, JSON]", translated["properties"])
        root_name = f"definition_{len(translated_definitions)}"
        if any(isinstance(node, dict) and node.get("$ref") == "#" for node in iter_json_schema_nodes(translated)):
            # A native property projection must keep # pointing at the original argument object.
            root_schema = {key: value for key, value in translated.items() if key not in ("$ref", "$defs")}
            for node in iter_json_schema_nodes(translated):
                if isinstance(node, dict) and node.get("$ref") == "#":
                    node["$ref"] = "#/$defs/" + root_name
            translated_definitions[root_name] = root_schema
        name = f"definition_{len(translated_definitions)}"
        translated_definitions[name] = properties[property_name]
        translated["$ref"] = "#/$defs/" + name
    if translated_definitions or "$defs" in translated:
        translated["$defs"] = translated_definitions
    return translated
