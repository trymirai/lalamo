import pytest

from lalamo.utils.lazy_collections import LazyDict, MapCollection, MapDictValues, MapIterable, MapSequence


def test_lazy_dict_reads_values_only_when_keys_are_accessed() -> None:
    accessed_keys: list[str] = []
    lazy_dict = LazyDict(
        stored_keys={"a", "b"},
        getter=lambda key: accessed_keys.append(key) or key.upper(),
    )

    assert len(lazy_dict) == 2
    assert set(lazy_dict) == {"a", "b"}
    assert accessed_keys == []

    assert lazy_dict["a"] == "A"
    assert accessed_keys == ["a"]


def test_lazy_dict_rejects_missing_keys_without_calling_getter() -> None:
    accessed_keys: list[str] = []
    lazy_dict = LazyDict(
        stored_keys={"a"},
        getter=lambda key: accessed_keys.append(key) or key.upper(),
    )

    with pytest.raises(KeyError, match="'missing'"):
        _ = lazy_dict["missing"]

    assert accessed_keys == []


def test_map_iterable_maps_values_during_iteration() -> None:
    mapped_values = MapIterable(lambda value: value * 2, [1, 2, 3])

    assert list(mapped_values) == [2, 4, 6]


def test_map_collection_membership_checks_mapped_values() -> None:
    mapped_values = MapCollection(lambda value: value * 2, [1, 2, 3])

    assert len(mapped_values) == 3
    assert 4 in mapped_values
    assert 2 in mapped_values
    assert 3 not in mapped_values


def test_map_sequence_maps_indexing_and_slicing() -> None:
    mapped_values = MapSequence(lambda value: value * value, [2, 3, 4])

    assert mapped_values[1] == 9
    assert mapped_values[:2] == [4, 9]
    assert list(mapped_values) == [4, 9, 16]


def test_map_dict_values_preserves_keys_and_maps_values_lazily() -> None:
    mapped_keys: list[str] = []
    mapped_values = MapDictValues(
        value_map=lambda value: mapped_keys.append(value) or value.upper(),
        collection={"a": "alpha", "b": "beta"},
    )

    assert len(mapped_values) == 2
    assert list(mapped_values) == ["a", "b"]
    assert list(mapped_values.keys()) == ["a", "b"]
    assert mapped_keys == []

    assert mapped_values["b"] == "BETA"
    assert mapped_keys == ["beta"]


def test_map_dict_values_values_view_maps_values() -> None:
    mapped_values = MapDictValues(
        value_map=lambda value: value.upper(),
        collection={"a": "alpha", "b": "beta"},
    )

    assert set(mapped_values.values()) == {"ALPHA", "BETA"}
    assert "ALPHA" in mapped_values.values()
    assert "alpha" not in mapped_values.values()
