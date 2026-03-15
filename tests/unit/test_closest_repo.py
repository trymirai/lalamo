import pytest

from lalamo.main import _closest_repo

REPO_IDS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-0.6B-MLX-4bit",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-4B-AWQ",
    "Qwen/Qwen3-4B-MLX-4bit",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-8B-AWQ",
    "Qwen/Qwen3-8B-MLX-4bit",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
    "amd/PARD-Qwen3-0.6B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
]


@pytest.mark.parametrize(
    "query",
    [
        "qwen3",
        "qwen-4bit",
        "mlx-community/Qwen3.5-0.8B-MLX-4bit",
        "Qwen3-4B",
        "Qwen2.5-0.6B-Instruct",
        "qwen3-0.6b-awq",
        "Qwen/Qwen3-0.5B",
        "Qwen/PARD-Qwen3-0.6B",
    ],
)
def test_ambiguous_query_returns_none(query: str) -> None:
    assert _closest_repo(query, REPO_IDS) is None


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-0.6B"),
        ("qwen3-0.6B", "Qwen/Qwen3-0.6B"),
        ("Qwen3-0.6B-MLX-4bit", "Qwen/Qwen3-0.6B-MLX-4bit"),
        ("qwen3-0.6B-4bit", "Qwen/Qwen3-0.6B-MLX-4bit"),
    ],
)
def test_unambiguous_query_matches(query: str, expected: str) -> None:
    assert _closest_repo(query, REPO_IDS) == expected


def test_empty_repo_list() -> None:
    assert _closest_repo("anything", []) is None


def test_below_min_score() -> None:
    assert _closest_repo("completely-unrelated-model", REPO_IDS) is None
