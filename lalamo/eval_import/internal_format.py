from dataclasses import dataclass


@dataclass(frozen=True)
class InternalEvalRecord:
    """Universal internal format for eval dataset records."""

    id: str
    question: str
    answer: str
    options: list[str] | None = None
    answer_index: int | None = None
    reasoning: str | None = None
    category: str | None = None
    metadata: dict[str, str] | None = None


@dataclass(frozen=True)
class DatasetMetadata:
    """Metadata about a converted eval dataset."""

    lalamo_version: str
    name: str
    repo: str
    splits: tuple[str, ...]
    schema_version: str
    total_examples: dict[str, int]
