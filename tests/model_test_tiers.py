from __future__ import annotations

import re
import tomllib
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

from tests.helpers import unsi

if TYPE_CHECKING:
    from lalamo.model_import.model_specs.common import ModelSpec

PARAM_UNITS = ("", "K", "M", "B")


class ModelTier(IntEnum):
    CANONICAL = 0
    CORE = 1
    STANDARD = 2
    EXTRA = 3


class ModelSize(IntEnum):
    SMALL = 0  # < 10B params
    LARGE = 1  # >= 10B params


class ModelTierListsConfig(TypedDict):
    canonical: list[str]
    core: list[str]
    standard: list[str]
    extra: list[str]


class CoherenceModelTestTiersConfig(TypedDict):
    tts_repos: list[str]


class ModelTestTiersConfig(TypedDict):
    tiers: ModelTierListsConfig
    coherence: CoherenceModelTestTiersConfig


def _load_model_test_tiers_config() -> ModelTestTiersConfig:
    with (Path(__file__).parent / "model_test_tiers.toml").open("rb") as config_file:
        return cast("ModelTestTiersConfig", tomllib.load(config_file))


MODEL_TEST_TIERS_CONFIG = _load_model_test_tiers_config()


def num_params(size: str) -> int | None:
    normalized = re.sub(r"([0-9.]+)\s*([A-Z])", r"\1 \2", size.strip())
    try:
        return unsi(normalized, base=1000, units=PARAM_UNITS)
    except (ValueError, IndexError):
        return None


def model_size(spec: ModelSpec) -> ModelSize:
    params = num_params(spec.size)
    if params is None:
        return ModelSize.LARGE
    if params < 10_000_000_000:
        return ModelSize.SMALL
    return ModelSize.LARGE


def _tier_repos(tier: ModelTier) -> tuple[str, ...]:
    return tuple(MODEL_TEST_TIERS_CONFIG["tiers"][tier.name.lower()])


MODEL_TIERS: tuple[tuple[str, ModelTier], ...] = tuple(
    (repo, tier) for tier in ModelTier for repo in _tier_repos(tier)
)

TIER_BY_REPO: dict[str, ModelTier] = dict(MODEL_TIERS)
CI_CORE_LM_REPOS: tuple[str, ...] = _tier_repos(ModelTier.CANONICAL) + _tier_repos(ModelTier.CORE)
COHERENCE_TTS_REPOS: tuple[str, ...] = tuple(MODEL_TEST_TIERS_CONFIG["coherence"]["tts_repos"])
