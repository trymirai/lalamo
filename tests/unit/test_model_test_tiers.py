from lalamo.model_registry import ModelRegistry
from tests.model_test_tiers import TIER_BY_REPO


def _registry_repos() -> frozenset[str]:
    registry = ModelRegistry.build(allow_third_party_plugins=False)
    return frozenset(spec.repo for spec in registry.models)


def test_every_registry_model_has_a_tier() -> None:
    missing = _registry_repos() - TIER_BY_REPO.keys()
    assert not missing, (
        f"Models in the registry without a tier assignment: {sorted(missing)}\n"
        "Add them to tests/model_test_tiers.py."
    )


def test_no_stale_tier_entries() -> None:
    stale = TIER_BY_REPO.keys() - _registry_repos()
    assert not stale, (
        f"Tier entries for models not in the registry: {sorted(stale)}\n"
        "Remove them from tests/model_test_tiers.py."
    )
