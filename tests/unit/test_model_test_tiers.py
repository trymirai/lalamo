from tests.conftest import ALL_MODEL_SPECS
from tests.model_test_tiers import TIER_BY_REPO


def test_tier_by_repo_matches_registry() -> None:
    registry_repos = {spec.repo for spec in ALL_MODEL_SPECS}
    tier_repos = set(TIER_BY_REPO)
    assert registry_repos == tier_repos
