from lalamo.model_import.model_specs.common import ModelType
from tests.conftest import ALL_MODEL_SPECS
from tests.model_test_tiers import CI_CORE_LM_REPOS, TIER_BY_REPO, ModelTier


def test_tier_by_repo_matches_registry() -> None:
    registry_repos = {spec.repo for spec in ALL_MODEL_SPECS}
    tier_repos = set(TIER_BY_REPO)
    assert registry_repos == tier_repos


def test_ci_core_lm_matrix_matches_model_tiers() -> None:
    core_lm_repos = {
        spec.repo
        for spec in ALL_MODEL_SPECS
        if spec.model_type == ModelType.LANGUAGE_MODEL and TIER_BY_REPO[spec.repo] <= ModelTier.CORE
    }
    assert core_lm_repos == set(CI_CORE_LM_REPOS)
    assert len(CI_CORE_LM_REPOS) == len(core_lm_repos)
