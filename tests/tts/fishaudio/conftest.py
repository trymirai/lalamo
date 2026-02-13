from pathlib import Path

import huggingface_hub
from huggingface_hub import HfApi
from pytest import fixture


@fixture
def fish_audio_local_model_path() -> Path:
    fish_audio_repo_id = "fishaudio/s1-mini"

    repos = huggingface_hub.scan_cache_dir().repos
    fish_audio_candidate_repos = list(filter(lambda repo: repo.repo_id == fish_audio_repo_id, repos))
    if not fish_audio_candidate_repos:
        return Path(huggingface_hub.snapshot_download(fish_audio_repo_id))

    api = HfApi()
    cache_info = api.model_info(fish_audio_repo_id)

    for repo in fish_audio_candidate_repos:
        path = repo.repo_path / "snapshots" / str(cache_info.sha)

        if path.exists():
            return path

    raise RuntimeError("The cached fishaudio repository is in an invalid state")
