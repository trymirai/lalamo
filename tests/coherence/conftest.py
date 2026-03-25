import logging
import subprocess
from collections.abc import Generator
from pathlib import Path

import pytest

log = logging.getLogger(__name__)

HF_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"


def _log_resource_usage(label: str) -> None:
    log.info("=== %s ===", label)
    log.info("disk usage:\n%s", subprocess.check_output(["df", "-h", "."], text=True))

    hf_size = sum(f.stat().st_size for f in HF_CACHE_DIR.rglob("*") if f.is_file()) if HF_CACHE_DIR.exists() else 0
    log.info("HF cache size: %.2f GB", hf_size / 1e9)

    tmp_size = sum(f.stat().st_size for f in Path("/tmp").rglob("*") if f.is_file())
    log.info("/tmp size: %.2f GB", tmp_size / 1e9)

    try:
        log.info("memory:\n%s", subprocess.check_output(["free", "-m"], text=True))
    except FileNotFoundError:
        log.info("memory:\n%s", subprocess.check_output(["vm_stat"], text=True))


@pytest.fixture(autouse=True)
def _log_resources(request: pytest.FixtureRequest) -> Generator[None]:
    _log_resource_usage(f"BEFORE {request.node.nodeid}")
    yield
    _log_resource_usage(f"AFTER {request.node.nodeid}")
