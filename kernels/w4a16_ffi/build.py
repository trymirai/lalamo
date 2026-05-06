from __future__ import annotations

import subprocess
from pathlib import Path
from shutil import which

import jax.ffi


def build() -> Path:
    root = Path(__file__).resolve().parent
    output = root / "liblalamo_w4a16_ffi.so"
    nvcc = which("nvcc") or "/usr/local/cuda/bin/nvcc"
    subprocess.run(
        [
            nvcc,
            "-O3",
            "-std=c++17",
            "-arch=sm_90",
            "--compiler-options=-fPIC",
            "-shared",
            f"-I{jax.ffi.include_dir()}",
            str(root / "w4a16_ffi.cu"),
            "-o",
            str(output),
        ],
        check=True,
    )
    return output


if __name__ == "__main__":
    print(build())
