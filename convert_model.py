from pathlib import Path

import jax.numpy as jnp
from safetensors.flax import save_file
from typer import run

from fartsovka.importers.executorch.importer import ExecutorchModel
from fartsovka.importers.executorch.importer import import_model as import_et

ACTIVATION_EXPORT_TYPE = jnp.bfloat16


def main(output_dir: Path) -> None:
    model = import_et(
        ExecutorchModel.LLAMA32_1B_INSTRUCT_QLORA,
        activation_precision=ACTIVATION_EXPORT_TYPE,
    )
    weights = model.export_weights()
    save_file(weights, output_dir / "fs_model.safetensors")


if __name__ == "__main__":
    run(main)
