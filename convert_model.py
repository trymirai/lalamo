import json
from pathlib import Path

from safetensors.flax import save_file
from typer import run

from fartsovka.common import DType
from fartsovka.model_import import REPO_TO_MODEL, import_model
from fartsovka.modules import DecoderConfig, config_converter


def main(
    model_repo: str,
    precision: str | None = None,
    output_dir: Path | None = None,
    context_length: int = 8192,
) -> None:
    if precision is not None:
        precision_dtype = config_converter.structure(precision, DType)  # type: ignore
    else:
        precision_dtype = None

    model_spec = REPO_TO_MODEL.get(model_repo)
    if model_spec is None:
        raise ValueError(
            f"Unsupported model repo: {model_repo}.\nCurrently supported repos: {list(REPO_TO_MODEL.keys())}.",
        )

    model = import_model(model_spec, precision=precision_dtype, context_length=context_length)
    config_json = config_converter.unstructure(model.config, DecoderConfig)
    weights = dict(model.export_weights())

    if output_dir is None:
        output_dir = Path(model_spec.name)
    output_dir.mkdir(exist_ok=True)
    save_file(weights, output_dir / "model.safetensors")
    with open(output_dir / "config.json", "w") as file:
        json.dump(config_json, file, indent=4)


if __name__ == "__main__":
    run(main)
