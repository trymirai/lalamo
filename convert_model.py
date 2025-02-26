import json
import shutil
from enum import Enum
from pathlib import Path
from typing import Annotated

from safetensors.flax import save_file
from typer import Argument, Option, run

from fartsovka.common import DType
from fartsovka.language_model import LanguageModelConfig
from fartsovka.model_import import REPO_TO_MODEL, import_language_model
from fartsovka.modules import config_converter


class Precision(Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"


def main(
    model_repo: Annotated[
        str,
        Argument(
            help="Huggingface model repo. Example: 'meta-llama/Llama-3.2-1B-Instruct'",
            show_default=False,
            metavar="MODEL_REPO",
        ),
    ],
    precision: Annotated[
        Precision | None,
        Option(
            help="Precision to use for activations and non-quantized weights.",
            show_default="Native precision of the model",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        Option(
            help="Directory to save the converted model to.",
            show_default="Creates a new directory with the model name",
        ),
    ] = None,
    context_length: Annotated[
        int,
        Option(
            help="Maximum supported context length. Used to precompute positional embeddings.",
        ),
    ] = 8192,
) -> None:
    if precision is not None:
        precision_dtype = config_converter.structure(precision.value, DType)  # type: ignore
    else:
        precision_dtype = None

    model_spec = REPO_TO_MODEL.get(model_repo)
    if model_spec is None:
        raise ValueError(
            f"Unsupported model repo: {model_repo}.\nCurrently supported repos: {list(REPO_TO_MODEL.keys())}.",
        )

    # Import the language model (decoder + tokenizer)
    lm = import_language_model(
        model_spec, 
        precision=precision_dtype, 
        context_length=context_length
    )
    
    # Export model config
    lm_config_json = config_converter.unstructure(lm.config, LanguageModelConfig)
    weights = dict(lm.export_weights())
    
    # Copy the tokenizer file 
    tokenizer_path = Path(lm.tokenizer.config.tokenizer_path)

    if output_dir is None:
        output_dir = Path(model_spec.name)
    output_dir.mkdir(exist_ok=True)
    
    # Save the model weights
    save_file(weights, output_dir / "model.safetensors")
    
    # Save the language model config
    with open(output_dir / "config.json", "w") as file:
        json.dump(lm_config_json, file, indent=4)
    
    # Copy the tokenizer file
    shutil.copy2(tokenizer_path, output_dir / "tokenizer.json")


if __name__ == "__main__":
    run(main)
