from pathlib import Path

from lalamo.evals.datasets.specs import REPO_TO_EVAL
from lalamo.evals.inference import LalamoInferenceEngine
from lalamo.evals.inference import run_inference as run_eval_inference
from lalamo.evals.inference.callbacks import BaseRunInferenceCallbacks


def infer_command_handler(
    eval_name: str,
    model_path: Path,
    dataset_dir: Path,
    output_dir: Path,
    callbacks: BaseRunInferenceCallbacks,
    engine: str = "lalamo",
    num_few_shot: int = 5,
    max_examples: int | None = None,
    category: str | None = None,
    batch_size: int | None = None,
    vram_gb: float | None = None,
    max_output_length: int = 2048,
) -> Path:
    if eval_name not in REPO_TO_EVAL:
        available = ", ".join(REPO_TO_EVAL.keys())
        raise ValueError(f"Unknown eval: {eval_name}. Available evals: {available}")

    eval_spec = REPO_TO_EVAL[eval_name]

    if engine == "lalamo":
        inference_engine = LalamoInferenceEngine(
            model_path=model_path,
            max_vram=int(vram_gb * 1024**3) if vram_gb else None,
            batch_size=batch_size,
            max_output_length=max_output_length,
        )
    else:
        raise ValueError(f"Unsupported engine: {engine}. Supported: lalamo")

    eval_adapter = eval_spec.handler_type()

    callbacks.started()

    predictions_path = run_eval_inference(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        inference_engine=inference_engine,
        eval_adapter=eval_adapter,
        num_few_shot=num_few_shot,
        max_examples=max_examples,
        category=category,
        callbacks=callbacks,
    )

    return predictions_path
