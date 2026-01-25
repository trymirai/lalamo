import json
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm
import traceback as tb

from lalamo.model_import import REPO_TO_MODEL, import_model
from lalamo.speculator.estimator import estimate_memory_from_batchsize

MODEL_LIST = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "LiquidAI/LFM2-700M",
]

BATCH_SIZES = [1, 4, 16, 64]

HEADERS = [
    "model",
    "batch_size",
    "peak",
    "estimate",
]


def _render_table(rows: list[list[str]]) -> str:
    widths = [len(h) for h in HEADERS]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(HEADERS))
    sep_line = "-+-".join("-" * widths[i] for i in range(len(HEADERS)))
    body_lines = [" | ".join(row[i].ljust(widths[i]) for i in range(len(HEADERS))) for row in rows]
    return "\n".join([header_line, sep_line, *body_lines])


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "n/a"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} PiB"


def main() -> None:
    script_path = Path(__file__).with_name("measure_memory_for_batch_size.py")
    rows: list[list[str]] = []
    models = {m: import_model(REPO_TO_MODEL[m]).model for m in MODEL_LIST}
    work_items = [(m, b) for m in MODEL_LIST for b in BATCH_SIZES]

    # Sanity check: estimating a huge batch shouldn't allocate real device memory.
    estimate_bytes = estimate_memory_from_batchsize(
        models[MODEL_LIST[0]],
        max_input_length=128,
        max_output_length=32,
        num_logits_per_token=8,
        batch_size=1_000,
    )
    print(f"large_batch_estimate_bytes={estimate_bytes}")

    for model_repo, batch_size in tqdm(work_items):
        try:
            result = subprocess.run(
                [sys.executable, str(script_path), model_repo, str(batch_size)],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception as exc:
            print(f"script failed {exc.stdout or ''}, {exc.stderr or ''}; {tb.format_exc()}")
            raise
        data = json.loads(result.stdout)
        estimate_bytes = estimate_memory_from_batchsize(
            models[model_repo],
            max_input_length=128,
            max_output_length=32,
            num_logits_per_token=8,
            batch_size=batch_size,
        )
        rows.append(
            [
                model_repo,
                str(batch_size),
                data.get("after_peak_human", "n/a"),
                _format_bytes(estimate_bytes),
            ],
        )
        print(_render_table(rows))
        print()


if __name__ == "__main__":
    main()
