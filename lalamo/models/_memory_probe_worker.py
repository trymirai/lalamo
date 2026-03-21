"""Subprocess entry point for GPU memory probe compilation.

Invoked as: python -m lalamo.models._memory_probe_worker <model_path> <batch_size> <seq_len> <max_output_length> [num_logits]

Outputs a single JSON object to stdout: {"memory_bytes": <int>}
"""

import json
import sys

from lalamo.models import LanguageModelConfig
from lalamo.models.common import InferenceConfig


def main() -> None:
    model_path = sys.argv[1]
    batch_size = int(sys.argv[2])
    seq_len = int(sys.argv[3])
    max_output_length = int(sys.argv[4])
    num_logits = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] != "none" else None

    model = LanguageModelConfig.load_model(model_path)
    config = InferenceConfig(
        max_output_length=max_output_length,
        padded_length=seq_len,
        num_top_logits_to_return=num_logits,
        batch_size=batch_size,
    )
    memory_bytes = model.estimate_memory_consumption(inference_config=config)
    json.dump({"memory_bytes": memory_bytes}, sys.stdout)


if __name__ == "__main__":
    main()
