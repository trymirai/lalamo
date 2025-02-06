To convert model run
```bash
uv run convert_model.py
```

Usage:
```
╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    model_repo      TEXT  [default: None] [required]                                                                                                                       │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --precision             TEXT     [default: None]                                                                                                                            │
│ --output-dir            PATH     [default: None]                                                                                                                            │
│ --context-length        INTEGER  [default: 8192]                                                                                                                            │
│ --help                           Show this message and exit.                                                                                                                │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

Currently supported models:
```
meta-llama/Llama-3.2-1B-Instruct
meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8
google/gemma-2-2b-it
Qwen/Qwen2.5-1.5B-Instruct
deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

To add support for a new model write the corresponding `ModelSpec` in `fartsovka.model_import.model_import.py`

Example:
```python
ModelSpec(
    name="Gemma-2-2B-Instruct",
    repo="google/gemma-2-2b-it",
    config_type=HFGemma2Config,
    config_file_name="config.json",
    weights_file_names=(
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ),
    weights_type=WeightsType.SAFETENSORS,
)
```
