# Lalamo

A set of tools for adapting Large Language Models to on-device inference using the Uzu inference engine.

To convert a model run

```bash
uv run lalamo convert MODEL_REPO
```

For more options see `uv run lalamo convert --help`

To get a list of supported models run `uv run lalamo list-models`

To add support for a new model write the corresponding `ModelSpec` in `lalamo.model_import.model_import.py`

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
