<p align="center">
  <picture>
    <img alt="Mirai" src="https://artifacts.trymirai.com/social/github/header.jpg" style="max-width: 100%;">
  </picture>
</p>

<a href="https://notebooklm.google.com/notebook/5851ef05-463e-4d30-bd9b-01f7668e8f8f/audio"><img src="https://img.shields.io/badge/Listen-Podcast-red" alt="Listen to our Podcast"></a>
<a href="https://docsend.com/view/x87pcxrnqutb9k2q"><img src="https://img.shields.io/badge/View-Our%20Deck-green" alt="View our Deck"></a>
<a href="mailto:alexey@getmirai.co,dima@getmirai.co,aleksei@getmirai.co?subject=Interested%20in%20Mirai"><img src="https://img.shields.io/badge/Contact%20Us-Email-blue" alt="Contact Us"></a>
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

# lalamo

A set of tools for adapting Large Language Models to on-device inference using the [uzu](https://github.com/trymirai/uzu) inference engine.

## Quick Start

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
