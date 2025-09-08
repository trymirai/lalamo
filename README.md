<p align="center">
  <picture>
    <img alt="Mirai" src="https://artifacts.trymirai.com/social/github/lalamo-header.jpg" style="max-width: 100%;">
  </picture>
</p>

<a href="https://artifacts.trymirai.com/social/about_us.mp3"><img src="https://img.shields.io/badge/Listen-Podcast-red" alt="Listen to our podcast"></a>
<a href="https://docsend.com/v/76bpr/mirai2025"><img src="https://img.shields.io/badge/View-Deck-red" alt="View our deck"></a>
<a href="mailto:alexey@getmirai.co,dima@getmirai.co,aleksei@getmirai.co?subject=Interested%20in%20Mirai"><img src="https://img.shields.io/badge/Send-Email-green" alt="Contact us"></a>
<a href="https://docs.trymirai.com/components/models"><img src="https://img.shields.io/badge/Read-Docs-blue" alt="Read docs"></a>
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

# lalamo

A set of tools for adapting Large Language Models to on-device inference using the [uzu](https://github.com/trymirai/uzu) inference engine.

## Quick Start

To get the list of [supported models](https://trymirai.com/models), run:

```bash
uv run lalamo list-models
```

To convert a model, run:

```bash
uv run lalamo convert MODEL_REPO
```

Note: on some CPU platform you may be getting an error saying `The precision 'F16_F16_F32' is not supported by dot_general on CPU`. This is due to a bug in XLA, which causes matmuls inside `jax.jit` not work correctly on CPUs. The workaround is to set the environment variable `JAX_DISABLE_JIT=1` when running the conversion.

After that, you can find the converted model in the `models` folder. For more options see `uv run lalamo convert --help`.

## Model Support

To add support for a new model, write the corresponding [ModelSpec](lalamo/model_import/model_specs), as shown in the example below:

```python
ModelSpec(
    vendor="Google",
    family="Gemma-3",
    name="Gemma-3-1B-Instruct",
    size="1B",
    quantization=None,
    repo="google/gemma-3-1b-it",
    config_type=HFGemma3TextConfig,
    weights_type=WeightsType.SAFETENSORS,
)
```
