<p align="center">
  <picture>
    <img alt="Mirai" src="https://artifacts.trymirai.com/social/github/lalamo-header.jpg" style="max-width: 100%;">
  </picture>
</p>

<a href="https://artifacts.trymirai.com/social/about_us.mp3"><img src="https://img.shields.io/badge/Listen-Podcast-red" alt="Listen to our podcast"></a>
<a href="https://docsend.com/v/76bpr/mirai2025"><img src="https://img.shields.io/badge/View-Deck-red" alt="View our deck"></a>
<a href="https://discord.com/invite/trymirai"><img src="https://img.shields.io/discord/1377764166764462120?label=Discord" alt="Discord"></a>
<a href="mailto:contact@getmirai.co?subject=Interested%20in%20Mirai"><img src="https://img.shields.io/badge/Send-Email-green" alt="Contact us"></a>
<a href="https://docs.trymirai.com/overview/lalamo"><img src="https://img.shields.io/badge/Read-Docs-blue" alt="Read docs"></a>
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

# lalamo

A set of tools for adapting Large Language Models to on-device inference using the [uzu](https://github.com/trymirai/uzu) inference engine.

## Changes in this fork

The initial scope of this fork is quantization and verified export for the D3
decision-model series. Support is tracked per model and source revision, starting
with the pinned original D3-Flash BF16 checkpoint. Its workflow exports asymmetric
INT4 group32 weights with block32 randomized Hadamard transforms, while preserving
the 255-code decision head in FP32. Two shared Lalamo fixes accompany the example.

See the [support scope](examples/d3_flash/README.md#initial-scope-and-supported-checkpoints)
for the implemented checkpoint and the requirements for adding another D3 model.

| Change | Purpose | Scope |
| --- | --- | --- |
| [DeltaNet loader dtype fix](lalamo/model_import/loaders/huggingface.py) | Cast `dt_bias` and `A_log` using the destination parameter's `.dtype`, supporting both empty templates and materialized arrays. | Shared Hugging Face loader; not restricted to D3. |
| [YAQA block update](lalamo/compressed/utils/yaqa.py) | Replace the inner sequential block-write loop with a scatter that drops padded writes, retaining the existing rounding and anti-diagonal order. | Shared blockwise YAQA path; the D3 recipe does not enable Fisher optimization. |
| [D3 conversion and verification](examples/d3_flash/README.md) | Map the Qwen3.5 decision backbone, stream conversion, preserve its readout and formatter contract, and verify the resulting bundle against the source. | Standalone scripts under `examples/d3_flash`; no default model-registry change. |

The [developer guide](examples/d3_flash/README.md) explains the architecture,
quantization policy, failure checks, output format, and reproducible commands.
It also separates the **168 passing CPU regression tests** from quality and
full-model runtime evaluation, which those tests do not establish.
The separate [Apple M6 three-way measurements](examples/d3_flash/reports/apple-m6-threeway-2026-10-15.md)
compare original BF16, simple MLX INT4 and the retained final Mirai INT4 artifact
on 1,356 frozen choice requests, including quality, probability and timing results.

The D3 branch includes upstream through
[`ad76c7d0`](https://github.com/trymirai/lalamo/commit/ad76c7d0cb474c9ff1562072bb1316037db2e6c2).
Against that baseline, existing runtime changes are confined to the two files
above; existing regression tests are retained. D3 scripts and their tests are
added under `examples/d3_flash`. The output uses `D3DecisionModelConfig` and
requires a compatible inference-engine adapter. Weights and that engine adapter
are not distributed in this repository.

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

## Optional Features

### PyAudio

PyAudio enables audio playback for TTS models and is used as an optional Lalamo feature because it requires [PortAudio](https://www.portaudio.com/) to be installed.

How to:

- **macOS**: `brew install portaudio` ([formula](https://formulae.brew.sh/formula/portaudio))
- **Debian/Ubuntu**: `apt-get install portaudio19-dev python-all-dev`
- **Other Linux**: [PortAudio build instructions](https://www.portaudio.com/docs/v19-doxydocs/compile_linux.html)

Then run :

```bash
uv run --with  pyaudio lalamo path/to/model --replay
```
