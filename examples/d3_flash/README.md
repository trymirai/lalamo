# D3 Flash format conversion

This folder contains a standalone converter from the pinned `vllm-sr/d3-flash` BF16 snapshot to Lalamo's asymmetric INT4, group-32, RHT32 format, plus a format verifier and CPU regression tests. The source is pinned to revision `521525effa79469d899a570726547dd7f00da3af`. It does not download or include model weights, tokenizer assets, or an inference engine.

Use Python `>=3.12,<3.13` (the commands below select 3.12) and install the locked CPU-capable project dependencies without the upstream development group:

```sh
uv sync --frozen --no-dev --extra cpu --python 3.12
```

Point the converter at an already available source snapshot and choose an output directory explicitly:

```sh
uv run --frozen --no-dev --extra cpu --python 3.12 python examples/d3_flash/convert_d3_mirai_format.py \
  --source /path/to/d3-flash-snapshot \
  --output /path/to/d3-flash-mirai-format-int4 \
  --backend cpu
```

The Metal backend needs the matching runtime plugin, installed for that invocation with `--with metaljax==0.11.9`:

```sh
uv run --frozen --no-dev --extra cpu --python 3.12 --with metaljax==0.11.9 \
  python examples/d3_flash/convert_d3_mirai_format.py \
  --source /path/to/d3-flash-snapshot \
  --output /path/to/d3-flash-mirai-format-int4 \
  --backend metal
```

The default report directory is next to the output and ends in `-report`. The source must contain the pinned `MODEL_MANIFEST.json`, source configuration, original checkpoint shards, decision readout, and tokenizer assets. `SOURCE_MODEL_MANIFEST.json` in this folder is a small checked-in copy of the source snapshot's `MODEL_MANIFEST.json` at the pinned revision. The converter pins and verifies that file's SHA-256; this copy is provenance metadata, not a model asset.

Check a completed output without loading its weights into a model:

```sh
uv run --frozen --no-dev --extra cpu --python 3.12 python examples/d3_flash/verify_d3_mirai_format.py \
  /path/to/d3-flash-snapshot \
  /path/to/d3-flash-mirai-format-int4
```

The verifier checks source identity, array coverage, native safetensors metadata and shapes, quantization layout, RHT signs, and exact preservation of the source passthrough tensors. Its report leaves runtime behavior, quality evaluation, and the full Mirai-M recipe as `not_assessed`. This conversion does not apply Fisher optimization or QAD, and the custom `D3DecisionModelConfig` still requires a compatible Uzu engine adapter.

Run the converter and verifier CPU regression tests with:

```sh
JAX_PLATFORMS=cpu uv run --frozen --no-dev --extra cpu --python 3.12 \
  --with pytest --with pytest-env python -m pytest -q \
  examples/d3_flash/test_convert_d3_mirai_format.py \
  examples/d3_flash/test_verify_d3_mirai_format.py
```
