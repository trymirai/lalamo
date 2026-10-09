import json
import os
from pathlib import Path

import jax
import pytest

from lalamo.models.language_model import LanguageModel
from lalamo.safetensors import safe_read
from lalamo.utils.sharding import ShardingConfig
from tests.unit.compressed.test_packed_checkpoint import assert_loaded_every_saved_tensor, saved_i4s4, weight_matrices

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif("PACKED_CHECKPOINT_DIRECTORY" not in os.environ, reason="requires a packed checkpoint package"),
]


def test_packed_checkpoint_loads_every_saved_tensor() -> None:
    directory = Path(os.environ["PACKED_CHECKPOINT_DIRECTORY"])
    # The test session exposes eight CPU devices; replicating the checkpoint on each would hold eight copies.
    config = ShardingConfig.replicated(jax.devices()[:1])
    with jax.set_mesh(config.mesh), (directory / "model.safetensors").open("rb") as stream:
        model = LanguageModel.load(directory, config)
        metadata, saved = safe_read(stream)
        trellis_paths = [name.removesuffix(".gains") for name in saved if name.endswith(".gains")]
        folds = {
            f"{path}.scales": (
                f"{path}.scales",
                f"{path}.gains",
                *sorted(name for name in saved if name.startswith(f"{path}.pre_gains.")),
                *sorted(name for name in saved if name.startswith(f"{path}.post_gains.")),
            )
            for path in trellis_paths
        }
        specs = {name.removesuffix(".spec"): json.loads(value) for name, value in (metadata or {}).items()}
        # (path, loaded path, first row, spec) of each saved leaf; a fused qkvg projection ends with the gate rows.
        leaves = [(path, path, 0, spec) for path, spec in specs.items()] + [
            (f"{path}.parts.{index}", path, sum(rows for rows, _ in spec["parts"][:index]), part)
            for path, spec in specs.items()
            for index, (_, part) in enumerate(spec.get("parts", ()))
        ]
        folded = [leaf for leaf in leaves if (leaf[3]["type"], leaf[3].get("layout")) == ("I4S4Spec", "output_input")]
        prefixes = tuple(f"{path}." for path, *_ in folded)
        unfolded = {name: saved[name] for name in saved if not name.startswith(prefixes)}
        assert_loaded_every_saved_tensor(model, unfolded, folds)
        for path, loaded_path, part_row, spec in folded:
            reference = saved_i4s4(saved, path, spec.get("post_gain_axes", ()), config)
            fused_path = loaded_path.replace(".qkv_proj", ".qkvg_proj").replace(".gate_proj", ".qkvg_proj")
            first_row = part_row - len(reference) * (".gate_projection." in path)
            loaded = weight_matrices(model)[fused_path].astype("float32").decompress()[first_row:][: len(reference)]
            assert ((loaded - reference) ** 2).sum() < 1e-4 * (reference**2).sum(), path
