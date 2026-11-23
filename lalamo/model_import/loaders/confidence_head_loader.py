import json
from pathlib import Path

import jax.numpy as jnp
from jaxtyping import DTypeLike

from lalamo.exportable import ExportResults
from lalamo.initializer import EmptyInitializer
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.speculators.confidence_head import ConfidenceHead, ConfidenceHeadConfig
from lalamo.safetensors import safe_read
from lalamo.utils.sharding import ShardingConfig
from lalamo.weight_matrix import FullPrecisionSpec, Layout, WeightMatrixSpec


def load_confidence_head(
    directory: Path | str,
    *,
    target_model_dim: int,
    sharding_config: ShardingConfig,
    dtype: DTypeLike | None = None,
) -> ConfidenceHead:
    directory = Path(directory)
    with (directory / "config.json").open() as config_file:
        source = json.load(config_file)
    if source["format"] != 2:
        raise ValueError("Confidence-head checkpoint must use format 2")
    tree_attention_config = None
    if source["attention"] is not None:
        attention = source["attention"]
        tree_attention_config = {
            "model_dim": attention["width"],
            "num_layers": attention["layers"],
            "num_heads": attention["heads"],
            "hidden_dim": attention["mlp_hidden"],
        }
    config = ConfidenceHeadConfig.from_json(
        {
            "model_dim": source["local"]["hidden"],
            "pool_hidden_dim": source["local"]["pool_hidden"],
            "edge_hidden_dim": source["local"]["edge_hidden"],
            "logit_cap": source["local"]["logit_cap"],
            "max_depth": source["trained_with"]["tree"]["max_depth"],
            "norm_config": NormalizationConfig(
                epsilon=source["layer_norm_eps"],
                scale_offset=None,
                upcast_mode=UpcastMode.FULL_LAYER,
                subtract_mean=True,
                has_biases=True,
            ).to_json(),
            "tree_attention_config": tree_attention_config,
        },
    )
    template = config.init(EmptyInitializer(dtype, sharding_config), target_model_dim)
    with (directory / "model.safetensors").open("rb") as weights_file:
        encoded_metadata, source_arrays, _ = safe_read(weights_file)
        arrays = dict(source_arrays)
        metadata = {name: json.loads(value) for name, value in (encoded_metadata or {}).items()}

    for suffix in ("biases", "weights.weights"):
        arrays[f"draft_projection.{suffix}"] = arrays.pop(f"draft.{suffix}")
    metadata["draft_projection.weights.spec"] = metadata.pop("draft.weights.spec")
    if template.tree is not None:
        arrays["tree.relation_bias"] = arrays.pop("tree.relation_weight")

    for path in (
        "pool_set.entry",
        "pool_set.hidden",
        "pool_set.out",
        "edge.entry",
        "edge.up",
        "edge.down",
        "edge.output",
        "output",
    ):
        spec = WeightMatrixSpec.from_json(metadata.pop(f"{path}.weights.spec"))
        if spec != FullPrecisionSpec(Layout.OUTPUT_INPUT):
            raise ValueError(f"Confidence-head {path} must use full-precision output_input weights")
        arrays[f"{path}.weights"] = arrays.pop(f"{path}.weights.weights")

    keep = arrays.pop("slot.keep")
    if keep.shape != (4,) or not bool(jnp.all(keep == 1)):
        raise ValueError("Confidence-head slot.keep must retain all four features")
    arrays["slot.stats"] = jnp.stack((arrays.pop("slot.means"), arrays.pop("slot.stds")))
    for path in ("edge.input_norm", "edge.block_norm", "edge.final_norm"):
        arrays[path] = jnp.stack((arrays.pop(f"{path}.scales"), arrays.pop(f"{path}.biases")))

    for suffix in ("weights", "biases"):
        (edge_readout,) = arrays.pop(f"edge.output.{suffix}")
        alpha, beta, third = arrays.pop(f"output.{suffix}")
        if bool(jnp.any(third != 0)):
            raise ValueError("Confidence-head third node readout must be zero")
        arrays[f"edge.readout.{suffix}"] = jnp.stack((edge_readout, alpha, beta))

    return template.load_exported(ExportResults(arrays, metadata))
