from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike

from lalamo.initializer import EmptyInitializer
from lalamo.model_import.loaders.huggingface import load_linear
from lalamo.model_import.model_configs.huggingface.weaver import HFWeaverConfig
from lalamo.modules import DenseMLP, Normalization
from lalamo.modules.speculators.weaver import Weaver, WeaverBlock
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.sharding import ShardingConfig
from lalamo.utils.surgery import load_as_at

__all__ = [
    "load_weaver",
]


def load_weaver_norm(
    norm: Normalization,
    weights_dict: dict[str, Array],
    path: ParameterPath,
) -> Normalization:
    scales = weights_dict[path / "weight"].astype(jnp.float32)
    biases = weights_dict[path / "bias"].astype(jnp.float32)
    return load_as_at(lambda module: (module.scales, module.biases), norm, (scales, biases))


def load_weaver_block(
    block: WeaverBlock,
    weights_dict: dict[str, Array],
    path: ParameterPath,
) -> WeaverBlock:
    return eqx.tree_at(
        lambda module: (
            module.pre_attention_norm,
            module.qkv_projection,
            module.out_projection,
            module.pre_mlp_norm,
            module.mlp,
        ),
        block,
        (
            load_weaver_norm(block.pre_attention_norm, weights_dict, path / "norm_attn"),
            load_linear(block.qkv_projection, weights_dict, path / "qkv_proj"),
            load_linear(block.out_projection, weights_dict, path / "o_proj"),
            load_weaver_norm(block.pre_mlp_norm, weights_dict, path / "norm_mlp"),
            load_weaver_mlp(block.mlp, weights_dict, path),
        ),
    )


def load_weaver_mlp(
    mlp: DenseMLP,
    weights_dict: dict[str, Array],
    path: ParameterPath,
) -> DenseMLP:
    gate_up_path = path / "gate_up_proj"
    gate_weights, up_weights = jnp.split(weights_dict[gate_up_path / "weight"], 2, axis=0)
    gate_biases, up_biases = jnp.split(weights_dict[gate_up_path / "bias"], 2, axis=0)
    reordered_weights = dict(weights_dict)
    reordered_weights[gate_up_path / "weight"] = jnp.concatenate([up_weights, gate_weights], axis=0)
    reordered_weights[gate_up_path / "bias"] = jnp.concatenate([up_biases, gate_biases], axis=0)
    return eqx.tree_at(
        lambda module: (module.up_projection, module.down_projection),
        mlp,
        (
            load_linear(mlp.up_projection, reordered_weights, gate_up_path),
            load_linear(mlp.down_projection, weights_dict, path / "down_proj"),
        ),
    )


def load_weaver(
    path: Path | str,
    sharding_config: ShardingConfig | None = None,
    dtype: DTypeLike | None = None,
) -> Weaver:
    import torch  # noqa: PLC0415

    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = HFWeaverConfig.from_dict(payload["config"]).to_weaver_config()
    sharding_config = sharding_config or ShardingConfig.replicated()
    weaver = config.init(EmptyInitializer(dtype, sharding_config))
    weights_dict: dict[str, Array] = {
        str(name): jnp.asarray(tensor.float().numpy()) for name, tensor in payload["state_dict"].items()
    }
    root = ParameterPath()
    return eqx.tree_at(
        lambda module: (
            module.embedding_norm,
            module.hidden_state_norm,
            module.output_norm,
            module.embedding_projection,
            module.hidden_state_projection,
            module.query_projection,
            module.blocks,
        ),
        weaver,
        (
            load_weaver_norm(weaver.embedding_norm, weights_dict, root / "embed_norm"),
            load_weaver_norm(weaver.hidden_state_norm, weights_dict, root / "output_norm"),
            load_weaver_norm(weaver.output_norm, weights_dict, root / "out_norm"),
            load_linear(weaver.embedding_projection, weights_dict, root / "token_in"),
            load_linear(weaver.hidden_state_projection, weights_dict, root / "proposal_in"),
            load_linear(weaver.query_projection, weights_dict, root / "lm_head_query_in"),
            tuple(
                load_weaver_block(block, weights_dict, root / "blocks" / index)
                for index, block in enumerate(weaver.blocks)
            ),
        ),
    )
