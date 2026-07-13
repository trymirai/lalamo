from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import torch
from jaxtyping import Array, DTypeLike

from lalamo.initializer import EmptyInitializer
from lalamo.model_import.loaders.huggingface import load_linear
from lalamo.model_import.model_configs.huggingface.weaver import HFWeaverConfig
from lalamo.modules import Normalization
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
            module.up_projection,
            module.down_projection,
        ),
        block,
        (
            load_weaver_norm(block.pre_attention_norm, weights_dict, path / "norm_attn"),
            load_linear(block.qkv_projection, weights_dict, path, sublayers_to_fuse=["q_proj", "k_proj", "v_proj"]),
            load_linear(block.out_projection, weights_dict, path / "o_proj"),
            load_weaver_norm(block.pre_mlp_norm, weights_dict, path / "norm_mlp"),
            load_linear(block.up_projection, weights_dict, path / "fc1"),
            load_linear(block.down_projection, weights_dict, path / "fc2"),
        ),
    )


def load_weaver(
    path: Path | str,
    sharding_config: ShardingConfig | None = None,
    dtype: DTypeLike | None = None,
) -> Weaver:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    speculator_kind = payload.get("metadata", {}).get("speculator_kind")
    if speculator_kind != "dflash_tfm_weaver":
        raise ValueError(
            "Expected a Weaver checkpoint with metadata.speculator_kind='dflash_tfm_weaver', "
            f"got {speculator_kind!r}.",
        )
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
            module.position_embeddings,
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
            weights_dict[root / "pos_emb"],
        ),
    )
