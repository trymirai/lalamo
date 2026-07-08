from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import torch
from jaxtyping import Array

from lalamo.initializer import EmptyInitializer
from lalamo.model_import.loaders.huggingface import load_linear
from lalamo.modules import LinearConfig, Normalization, NormalizationConfig
from lalamo.modules.normalization import UpcastMode
from lalamo.speculator.weaver import Weaver, WeaverBlock, WeaverConfig
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
            module.norm_attn,
            module.q_proj,
            module.k_proj,
            module.v_proj,
            module.o_proj,
            module.norm_mlp,
            module.fc1,
            module.fc2,
        ),
        block,
        (
            load_weaver_norm(block.norm_attn, weights_dict, path / "norm_attn"),
            load_linear(block.q_proj, weights_dict, path / "q_proj"),
            load_linear(block.k_proj, weights_dict, path / "k_proj"),
            load_linear(block.v_proj, weights_dict, path / "v_proj"),
            load_linear(block.o_proj, weights_dict, path / "o_proj"),
            load_weaver_norm(block.norm_mlp, weights_dict, path / "norm_mlp"),
            load_linear(block.fc1, weights_dict, path / "fc1"),
            load_linear(block.fc2, weights_dict, path / "fc2"),
        ),
    )


def load_weaver(path: Path | str, sharding_config: ShardingConfig | None = None) -> Weaver:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config_dict = payload["config"]
    config = WeaverConfig(
        d_model=config_dict["d_model"],
        d_embed=config_dict["d_embed"],
        d_rank=config_dict["d_rank"],
        num_layers=config_dict["num_layers"],
        num_heads=config_dict["num_heads"],
        mlp_dim=config_dict["mlp_dim"],
        k=config_dict["K"],
        candidate_pool_size=config_dict["candidate_pool_size"],
        linear_config=LinearConfig(),
        norm_config=NormalizationConfig(
            epsilon=1e-6,
            scale_offset=None,
            upcast_mode=UpcastMode.FULL_LAYER,
            subtract_mean=False,
            has_biases=True,
        ),
    )
    sharding_config = sharding_config or ShardingConfig.replicated()
    weaver = config.init(EmptyInitializer(jnp.float32, sharding_config))
    weights_dict: dict[str, Array] = {
        str(name): jnp.asarray(tensor.float().numpy()) for name, tensor in payload["state_dict"].items()
    }
    root = ParameterPath()
    return eqx.tree_at(
        lambda module: (
            module.embed_norm,
            module.output_norm,
            module.out_norm,
            module.token_in,
            module.proposal_in,
            module.lm_head_query_in,
            module.blocks,
            module.pos_emb,
        ),
        weaver,
        (
            load_weaver_norm(weaver.embed_norm, weights_dict, root / "embed_norm"),
            load_weaver_norm(weaver.output_norm, weights_dict, root / "output_norm"),
            load_weaver_norm(weaver.out_norm, weights_dict, root / "out_norm"),
            load_linear(weaver.token_in, weights_dict, root / "token_in"),
            load_linear(weaver.proposal_in, weights_dict, root / "proposal_in"),
            load_linear(weaver.lm_head_query_in, weights_dict, root / "lm_head_query_in"),
            tuple(
                load_weaver_block(block, weights_dict, root / "blocks" / index)
                for index, block in enumerate(weaver.blocks)
            ),
            weights_dict[root / "pos_emb"],
        ),
    )
