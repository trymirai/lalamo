from collections.abc import Mapping
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike

from lalamo.initializer import EmptyInitializer
from lalamo.model_import.loaders.binding import Binder, Loader
from lalamo.model_import.loaders.huggingface import load_linear
from lalamo.model_import.model_configs.huggingface.weaver import HFWeaverConfig
from lalamo.module import LalamoModule
from lalamo.modules.mlp import DenseMLP
from lalamo.modules.normalization import Normalization
from lalamo.modules.speculators.weaver import Weaver
from lalamo.modules.token_mixers.attention import Attention
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.sharding import ShardingConfig
from lalamo.utils.surgery import load_as_at
from lalamo.weight_matrix import CompressionImplementation

__all__ = ["load_weaver"]


def load_weaver_norm(
    norm: Normalization,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,  # noqa: ARG001
) -> Normalization:
    scales = weights_dict[path / "weight"].astype(jnp.float32)
    biases = weights_dict[path / "bias"].astype(jnp.float32)
    return load_as_at(lambda module: (module.scales, module.biases), norm, (scales, biases))


def _load_weaver_attention(
    attention: Attention,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    *,
    implementation: CompressionImplementation,
) -> Attention:
    return load_as_at(
        lambda module: (module.qkvg_projection, module.out_projection),
        attention,
        (
            load_linear(attention.qkvg_projection, weights_dict, path / "qkv_proj", implementation=implementation),
            load_linear(attention.out_projection, weights_dict, path / "o_proj", implementation=implementation),
        ),
    )


def load_weaver_mlp(
    mlp: DenseMLP,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
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
            load_linear(mlp.up_projection, reordered_weights, gate_up_path, implementation=implementation),
            load_linear(mlp.down_projection, weights_dict, path / "down_proj", implementation=implementation),
        ),
    )


def _weaver_loader(module: LalamoModule) -> Loader:
    match module:
        case Normalization():
            return load_weaver_norm
        case Attention():
            return _load_weaver_attention
        case DenseMLP():
            return load_weaver_mlp
    raise TypeError(f"Unsupported Weaver primitive: {type(module).__name__}.")


def load_weaver(
    path: Path | str,
    sharding_config: ShardingConfig | None = None,
    dtype: DTypeLike | None = None,
) -> Weaver:
    import torch  # noqa: PLC0415

    payload = torch.load(path, map_location="cpu", weights_only=True)
    binder = Binder(_weaver_loader)
    config = HFWeaverConfig.from_dict(payload["config"]).build_weaver(binder)
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
            module.embedding_projection,
            module.hidden_state_projection,
            module.query_projection,
            module.program,
        ),
        weaver,
        (
            load_weaver_norm(weaver.embedding_norm, weights_dict, root / "embed_norm"),
            load_weaver_norm(weaver.hidden_state_norm, weights_dict, root / "output_norm"),
            load_linear(weaver.embedding_projection, weights_dict, root / "token_in"),
            load_linear(weaver.hidden_state_projection, weights_dict, root / "proposal_in"),
            load_linear(weaver.query_projection, weights_dict, root / "lm_head_query_in"),
            binder.load(weaver.program, config.program, weights_dict),
        ),
    )
