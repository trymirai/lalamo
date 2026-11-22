from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path

from jaxtyping import Array, DTypeLike

from lalamo.initializer import EmptyInitializer
from lalamo.model_import.common import _combine_weight_shards
from lalamo.model_import.model_configs.huggingface.dspark import HFDSparkConfig
from lalamo.model_import.origins import LocalOrigin, WeightFormat
from lalamo.modules.speculators.dspark import DSparkLayer, DSparkSpeculator, MarkovHead, TapMix
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.sharding import ShardingConfig
from lalamo.utils.surgery import load_as_at
from lalamo.weight_matrix import CompressionImplementation

from .huggingface import load_linear, load_rmsnorm, load_transformer_layer

__all__ = [
    "DSPARK_WEIGHTS_FILE",
    "load_dspark",
    "load_hf_dspark",
]

DSPARK_WEIGHTS_FILE = "mtp.safetensors"


def _load_tap_mix(
    module: TapMix,
    mix_logits: Array,
    weights_dict: Mapping[str, Array],
    norm_path: ParameterPath,
) -> TapMix:
    return load_as_at(
        lambda mix: (mix.mix_logits, mix.norm),
        module,
        (mix_logits.astype(module.mix_logits.dtype), load_rmsnorm(module.norm, weights_dict, norm_path)),
    )


def _load_layer(
    module: DSparkLayer,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    mix_logits: Array,
    *,
    implementation: CompressionImplementation,
) -> DSparkLayer:
    attention_path = path / "self_attn"
    transformer_layer = load_transformer_layer(
        module.transformer_layer,
        weights_dict,
        path,
        path,
        "self_attn",
        "mlp",
        "input_layernorm",
        "post_attention_layernorm",
        "up_proj",
        "gate_proj",
        "down_proj",
        permute_conv=False,
        implementation=implementation,
    )
    return load_as_at(
        lambda layer: (
            layer.transformer_layer,
            layer.context_mix,
            layer.context_kv_projection,
            layer.context_key_norm,
        ),
        module,
        (
            transformer_layer,
            _load_tap_mix(module.context_mix, mix_logits, weights_dict, path / "dflare_norm"),
            load_linear(module.context_kv_projection, weights_dict, attention_path, ["k_proj_ctx", "v_proj_ctx"]),
            load_rmsnorm(module.context_key_norm, weights_dict, attention_path / "k_norm_ctx"),
        ),
    )


def _load_markov_head(
    module: MarkovHead,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    mix_logits: Array,
    *,
    implementation: CompressionImplementation,
) -> MarkovHead:
    return load_as_at(
        lambda head: (head.context_mix, head.token_embeddings, head.recurrence, head.output),
        module,
        (
            _load_tap_mix(module.context_mix, mix_logits, weights_dict, path / "dflare_norm"),
            weights_dict[path / "markov_w1" / "weight"].astype(module.token_embeddings.dtype),
            load_linear(module.recurrence, weights_dict, path / "rnn_proj", implementation=implementation),
            load_linear(module.output, weights_dict, path / "hidden_out", implementation=implementation),
        ),
    )


def load_dspark(
    module: DSparkSpeculator,
    weights_dict: Mapping[str, Array],
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> DSparkSpeculator:
    # One row of mix logits per draft layer, then the Markov head's.
    mix_logits = weights_dict[ParameterPath("fuse_weight")]
    layers = tuple(
        _load_layer(
            layer,
            weights_dict,
            ParameterPath("layers") / index,
            mix_logits[index],
            implementation=implementation,
        )
        for index, layer in enumerate(module.layers)
    )
    return load_as_at(
        lambda speculator: (
            speculator.mask_embedding,
            speculator.layers,
            speculator.output_norm,
            speculator.markov_head,
        ),
        module,
        (
            weights_dict[ParameterPath("mask_embed")].astype(module.mask_embedding.dtype),
            layers,
            load_rmsnorm(module.output_norm, weights_dict, ParameterPath("norm")),
            _load_markov_head(
                module.markov_head,
                weights_dict,
                ParameterPath("markov_head"),
                mix_logits[len(module.layers)],
                implementation=implementation,
            ),
        ),
    )


def load_hf_dspark(
    target_dir: Path | str,
    *,
    sharding_config: ShardingConfig,
    dtype: DTypeLike | None = None,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> DSparkSpeculator:
    """Loads the DSpark draft bundled with a target checkpoint: its `mtp_config` and `mtp.safetensors`."""
    target_dir = Path(target_dir)
    config = HFDSparkConfig.from_target_json(target_dir / "config.json").to_dspark_config()
    template = config.init(EmptyInitializer(dtype, sharding_config))
    origin = LocalOrigin(
        root=str(target_dir),
        weight_files=(DSPARK_WEIGHTS_FILE,),
        weight_format=WeightFormat.SAFETENSORS,
    )
    with ExitStack() as stack:
        weight_shards = tuple(stack.enter_context(weight_shard) for weight_shard in origin.get_weights())
        checkpoint = _combine_weight_shards(weight_shards)
        return load_dspark(template, checkpoint.weights, implementation=implementation)
