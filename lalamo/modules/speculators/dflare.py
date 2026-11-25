from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Bool, Float, Int

from lalamo.initializer import Initializer
from lalamo.kernels.attention import xla_attention
from lalamo.module import Keychain, LalamoConfig, LalamoModule
from lalamo.modules.linear import Linear, LinearConfig
from lalamo.modules.mlp import DenseMLP
from lalamo.modules.normalization import Normalization, NormalizationConfig
from lalamo.modules.residual import AdditiveResidualConfig
from lalamo.modules.rope import PositionalEmbeddings, RoPE, RoPEConfig
from lalamo.modules.speculator import Speculator, SpeculatorConfig
from lalamo.modules.speculators.dflash import _layer_attention, _layer_attention_config
from lalamo.modules.speculators.dspark import DSparkHead, DSparkHeadConfig, mix_taps
from lalamo.modules.transformer_layer import TransformerForwardPassConfig, TransformerLayer, TransformerLayerConfig
from lalamo.modules.utils import call_vmapped, call_vmapped_twice

__all__ = [
    "DFlareContext",
    "DFlareDraftConfig",
    "DFlareDraftModel",
    "DFlareLayerContext",
    "DFlareSpeculator",
    "DFlareSpeculatorConfig",
]


class DFlareLayerContext(NamedTuple):
    tap_weights: Float[Array, "taps channels"]
    norm: Normalization
    key_value_projection: Linear
    key_norm: Normalization


class DFlareContext(NamedTuple):
    """Keys and values every draft layer reads for committed rows: keys normalized and rotated to their positions."""

    keys: Float[Array, "layers rows groups head_channels"]
    values: Float[Array, "layers rows groups head_channels"]


@dataclass(frozen=True)
class DFlareDraftConfig(LalamoConfig):
    model_dim: int
    hidden_dim: int
    block_size: int
    target_layer_ids: tuple[int, ...]
    num_target_layers: int
    vocab_size: int
    context_projection_config: LinearConfig
    context_norm_config: NormalizationConfig
    rope_config: RoPEConfig
    layer_configs: tuple[TransformerLayerConfig, ...]
    output_norm_config: NormalizationConfig

    def init(self, initializer: Initializer) -> "DFlareDraftModel":
        attention_configs = tuple(_layer_attention_config(layer_config) for layer_config in self.layer_configs)
        return DFlareDraftModel(
            config=self,
            sharding_config=initializer.sharding_config,
            mask_embedding=initializer.zeros((self.model_dim,)),
            contexts=tuple(
                DFlareLayerContext(
                    tap_weights=initializer.zeros((len(self.target_layer_ids), self.model_dim), dtype=jnp.float32),
                    norm=self.context_norm_config.init(initializer, self.model_dim),
                    key_value_projection=self.context_projection_config.init(
                        initializer,
                        self.model_dim,
                        (attention_config.num_groups * attention_config.head_dim,) * 2,
                        has_biases=False,
                    ),
                    key_norm=self.context_norm_config.init(initializer, attention_config.head_dim),
                )
                for attention_config in attention_configs
            ),
            rope=self.rope_config.init(initializer),
            layers=tuple(
                layer_config.init(initializer, AdditiveResidualConfig(), self.model_dim, self.hidden_dim)
                for layer_config in self.layer_configs
            ),
            output_norm=self.output_norm_config.init(initializer, self.model_dim),
        )


class DFlareDraftModel(LalamoModule[DFlareDraftConfig]):
    """A block drafter that reads the target's hidden states through per-layer context keys and values.

    A committed row at position r contributes, to every draft layer, one key and one value projected from that
    layer's mix of the row's target taps (`project_context`). A block anchored at position p is the embedding of
    the anchor token followed by `block_size - 1` mask embeddings at positions p .. p + block_size - 1; its rows
    attend to each other in both directions and to the committed rows before p (`__call__`). A layer with a
    sliding window sees only the newest rows of the context, the same ones for every row of the block.

    Taps of a row, in the order of `config.target_layer_ids`: the hidden state after the target layer visit with
    that index, and for the index `num_target_layers - 1` the target's final hidden state after its output
    normalization (the loader folds the scales of that normalization into the tap weight, because lalamo keeps
    them in the LM head).
    """

    mask_embedding: Float[Array, " channels"]
    contexts: tuple[DFlareLayerContext, ...]
    rope: RoPE
    layers: tuple[TransformerLayer, ...]
    output_norm: Normalization

    def context_window(self, layer_index: int) -> int | None:
        """How many of the newest committed rows the layer sees; None for all of them.

        The layer config stores twice the trained window, the symmetric window of a non-causal lalamo attention.
        The trained window counts the anchor row, so one row fewer is context.
        """
        window = _layer_attention_config(self.config.layer_configs[layer_index]).sliding_window_size
        return None if window is None else window // 2 - 1

    @eqx.filter_jit
    def project_context(
        self,
        taps: Float[Array, "rows taps channels"],
        positions: Int[Array, " rows"],
        forward_pass_config: TransformerForwardPassConfig = TransformerForwardPassConfig(),
        *,
        keychain: Keychain,
    ) -> DFlareContext:
        """Context keys and values of committed rows from their target taps; rows are independent of each other."""
        mixer_config = forward_pass_config.mixer_forward_pass_config
        positional_embeddings = self.rope(positions).astype(mixer_config.rope_dtype)

        def project(
            layer: TransformerLayer,
            context: DFlareLayerContext,
            layer_keychain: Keychain,
        ) -> tuple[Float[Array, "rows groups head_channels"], Float[Array, "rows groups head_channels"]]:
            attention_config = _layer_attention(layer).config
            mixed = call_vmapped(partial(mix_taps, context.tap_weights, context.norm), taps)
            key_projection, value_projection = call_vmapped(
                context.key_value_projection,
                mixed.astype(taps.dtype),
                forward_pass_config=mixer_config.matmul_config,
                keychain=layer_keychain,
            )
            keys = rearrange(
                key_projection,
                "rows (groups head_channels) -> rows groups head_channels",
                groups=attention_config.num_groups,
                head_channels=attention_config.head_dim,
            )
            keys = call_vmapped_twice(context.key_norm, keys)
            keys = call_vmapped(positional_embeddings.apply, keys, in_axes=1, out_axes=1)
            values = rearrange(
                value_projection,
                "rows (groups head_channels) -> rows groups head_channels",
                groups=attention_config.num_groups,
                head_channels=attention_config.head_dim,
            )
            return keys.astype(jnp.float32), values.astype(jnp.float32)

        layer_keychains = keychain.split(len(self.layers))
        projected = tuple(
            project(layer, context, layer_keychain)
            for layer, context, layer_keychain in zip(self.layers, self.contexts, layer_keychains, strict=True)
        )
        return DFlareContext(
            keys=jnp.stack([keys for keys, _ in projected]),
            values=jnp.stack([values for _, values in projected]),
        )

    def _layer(
        self,
        layer: TransformerLayer,
        block: Float[Array, "block channels"],
        positional_embeddings: PositionalEmbeddings,
        context_keys: Float[Array, "rows groups head_channels"],
        context_values: Float[Array, "rows groups head_channels"],
        visible: Bool[Array, " rows"],
        forward_pass_config: TransformerForwardPassConfig,
        *,
        keychain: Keychain,
    ) -> Float[Array, "block channels"]:
        mixer, mlp = layer.mixer, layer.mlp
        attention, mlp_body = _layer_attention(layer), mlp.body
        plain_sublayers = all(
            sublayer.conv is None and sublayer.post_norm is None and sublayer.output_conv is None
            for sublayer in (mixer, mlp)
        )
        if mixer.pre_norm is None or mlp.pre_norm is None or not plain_sublayers or not isinstance(mlp_body, DenseMLP):
            raise ValueError(
                "DFlare draft layers are pre-norm attention and dense MLP sublayers without convolutions."
            )
        mixer_config = forward_pass_config.mixer_forward_pass_config
        projection_keychain, output_keychain, mlp_keychain = keychain.split(3)
        (block_size, _), dtype = block.shape, block.dtype

        queries, keys, values, _, _ = attention.project_heads(
            call_vmapped(mixer.pre_norm, block),
            positional_embeddings,
            mixer_config,
            keychain=projection_keychain,
        )
        assert keys is not None and values is not None
        # Every row of the block sees the same context rows and the whole block.
        mask = jnp.concatenate(
            [
                jnp.broadcast_to(visible[None, :], (block_size, visible.shape[0])),
                jnp.ones((block_size, block_size), dtype=jnp.bool_),
            ],
            axis=1,
        )
        attended = xla_attention(
            queries.astype(jnp.float32),
            jnp.concatenate([context_keys.astype(jnp.float32), keys.astype(jnp.float32)]),
            jnp.concatenate([context_values.astype(jnp.float32), values.astype(jnp.float32)]),
            None,
            mask,
            attention.config.scale,
            attention.config.logit_soft_cap,
        )
        (mixer_output,) = call_vmapped(
            attention.out_projection,
            rearrange(attended, "block heads head_channels -> block (heads head_channels)").astype(dtype),
            forward_pass_config=mixer_config.matmul_config,
            keychain=output_keychain,
        )
        block = block + mixer_output.astype(dtype)
        mlp_output = call_vmapped(
            mlp_body.call_unbatched,
            call_vmapped(mlp.pre_norm, block),
            forward_pass_config=forward_pass_config.mlp_forward_pass_config,
            keychain=mlp_keychain,
        )
        return block + mlp_output.astype(dtype)

    def __call__(
        self,
        anchor_embedding: Float[Array, " channels"],
        context: DFlareContext,
        context_positions: Int[Array, " rows"],
        context_valid: Bool[Array, " rows"],
        position: Int[Array, ""] | int,
        forward_pass_config: TransformerForwardPassConfig = TransformerForwardPassConfig(),
        *,
        keychain: Keychain,
    ) -> Float[Array, "proposals channels"]:
        """Normalized hidden states of the `block_size - 1` proposal positions of the block anchored at `position`.

        `anchor_embedding` is the target's input embedding of the anchor token, the newest token of the sequence,
        which sits at `position`. Context rows count when they are valid and lie before `position`, so the
        context of a whole sequence serves every anchor in it. The LM head of the target turns a returned row
        into the base logits of its position.
        """
        # The position enters the compiled block as an array, so anchors do not recompile it.
        return self._block(
            anchor_embedding,
            context,
            context_positions,
            context_valid,
            jnp.asarray(position, dtype=jnp.int32),
            forward_pass_config,
            keychain=keychain,
        )

    @eqx.filter_jit
    def _block(
        self,
        anchor_embedding: Float[Array, " channels"],
        context: DFlareContext,
        context_positions: Int[Array, " rows"],
        context_valid: Bool[Array, " rows"],
        position: Int[Array, ""],
        forward_pass_config: TransformerForwardPassConfig,
        *,
        keychain: Keychain,
    ) -> Float[Array, "proposals channels"]:
        block_size = self.config.block_size
        (channels,) = anchor_embedding.shape
        dtype = anchor_embedding.dtype
        block = jnp.concatenate(
            [
                anchor_embedding[None, :],
                jnp.broadcast_to(self.mask_embedding.astype(dtype), (block_size - 1, channels)),
            ],
        )
        context_positions = context_positions.astype(jnp.int32)
        positional_embeddings = self.rope(position + jnp.arange(block_size, dtype=jnp.int32)).astype(
            forward_pass_config.mixer_forward_pass_config.rope_dtype,
        )
        committed = context_valid & (context_positions < position)
        layer_keychains = keychain.split(len(self.layers))
        for layer_index, (layer, layer_keychain) in enumerate(zip(self.layers, layer_keychains, strict=True)):
            window = self.context_window(layer_index)
            visible = committed if window is None else committed & (context_positions >= position - window)
            block = self._layer(
                layer,
                block,
                positional_embeddings,
                context.keys[layer_index],
                context.values[layer_index],
                visible,
                forward_pass_config,
                keychain=layer_keychain,
            )
        hidden = call_vmapped(self.output_norm, block)
        return hidden[1:]


@dataclass(frozen=True)
class DFlareSpeculatorConfig(SpeculatorConfig):
    draft_config: DFlareDraftConfig
    dspark_config: DSparkHeadConfig

    def init(self, initializer: Initializer) -> "DFlareSpeculator":
        return DFlareSpeculator(
            config=self,
            sharding_config=initializer.sharding_config,
            draft_model=self.draft_config.init(initializer),
            dspark_head=self.dspark_config.init(
                initializer,
                self.draft_config.model_dim,
                self.draft_config.vocab_size,
                len(self.draft_config.target_layer_ids),
            ),
        )


class DFlareSpeculator(Speculator[DFlareSpeculatorConfig]):
    draft_model: DFlareDraftModel
    dspark_head: DSparkHead
