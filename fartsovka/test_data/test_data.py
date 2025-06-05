import jax
import jax.numpy as jnp
from jaxtyping import Int, Bool, Array
from dataclasses import dataclass

from fartsovka.common import ParameterDict, DType
from fartsovka.modules import Decoder, WeightLayout, DecoderLayer
from fartsovka.modules.linear import LinearBase
from fartsovka.modules.attention import Attention
from fartsovka.modules.rope import PositionalEmbeddings
from fartsovka.modules.normalization import RMSNorm
from fartsovka.modules.mlp import MLP
from fartsovka.modules.activations import Activation


@dataclass
class ModuleSample:
    inputs: tuple[Array, ...]
    outputs: tuple[Array, ...]

    def export(self) -> ParameterDict:
        return ParameterDict(
            inputs=self.inputs,
            outputs=self.outputs
        )


class TestDataContext:
    rnd_key: Array
    sequence_length: int
    weights_layout: WeightLayout

    global_positional_embeddings: PositionalEmbeddings
    local_positional_embeddings: PositionalEmbeddings

    token_ids: Int[Array, "suffix_tokens"]
    token_positions: Int[Array, "suffix_tokens"]
    mask: Bool[Array, "suffix_tokens total_tokens"]

    def __init__(self, seed: int, sequence_length: int, model: Decoder, weights_layout: WeightLayout):
        self.rnd_key = jax.random.PRNGKey(seed)
        self.sequence_length = sequence_length
        self.weights_layout = weights_layout

        self.token_ids = jax.random.randint(
            self.get_rnd_key(),
            (sequence_length,),
            minval=0,
            maxval=10000,
            dtype=int
        ).astype(jnp.uint64)
        self.token_positions = jnp.arange(sequence_length).astype(jnp.uint64)
        self.mask = jnp.tril(jnp.ones((sequence_length, sequence_length), dtype=bool))

        self.global_positional_embeddings = model.global_rope(self.token_positions)
        if model.local_rope:
            self.local_positional_embeddings = model.local_rope(self.token_positions)
        else:
            self.local_positional_embeddings = self.global_positional_embeddings

    def get_rnd_key(self) -> Array:
        key_1, key_2 = jax.random.split(self.rnd_key, num=2)
        self.rnd_key = key_1
        return key_2

    def apply_layout_to_array(self, array: Array) -> Array:
        match self.weights_layout:
            case WeightLayout.INPUT_OUTPUT:
                return jnp.transpose(array)
            case WeightLayout.OUTPUT_INPUT:
                return array
            case _:
                return array

    def apply_layout_to_arrays(self, arrays: tuple[Array, ...]) -> tuple[Array, ...]:
        return tuple([self.apply_layout_to_array(array) for array in arrays])

    def export(self) -> ParameterDict:
        def export_positional_embedding(positional_embedding: PositionalEmbeddings) -> ParameterDict:
            return ParameterDict(
                cosines=positional_embedding.cosines,
                sines=positional_embedding.sines,
            )

        return ParameterDict(
            token_ids=self.token_ids,
            token_positions=self.token_positions,
            mask=self.mask,
            global_positional_embeddings=export_positional_embedding(self.global_positional_embeddings),
            local_positional_embeddings=export_positional_embedding(self.local_positional_embeddings)
        )


def export_test_data_activation(context: TestDataContext, module: Activation, shape: [int], precision: DType) -> ParameterDict:
    sample_input = jax.random.uniform(
        context.get_rnd_key(),
        shape,
        minval=-2,
        maxval=2,
        dtype=precision,
    )
    sample_output = module(sample_input)
    sample = ModuleSample(inputs=(sample_input,), outputs=(sample_output,))
    return ParameterDict(value=sample.export())


def export_test_data_linear(context: TestDataContext, module: LinearBase) -> ParameterDict:
    sample_input = context.apply_layout_to_array(jax.random.uniform(
        context.get_rnd_key(),
        (context.sequence_length, module.input_dim),
        minval=-2,
        maxval=2,
        dtype=module.config.precision,
    ))
    sample_outputs = context.apply_layout_to_arrays(module(sample_input))
    sample = ModuleSample(inputs=(sample_input,), outputs=sample_outputs)
    return ParameterDict(value=sample.export())


def export_test_data_mlp(context: TestDataContext, module: MLP) -> ParameterDict:
    sample_input = context.apply_layout_to_array(jax.random.uniform(
        context.get_rnd_key(),
        (context.sequence_length, module.model_dim),
        minval=-2,
        maxval=2,
        dtype=module.config.linear_config.precision,
    ))
    sample_output = context.apply_layout_to_array(module(sample_input))
    sample = ModuleSample(inputs=(sample_input,), outputs=(sample_output,))
    return ParameterDict(
        value=sample.export(),
        up_projection=export_test_data_linear(context, module.up_projection),
        down_projection=export_test_data_linear(context, module.down_projection),
        activation=export_test_data_activation(context, module.config.activation, [context.sequence_length, module.hidden_dim], module.config.linear_config.precision),
    )


def export_test_data_normalization(context: TestDataContext, module: RMSNorm) -> ParameterDict:
    sample_input = jax.random.uniform(
        context.get_rnd_key(),
        (context.sequence_length, module.input_dim),
        minval=-1,
        maxval=1,
        dtype=module.config.scale_precision,
    )
    sample_output = module(sample_input)
    sample = ModuleSample(inputs=(sample_input,), outputs=(sample_output,))
    return ParameterDict(value=sample.export())


def export_test_data_attention(context: TestDataContext, module: Attention) -> ParameterDict:
    sample_input = jax.random.uniform(
            context.get_rnd_key(),
            (context.sequence_length, module.model_dim),
            minval=-3,
            maxval=3,
            dtype=module.config.out_projection_config.precision,
        )
    sample_output = module(sample_input, context.global_positional_embeddings, context.local_positional_embeddings, None, context.mask, False)
    sample = ModuleSample(inputs=(sample_input,), outputs=(sample_output.attention_output,))

    return ParameterDict(
        value=sample.export(),
        qkv_projection=export_test_data_linear(context, module.qkv_projection),
        out_projection=export_test_data_linear(context, module.out_projection),
    )


def export_test_data_decoder_layer(context: TestDataContext, module: DecoderLayer) -> ParameterDict:
    sample_input = jax.random.uniform(
        context.get_rnd_key(),
        (context.sequence_length, module.attention.model_dim),
        minval=-4,
        maxval=4,
        dtype=module.config.mlp_config.linear_config.precision,
    )
    sample_output = module(sample_input, context.global_positional_embeddings, context.local_positional_embeddings, None, context.mask, False)
    sample = ModuleSample(inputs=(sample_input,), outputs=(sample_output.output,))

    result = ParameterDict(
        value=sample.export(),
        pre_attention_norm=export_test_data_normalization(context, module.pre_attention_norm),
        attention=export_test_data_attention(context, module.attention),
        pre_mlp_norm=export_test_data_normalization(context, module.pre_mlp_norm),
        mlp=export_test_data_mlp(context, module.mlp),
    )
    if module.post_attention_norm is not None:
        result["post_attention_norm"] = export_test_data_normalization(context, module.post_attention_norm)
    if module.post_mlp_norm is not None:
        result["post_mlp_norm"] = export_test_data_normalization(context, module.post_mlp_norm)
    return result


def export_test_data_decoder(context: TestDataContext, module: Decoder) -> ParameterDict:
    embed_sample_input = context.token_ids
    embed_sample_output = module.embedding.embed(embed_sample_input)
    embed_sample = ModuleSample(inputs=(embed_sample_input,), outputs=(embed_sample_output,))

    test_data_layers: [ParameterDict] = []
    for layer in module.layers:
        test_data_layer = export_test_data_decoder_layer(context, layer)
        test_data_layers.append(test_data_layer)

    readout_sample_input = context.apply_layout_to_array(jax.random.uniform(
        context.get_rnd_key(),
        (context.sequence_length, module.config.model_dim),
        minval=-5,
        maxval=5,
        dtype=module.config.embedding_config.precision,
    ))
    readout_sample_output = context.apply_layout_to_array(module.embedding.readout(readout_sample_input))
    readout_sample = ModuleSample(inputs=(readout_sample_input,), outputs=(readout_sample_output,))

    sample_input = context.token_ids
    sample_output = module(sample_input, context.token_positions, None, context.mask, False)
    sample = ModuleSample(inputs=(sample_input,), outputs=(sample_output.output,))

    return ParameterDict(
        value=sample.export(),
        embed=ParameterDict(value=embed_sample.export()),
        layers=test_data_layers,
        output_norm=export_test_data_normalization(context, module.output_norm),
        readout=ParameterDict(value=readout_sample.export()),
    )


def export_test_data(model: Decoder, weights_layout: WeightLayout) -> ParameterDict:
    jax_config_enable_x64_key = "jax_enable_x64"
    jax_config_enable_x64_value = jax.config.read(jax_config_enable_x64_key)
    jax.config.update(jax_config_enable_x64_key, True)

    context = TestDataContext(42, 16, model, weights_layout)
    decoder_test_data = export_test_data_decoder(context, model)
    context_test_data = context.export()

    jax.config.update(jax_config_enable_x64_key, jax_config_enable_x64_value)

    return ParameterDict(
        decoder=decoder_test_data,
        context=context_test_data,
    )