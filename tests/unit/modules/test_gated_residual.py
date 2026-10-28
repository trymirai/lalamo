import jax
import jax.numpy as jnp
import pytest
import torch
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextDecoderLayer, Qwen4ExpTextGatedResidual

from lalamo.model_import.loaders.huggingface import load_gated_residual_connection, load_gated_residual_head
from lalamo.modules import DecoderForwardPassConfig, EmbeddingForwardPassConfig, TransformerForwardPassConfig
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.torch_interop import torch_to_jax
from lalamo.weight_matrix import MatmulConfig
from tests.common import assert_close
from tests.helpers import build_tiny_attention_decoder
from tests.unit.modules.qwen4_exp import (
    HIDDEN_SIZE,
    STREAM_COUNT,
    as_stream,
    float_weights,
    load_decoder_layer,
    make_gated_residual_config,
    make_hf_config,
    make_hyper_input,
    make_initializer,
    make_keychain,
    merge_streams,
    randomized,
)


@pytest.mark.usefixtures("fake_mesh")
def test_connection_matches_hf() -> None:
    hf_connection = randomized(Qwen4ExpTextGatedResidual(make_hf_config()))
    connection = load_gated_residual_connection(
        make_gated_residual_config().init_connection(make_initializer(), HIDDEN_SIZE),
        float_weights(hf_connection),
        ParameterPath(),
    )
    hyper_input = make_hyper_input()
    sublayer_matrix = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE)

    with torch.no_grad():
        mixed_input, hf_hyper_input, injection_weights = hf_connection(hyper_input)
        update = mixed_input @ sublayer_matrix
        reference = hf_hyper_input + (update.unsqueeze(-2) * injection_weights.unsqueeze(-1)).flatten(-2)

    result, _ = connection(
        as_stream(hyper_input),
        lambda inputs, _: (inputs @ torch_to_jax(sublayer_matrix), None),
        MatmulConfig.for_tracer_tests(),
        keychain=make_keychain(),
    )

    assert_close(result=merge_streams(result), reference=torch_to_jax(reference))


@pytest.mark.usefixtures("fake_mesh")
def test_head_matches_hf() -> None:
    hf_mixer = randomized(Qwen4ExpTextGatedResidual(make_hf_config(), use_combine=False))
    head = load_gated_residual_head(
        make_gated_residual_config().init_head(make_initializer(), HIDDEN_SIZE),
        float_weights(hf_mixer),
        ParameterPath(),
    )
    hyper_input = make_hyper_input()
    embeddings = hyper_input[..., :HIDDEN_SIZE]

    with torch.no_grad():
        reference = hf_mixer(hyper_input)

    assert_close(
        result=merge_streams(head.expand(torch_to_jax(embeddings))),
        reference=torch_to_jax(embeddings.repeat(1, 1, STREAM_COUNT)),
    )
    assert_close(
        result=head.readout(
            as_stream(hyper_input),
            MatmulConfig.for_tracer_tests(),
            keychain=make_keychain(),
        ),
        reference=torch_to_jax(reference),
    )


@pytest.mark.usefixtures("fake_mesh")
def test_decoder_layer_matches_hf() -> None:
    hf_config = make_hf_config()
    hf_layer = randomized(Qwen4ExpTextDecoderLayer(hf_config, layer_idx=0))
    layer = load_decoder_layer(hf_config, float_weights(hf_layer))
    hyper_input = make_hyper_input()

    with torch.no_grad():
        reference = hf_layer(hyper_input, position_embeddings=None)

    result = layer(
        as_stream(hyper_input),
        positional_embeddings=None,
        forward_pass_config=TransformerForwardPassConfig.for_tracer_tests(),
        keychain=make_keychain(),
    )

    assert_close(result=merge_streams(result.outputs), reference=torch_to_jax(reference))


def test_gated_decoder_is_self_consistent() -> None:
    decoder = build_tiny_attention_decoder((None, None), residual_config=make_gated_residual_config())
    batch_size, sequence_length, prefix_length, num_suffix_tokens = 2, 10, 6, 2
    token_ids = jax.random.randint(
        jax.random.key(0),
        (batch_size, sequence_length),
        0,
        decoder.vocab_size,
        dtype=jnp.int32,
    )
    token_positions = jnp.broadcast_to(jnp.arange(sequence_length, dtype=jnp.int32), (batch_size, sequence_length))
    forward_pass_config = DecoderForwardPassConfig(
        embedding_forward_pass_config=EmbeddingForwardPassConfig(activation_dtype=jnp.float32),
    )

    full_result = decoder(
        token_ids,
        token_positions,
        forward_pass_config=forward_pass_config,
        keychain=make_keychain(),
    )
    prefix_result = decoder(
        token_ids[:, :prefix_length],
        token_positions[:, :prefix_length],
        state=decoder.init_static_state(batch_size, 16, jnp.float32),
        return_updated_state=True,
        forward_pass_config=forward_pass_config,
        keychain=make_keychain(),
    )
    continuation_result = decoder(
        token_ids[:, prefix_length:],
        token_positions[:, prefix_length:],
        state=prefix_result.updated_state,
        forward_pass_config=forward_pass_config,
        keychain=make_keychain(),
    )
    suffix_result = decoder(
        token_ids,
        token_positions,
        forward_pass_config=forward_pass_config,
        return_suffix_tokens=num_suffix_tokens,
        keychain=make_keychain(),
    )

    assert_close(result=continuation_result.logits, reference=full_result.logits[:, prefix_length:])
    assert_close(result=suffix_result.logits, reference=full_result.logits[:, -num_suffix_tokens:])
