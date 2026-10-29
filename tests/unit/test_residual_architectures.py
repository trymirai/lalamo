import asyncio
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import Enum
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Float
from tokenizers import Tokenizer
from tokenizers.decoders import Fuse
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from lalamo.inference.continuous_batching import (
    ContinuousBatchingConfig,
    ContinuousBatchingEngine,
    FinishReason,
    GeneratedToken,
    SequenceFinished,
    TokenEvent,
)
from lalamo.initializer import RandomInitializer
from lalamo.models import GenerationConfig, LanguageModel
from lalamo.models.chat_codec import ChatCodecConfig
from lalamo.models.language_model import LanguageModelConfig
from lalamo.module import Keychain
from lalamo.modules.decoder import Decoder, DecoderConfig
from lalamo.modules.embedding import TiedEmbeddingConfig
from lalamo.modules.flow import (
    Add,
    Context,
    EmitConfig,
    Expand,
    FlowForwardPassConfig,
    SequenceConfig,
    SinkConfig,
    StepConfig,
    SumStreams,
    leaves,
    walk,
)
from lalamo.modules.forward_pass import ModelForwardPassConfig
from lalamo.modules.mlp import DenseMLP
from lalamo.modules.normalization import Normalization
from lalamo.modules.residual_connections import (
    AttentionResidual,
    AttentionResidualConfig,
    HyperConnection,
    HyperConnectionConfig,
    HyperConnectionDynamicsConfig,
    ManifoldHyperConnection,
    ManifoldHyperConnectionConfig,
    routed,
)
from lalamo.modules.token_mixers.attention import Attention
from lalamo.utils.sharding import ShardingConfig
from tests.helpers import tiny_attention_config, tiny_mlp_config, tiny_norm_config

STREAMS = 3
NUM_LAYERS = 3
TOLERANCE = 3e-5


class ResidualVariant(Enum):
    STATIC_HC = "static_hc"
    DYNAMIC_HC = "dynamic_hc"
    MANIFOLD_HC = "manifold_hc"
    FULL_ATTNRES = "full_attnres"
    BLOCK_ATTNRES = "block_attnres"

    @property
    def attention_residual_block_size(self) -> int | None:
        """Sublayers per Attention Residuals block; the full variant is the one-sublayer block."""
        match self:
            case ResidualVariant.FULL_ATTNRES:
                return 1
            case ResidualVariant.BLOCK_ATTNRES:
                return 4
            case _:
                return None


def _sublayers() -> list[tuple[StepConfig, StepConfig]]:
    return [
        (tiny_norm_config(), body) for _ in range(NUM_LAYERS) for body in (tiny_attention_config(), tiny_mlp_config())
    ]


def _hyper_connection_program(variant: ResidualVariant) -> list[StepConfig]:
    def connection(index: int) -> StepConfig:
        match variant:
            case ResidualVariant.STATIC_HC:
                return HyperConnectionConfig(STREAMS, index % STREAMS, dynamics=None)
            case ResidualVariant.DYNAMIC_HC:
                return HyperConnectionConfig(
                    STREAMS, index % STREAMS, HyperConnectionDynamicsConfig(tiny_norm_config())
                )
            case _:
                # Few enough Sinkhorn iterations that the projection has not converged, which pins its exact numerics.
                return ManifoldHyperConnectionConfig(STREAMS, sinkhorn_iterations=3)

    return [
        Expand(STREAMS),
        *(routed(connection(index), *body) for index, body in enumerate(_sublayers())),
        SumStreams(STREAMS),
        tiny_norm_config(),
    ]


def _attention_residual_program(block_size: int) -> list[StepConfig]:
    """Paper Attention Residuals: every sublayer reads a softmax over completed blocks and the current partial sum."""

    def reader() -> AttentionResidualConfig:
        return AttentionResidualConfig(tiny_norm_config())

    blocks: tuple[EmitConfig, ...] = ()
    steps: list[StepConfig] = []
    for index, body in enumerate(_sublayers()):
        if index % block_size == 0:
            # The stream holds the completed block, which seeds the next block's partial sum.
            block = EmitConfig(f"block.{len(blocks)}")
            steps += [block, SinkConfig(blocks, reader()), *body]
            blocks += (block,)
        else:
            contribution = EmitConfig("contribution", SequenceConfig((SinkConfig(blocks, reader()), *body)))
            steps += [contribution, SinkConfig((contribution,), Add())]
    return [*steps, SinkConfig(blocks, reader()), tiny_norm_config()]


def residual_decoder_config(variant: ResidualVariant) -> DecoderConfig:
    block_size = variant.attention_residual_block_size
    if block_size is None:
        steps = _hyper_connection_program(variant)
    else:
        steps = _attention_residual_program(block_size)
    return DecoderConfig(
        embedding_config=TiedEmbeddingConfig(input_scale=None, logit_soft_cap=None),
        program=SequenceConfig(tuple(steps)),
        model_dim=8,
        vocab_size=32,
    )


def _rms_norm(norm: Normalization, inputs: Float[Array, "*axes channels"]) -> Float[Array, "*axes channels"]:
    assert norm.scales is not None
    variance = jnp.mean(jnp.square(inputs), axis=-1, keepdims=True)
    return inputs * jax.lax.rsqrt(variance + norm.config.epsilon) * norm.scales


def _hyper_connection_reference(
    connection: HyperConnection | ManifoldHyperConnection,
    hidden: Float[Array, "batch tokens streams channels"],
    sublayer: Callable[[Array], Array],
) -> Float[Array, "batch tokens streams channels"]:
    """One sublayer of (m)HC: `H' = A_r^T H + B^T sublayer(A_m^T H)`."""
    if isinstance(connection, HyperConnection):
        read = jnp.broadcast_to(connection.static_alpha[:, 0], hidden.shape[:-1])
        mixing = jnp.broadcast_to(connection.static_alpha[:, 1:], (*hidden.shape[:-1], STREAMS))
        write = jnp.broadcast_to(connection.static_beta, hidden.shape[:-1])
        if connection.dynamics is not None:
            dynamics = connection.dynamics
            normalized = _rms_norm(dynamics.norm, hidden)
            alpha = dynamics.alpha_scale * jnp.tanh(normalized @ dynamics.alpha_projection.T)
            read = read + alpha[..., 0]
            mixing = mixing + alpha[..., 1:]
            write = write + dynamics.beta_scale * jnp.tanh(normalized @ dynamics.beta_projection)
    else:
        flat = hidden.reshape(*hidden.shape[:-2], -1)
        flat = flat / jnp.sqrt(jnp.mean(jnp.square(flat), axis=-1, keepdims=True) + connection.config.norm_epsilon)
        epsilon = connection.config.epsilon
        read = jax.nn.sigmoid(flat @ connection.pre_projection.T * connection.pre_scale + connection.pre_bias)
        read = read + epsilon
        write = 2 * jax.nn.sigmoid(flat @ connection.post_projection.T * connection.post_scale + connection.post_bias)
        logits = jnp.einsum("...w,stw->...st", flat, connection.residual_projection)
        mixing = jax.nn.softmax(logits * connection.residual_scale + connection.residual_bias, axis=-1) + epsilon
        mixing = mixing / (mixing.sum(axis=-2, keepdims=True) + epsilon)
        for _ in range(connection.config.sinkhorn_iterations - 1):
            mixing = mixing / (mixing.sum(axis=-1, keepdims=True) + epsilon)
            mixing = mixing / (mixing.sum(axis=-2, keepdims=True) + epsilon)
    contribution = sublayer(jnp.einsum("...s,...sc->...c", read, hidden))
    return jnp.einsum("...st,...sc->...tc", mixing, hidden) + write[..., None] * contribution[..., None, :]


def _attention_residual_reference(
    reader: AttentionResidual,
    current: Float[Array, "batch tokens channels"],
    previous: list[Float[Array, "batch tokens channels"]],
) -> Float[Array, "batch tokens channels"]:
    values = jnp.stack((*previous, current))
    weights = jax.nn.softmax(_rms_norm(reader.key_norm, values) @ reader.query, axis=0)
    return jnp.sum(weights[..., None] * values, axis=0)


def reference_outputs(
    decoder: Decoder,
    variant: ResidualVariant,
    embeddings: Float[Array, "batch tokens channels"],
    token_positions: Array,
) -> Float[Array, "batch tokens channels"]:
    """A per-sublayer loop over the decoder's parameters, written from the papers rather than the program."""
    ctx = Context(
        None,
        token_positions,
        None,
        None,
        FlowForwardPassConfig.for_tracer_tests(),
        decoder.sharding_config,
    )
    keychain = Keychain.init(0, sharding_config=decoder.sharding_config)
    primitives = leaves(decoder.program)
    *sublayer_norms, final_norm = (step for step in primitives if isinstance(step, Normalization))
    mixers = [step for step in primitives if isinstance(step, Attention)]
    mlps = [step for step in primitives if isinstance(step, DenseMLP)]
    connections = [
        step for step in primitives if isinstance(step, HyperConnection | ManifoldHyperConnection | AttentionResidual)
    ]

    def sublayer(index: int, inputs: Array) -> Array:
        normalized = sublayer_norms[index].forward(inputs, ctx=ctx, keychain=keychain)
        if index % 2 == 0:
            return mixers[index // 2].forward(normalized, state=None, ctx=ctx, keychain=keychain).outputs
        return mlps[index // 2].forward(normalized, ctx=ctx, keychain=keychain)

    block_size = variant.attention_residual_block_size
    if block_size is None:
        hidden = jnp.repeat(embeddings[..., None, :], STREAMS, axis=-2)
        for index, connection in enumerate(connections):
            assert isinstance(connection, HyperConnection | ManifoldHyperConnection)
            hidden = _hyper_connection_reference(connection, hidden, lambda inputs: sublayer(index, inputs))  # noqa: B023
        outputs = hidden.sum(axis=-2)
    else:
        *readers, final_reader = connections
        blocks = [embeddings]
        partial = None
        for index, reader in enumerate(readers):
            assert isinstance(reader, AttentionResidual)
            if index % block_size == 0 and partial is not None:
                blocks, partial = [*blocks, partial], None
            if partial is None:
                *previous, current = blocks
                partial = sublayer(index, _attention_residual_reference(reader, current, previous))
            else:
                partial = partial + sublayer(index, _attention_residual_reference(reader, partial, blocks))
        assert isinstance(final_reader, AttentionResidual) and partial is not None
        outputs = _attention_residual_reference(final_reader, partial, blocks)
    return final_norm.forward(outputs, ctx=ctx, keychain=keychain)


def _perturbed[ModuleT: eqx.Module](module: ModuleT, key: Array) -> ModuleT:
    """Moves every parameter off its initialization, so identity-like paper inits cannot hide wiring bugs."""
    parameters, static = eqx.partition(module, eqx.is_inexact_array)
    flat, treedef = jax.tree.flatten(parameters)
    flat = [
        leaf + 0.3 * jax.random.normal(leaf_key, leaf.shape, leaf.dtype)
        for leaf, leaf_key in zip(flat, jax.random.split(key, len(flat)), strict=True)
    ]
    return eqx.combine(jax.tree.unflatten(treedef, flat), static)


@contextmanager
def _on_backend(backend: str) -> Iterator[ShardingConfig]:
    sharding = ShardingConfig.replicated(jax.devices(backend)[:1])
    with (
        jax.default_device(jax.devices(backend)[0]),
        jax.set_mesh(sharding.mesh),
        jax.default_matmul_precision("highest"),
    ):
        yield sharding


def _residual_model(variant: ResidualVariant, sharding: ShardingConfig) -> LanguageModel:
    codec = ChatCodecConfig("{{ messages[0]['content'] }}", None, "system", "user", "assistant", None, None)
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, **{f"t{index}": index for index in range(1, 32)}}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.decoder = Fuse()
    config = LanguageModelConfig(
        token_codec_config=codec,
        decoder_config=residual_decoder_config(variant),
        generation_config=GenerationConfig(),
    )
    model = config.init(tokenizer, RandomInitializer(jnp.float32, sharding, key=jax.random.key(0)))
    return eqx.tree_at(
        lambda language_model: language_model.decoder, model, _perturbed(model.decoder, jax.random.key(1))
    )


VARIANTS = pytest.mark.parametrize(
    "variant", list(ResidualVariant), ids=[variant.value for variant in ResidualVariant]
)


@VARIANTS
def test_residual_decoder_matches_reference_cache_suffix_padding_and_tree(variant: ResidualVariant) -> None:
    with _on_backend("cpu") as sharding:
        decoder = _residual_model(variant, sharding).decoder
        tokens = jnp.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], dtype=jnp.int32)
        positions = jnp.broadcast_to(jnp.arange(6, dtype=jnp.int32), tokens.shape)
        keychain = Keychain.init(0, sharding_config=sharding)
        forward = ModelForwardPassConfig.for_tracer_tests()

        def assert_close(actual: Array, expected: Array) -> None:
            np.testing.assert_allclose(actual, expected, atol=TOLERANCE, rtol=TOLERANCE)

        full = decoder(tokens, positions, return_activation_trace=True, forward_pass_config=forward, keychain=keychain)
        trace = full.activation_trace
        assert trace is not None
        assert_close(trace.outputs, reference_outputs(decoder, variant, trace.embeddings, positions))

        state = decoder.init_static_state(2, 8, jnp.float32)
        chunks = []
        for start, end in ((0, 1), (1, 4), (4, 6)):
            chunk = decoder(
                tokens[:, start:end],
                positions[:, start:end],
                state,
                return_updated_state=True,
                forward_pass_config=forward,
                keychain=keychain,
            )
            assert chunk.updated_state is not None
            state = chunk.updated_state
            chunks.append(chunk.logits)
        assert_close(jnp.concatenate(chunks, axis=1), full.logits)

        for suffix_length in (1, 3):
            suffix = decoder(
                tokens, positions, return_suffix_tokens=suffix_length, forward_pass_config=forward, keychain=keychain
            )
            assert_close(suffix.logits, full.logits[:, -suffix_length:])

        lengths = jnp.array([6, 4], dtype=jnp.int32)
        padded_suffix = decoder(
            tokens,
            positions,
            decoder.init_static_state(2, 8, jnp.float32),
            lengths_without_padding=lengths,
            return_suffix_tokens=2,
            forward_pass_config=forward,
            keychain=keychain,
        )
        for row, length in enumerate(lengths.tolist()):
            unpadded = decoder(
                tokens[row : row + 1, :length],
                positions[row : row + 1, :length],
                forward_pass_config=forward,
                keychain=keychain,
            )
            assert_close(padded_suffix.logits[row], unpadded.logits[0, -2:])

        prefix = decoder(
            tokens[:, :3], positions[:, :3], return_updated_state=True, forward_pass_config=forward, keychain=keychain
        )
        assert prefix.updated_state is not None
        tree = decoder(
            tokens[:, 3:],
            positions[:, 3:],
            prefix.updated_state,
            attention_parent_indices=jnp.broadcast_to(jnp.array([-1, 0, 1], dtype=jnp.int32), (2, 3)),
            forward_pass_config=forward,
            keychain=keychain,
        )
        assert_close(tree.logits, full.logits[:, 3:])


@VARIANTS
def test_residual_model_native_roundtrip_preserves_graph_and_logits(variant: ResidualVariant, tmp_path: Path) -> None:
    with _on_backend("cpu") as sharding:
        model = _residual_model(variant, sharding)
        model.save(tmp_path)
        restored = LanguageModel.load(tmp_path, sharding)
        assert restored.decoder.config.to_json() == model.decoder.config.to_json()
        # Shared emitters come back as shared objects, so every sink still reads the emission it was wired to.
        restored_emitters = {
            id(step) for step in walk(restored.decoder.config.program) if isinstance(step, EmitConfig)
        }
        restored_sources = {
            id(source)
            for step in walk(restored.decoder.config.program)
            if isinstance(step, SinkConfig)
            for source in step.sources
        }
        assert restored_sources <= restored_emitters
        tokens = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
        positions = jnp.arange(4, dtype=jnp.int32)[None]
        keychain = Keychain.init(0, sharding_config=sharding)
        forward = ModelForwardPassConfig.for_tracer_tests()
        np.testing.assert_array_equal(
            restored.decoder(tokens, positions, forward_pass_config=forward, keychain=keychain).logits,
            model.decoder(tokens, positions, forward_pass_config=forward, keychain=keychain).logits,
        )


@VARIANTS
@pytest.mark.parametrize("backend", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
def test_residual_model_continuous_batching_matches_dense(variant: ResidualVariant, backend: str) -> None:
    with _on_backend(backend) as sharding:
        model = _residual_model(variant, sharding)
        batching = ContinuousBatchingConfig(
            total_pages=4, slot_count=2, max_context_length=32, prefill_batch_size=2, prefill_chunk_size=4
        )
        if backend == "cpu":
            with pytest.raises(ValueError, match="Paged batching requires causal GPU attention"):
                ContinuousBatchingEngine.init(model, batching)
            return
        engine = ContinuousBatchingEngine.init(model, batching)
        generation = GenerationConfig(temperature=0.0)
        prompts = ((1, 2, 3, 4, 5), (3, 2))
        budgets = (3, 4)
        events: list[list[TokenEvent]] = [[], []]
        for prompt, budget, received in zip(prompts, budgets, events, strict=True):
            engine.submit(prompt, budget, generation, 0, on_events=received.extend)
        while engine.step():
            pass
        for prompt, budget, received in zip(prompts, budgets, events, strict=True):
            expected = model.stream_tokens(
                jnp.array(prompt, dtype=jnp.int32),
                generation,
                budget,
                keychain=Keychain.init(0, sharding_config=sharding),
            )
            assert [event.token_id for event in received if isinstance(event, GeneratedToken)] == [
                int(token) for token in expected
            ]
            assert received[-1] == SequenceFinished(FinishReason.LENGTH, budget)


@pytest.mark.gpu
@pytest.mark.slow
@VARIANTS
def test_residual_model_native_checkpoint_openai_service(variant: ResidualVariant, tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    import httpx2  # noqa: PLC0415
    from openai import AsyncOpenAI  # noqa: PLC0415

    from lalamo.server import create_app  # noqa: PLC0415

    with _on_backend("gpu") as sharding:
        model = _residual_model(variant, sharding)
        generation = GenerationConfig(temperature=0.0)
        prompts = ((1, 2, 3, 4, 5, 6), (7, 8, 9))
        expected_texts = [
            model.token_codec.decode_tokens(
                [
                    int(token)
                    for token in model.stream_tokens(
                        jnp.array(prompt, dtype=jnp.int32),
                        generation,
                        4,
                        keychain=Keychain.init(0, sharding_config=sharding),
                    )
                ]
            )
            for prompt in prompts
        ]
        model.save(tmp_path / variant.value)
        restored = LanguageModel.load(tmp_path / variant.value, sharding)

        async def run() -> None:
            api = create_app(
                restored,
                variant.value,
                ContinuousBatchingConfig(total_pages=4, slot_count=2, prefill_batch_size=2, prefill_chunk_size=4),
            )
            async with (
                api.router.lifespan_context(api),
                httpx2.AsyncClient(transport=httpx2.ASGITransport(app=api)) as http,
            ):
                client = AsyncOpenAI(api_key="test", base_url="http://test/v1", http_client=http)
                replies = await asyncio.gather(
                    *(
                        client.chat.completions.create(
                            model=variant.value,
                            messages=[{"role": "user", "content": " ".join(f"t{token}" for token in prompt)}],
                            max_completion_tokens=4,
                            temperature=0,
                        )
                        for prompt in prompts
                    )
                )
                for reply, expected in zip(replies, expected_texts, strict=True):
                    (choice,) = reply.choices
                    assert choice.message.content == expected
                    assert choice.finish_reason == "length"

        asyncio.run(run())
