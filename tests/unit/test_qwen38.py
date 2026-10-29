import asyncio
from collections.abc import Generator, Mapping
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array
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
from lalamo.initializer import EmptyInitializer
from lalamo.model_import.loaders.binding import Binder, Binding
from lalamo.model_import.loaders.huggingface import huggingface_loader, load_huggingface_decoder
from lalamo.model_import.model_configs.huggingface.qwen3_8 import HFQwen38Config
from lalamo.models import GenerationConfig
from lalamo.models.chat_codec import ChatCodecConfig
from lalamo.models.language_model import LanguageModel, LanguageModelConfig
from lalamo.module import ForwardPassMode, Keychain, ShardingConfig
from lalamo.modules.decoder import Decoder
from lalamo.modules.flow import leaves
from lalamo.modules.forward_pass import ModelForwardPassConfig
from lalamo.modules.token_mixers.qsa import QSAConfig
from tests.common import DEFAULT_ATOL, DEFAULT_RTOL
from tests.tracer.tracer import DType


def _load_decoder(
    foreign: HFQwen38Config,
    initializer: EmptyInitializer,
    weights: Mapping[str, Array],
) -> tuple[Decoder, tuple[Binding | None, ...]]:
    binder = Binder(huggingface_loader)
    config = foreign.build_decoder(binder)
    decoder = load_huggingface_decoder(config.init(initializer), config, binder, weights)
    return decoder, binder.sources(config.program)


def _load_reference_decoder(config_path: Path, initializer: EmptyInitializer) -> Decoder:
    with np.load(Path(__file__).parents[1] / "data" / "qwen38_reference.npz") as reference, jax.enable_x64():
        weights = {
            name: jnp.asarray(reference[name]) for name in reference.files if name.startswith(("model.", "lm_head."))
        }
    decoder, _ = _load_decoder(HFQwen38Config.from_json(config_path), initializer, weights)
    return decoder


@pytest.mark.parametrize("layer_ids", [[0], [5], [4]])
def test_qwen38_rejects_invalid_ngram_layer_ids(layer_ids: list[int], tmp_path: Path) -> None:
    with np.load(Path(__file__).parents[1] / "data" / "qwen38_reference.npz") as reference:
        path = tmp_path / "config.json"
        path.write_text(str(reference["config"]))
    config = HFQwen38Config.from_json(path)
    with pytest.raises(ValueError, match="one-based linear-attention"):
        replace(config.text_config, ple_layer_ids=layer_ids)


@pytest.fixture(params=["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
def qwen38_decoder(request: pytest.FixtureRequest, tmp_path: Path) -> Generator[Decoder, None, None]:
    # Released Transformers equations, including pruning (17 tokens, a 4-token QSA budget).
    with np.load(Path(__file__).parents[1] / "data" / "qwen38_reference.npz") as reference:
        config_path = tmp_path / "config.json"
        config_path.write_text(str(reference["config"]))
        sharding = ShardingConfig.replicated(jax.devices(request.param)[:1])
        with (
            jax.default_device(jax.devices(request.param)[0]),
            jax.set_mesh(sharding.mesh),
            jax.default_matmul_precision("highest"),
        ):
            yield _load_reference_decoder(config_path, EmptyInitializer(jnp.float32, sharding))


@pytest.mark.parametrize("state_capacity", [pytest.param(None, id="dynamic"), pytest.param(32, id="static")])
def test_qwen38_checkpoint_logits_cached_padding_and_suffix(
    qwen38_decoder: Decoder, state_capacity: int | None
) -> None:
    decoder = qwen38_decoder
    sharding = decoder.sharding_config
    with np.load(Path(__file__).parents[1] / "data" / "qwen38_reference.npz") as reference:
        tokens = jnp.tile(jnp.asarray(reference["tokens"]), (2, 1))
        positions = jnp.tile(jnp.asarray(reference["positions"]), (2, 1))
        lengths = jnp.array([17, 11])
        keychain = Keychain.init(0, sharding_config=sharding)
        forward = ModelForwardPassConfig.for_tracer_tests()
        full = decoder(
            tokens, positions, lengths_without_padding=lengths, forward_pass_config=forward, keychain=keychain
        )
        np.testing.assert_allclose(full.logits[:1], reference["logits"], atol=3e-5, rtol=3e-4)
        np.testing.assert_allclose(full.logits[1, :11], reference["logits"][0, :11], atol=3e-5, rtol=3e-4)
        blank_state = None
        if state_capacity is not None:
            blank_state = decoder.init_static_state(2, state_capacity, jnp.float32)
        state = blank_state
        pieces = []
        for start, end in ((0, 1), (1, 4), (4, 7), (7, 17)):
            result = decoder(
                tokens[:, start:end],
                positions[:, start:end],
                state,
                return_updated_state=True,
                lengths_without_padding=jnp.clip(lengths - start, 0, end - start),
                forward_pass_config=forward,
                keychain=keychain,
            )
            state = result.updated_state
            pieces.append(result.logits)
        for row, length in enumerate(lengths.tolist()):
            np.testing.assert_allclose(
                jnp.concatenate(pieces, axis=1)[row, :length], full.logits[row, :length], atol=3e-5, rtol=3e-4
            )
        suffix = decoder(
            tokens,
            positions,
            blank_state,
            return_updated_state=True,
            lengths_without_padding=lengths,
            return_suffix_tokens=2,
            forward_pass_config=forward,
            keychain=keychain,
        )
        for row, length in enumerate(lengths.tolist()):
            np.testing.assert_allclose(suffix.logits[row], full.logits[row, length - 2 : length], atol=3e-5, rtol=3e-4)
        next_tokens = jnp.array([[18], [12]])
        cached = decoder(next_tokens, lengths[:, None], state, forward_pass_config=forward, keychain=keychain)
        suffix_cached = decoder(
            next_tokens, lengths[:, None], suffix.updated_state, forward_pass_config=forward, keychain=keychain
        )
        dense_tokens = jnp.pad(tokens, ((0, 0), (0, 1))).at[jnp.arange(2), lengths].set(next_tokens[:, 0])
        dense = decoder(
            dense_tokens,
            jnp.tile(jnp.arange(18), (2, 1)),
            lengths_without_padding=lengths + 1,
            forward_pass_config=forward,
            keychain=keychain,
        )
        np.testing.assert_allclose(cached.logits, suffix_cached.logits, atol=3e-5, rtol=3e-4)
        np.testing.assert_allclose(cached.logits[:, 0], dense.logits[jnp.arange(2), lengths], atol=3e-5, rtol=3e-4)


@pytest.fixture
def qwen38_model(qwen38_decoder: Decoder, tmp_path: Path) -> LanguageModel:
    sharding = qwen38_decoder.sharding_config
    # Match the BF16 CLI import; the independent decoder fixture retains the FP32 oracle.
    decoder = _load_reference_decoder(tmp_path / "config.json", EmptyInitializer(jnp.bfloat16, sharding))
    codec = ChatCodecConfig("{{ messages[0]['content'] }}", None, "system", "user", "assistant", None, None)
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, **{f"t{index}": index for index in range(1, 32)}}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.decoder = Fuse()
    config = LanguageModelConfig(
        token_codec_config=codec, decoder_config=decoder.config, generation_config=GenerationConfig()
    )
    return LanguageModel(config=config, sharding_config=sharding, token_codec=codec.init(tokenizer), decoder=decoder)


def test_qwen38_continuous_batching_prefill_decode_and_prefix(qwen38_model: LanguageModel) -> None:
    model = qwen38_model
    sharding = model.sharding_config
    batching = ContinuousBatchingConfig(total_pages=8, slot_count=2, prefill_batch_size=2, prefill_chunk_size=4)
    if sharding.mesh.devices.flat[0].platform == "cpu":
        with pytest.raises(ValueError, match="Paged batching requires causal GPU attention"):
            ContinuousBatchingEngine.init(model, batching)
        return
    engine = ContinuousBatchingEngine.init(model, batching)
    assert engine.context_limit == 64
    generation = GenerationConfig(temperature=0.0)
    first_prompt = (1, 2, 31, *(4 + index % 27 for index in range(38)))
    cancelled_events: list[TokenEvent] = []
    cancel = engine.submit(first_prompt, 4, generation, 0, on_events=cancelled_events.extend)
    assert engine.step()
    cancel.set()
    assert not engine.step()
    assert not cancelled_events
    cached_prefix = None
    # Reusing a completed prefix exercises both token/conv state and paged indexer history.
    for round_index, prompts in enumerate(
        ((first_prompt, first_prompt[:31]), ((*first_prompt, 18, 19), first_prompt[:35]))
    ):
        events: list[list[TokenEvent]] = [[], []]
        for prompt, received in zip(prompts, events, strict=True):
            engine.submit(prompt, 4, generation, 0, return_logprobs=True, on_events=received.extend)
        assert engine.step()
        if cached_prefix is not None:
            # Prefix hits have no event; check cached-page ownership after the public scheduler step.
            assert all(prefix is not cached_prefix for prefix in engine._prefix_cache)  # noqa: SLF001
            (continued,) = [sequence for sequence in engine._active if sequence.prompt_token_ids == prompts[0]]  # noqa: SLF001
            assert continued.pages[: len(cached_prefix.pages)] == cached_prefix.pages
        while engine.step():
            pass
        for prompt, received in zip(prompts, events, strict=True):
            generated = [event for event in received if isinstance(event, GeneratedToken)]
            assert len(generated) == 4
            keychain = Keychain.init(0, sharding_config=sharding)
            forward = ModelForwardPassConfig.for_inference()
            result = model.decoder(
                jnp.asarray([prompt], dtype=jnp.int32),
                jnp.arange(len(prompt), dtype=jnp.int32)[None],
                model.decoder.init_static_state(
                    1, engine.context_limit, forward.embedding_forward_pass_config.activation_dtype
                ),
                return_updated_state=True,
                forward_pass_config=forward,
                keychain=keychain,
            )
            for step, event in enumerate(generated):
                assert event.logprobs is not None
                probabilities = np.asarray(jax.nn.softmax(result.logits[0, -1].astype(jnp.float32)))
                np.testing.assert_allclose(
                    np.exp(event.logprobs.top_logprobs),
                    probabilities[list(event.logprobs.top_token_ids)],
                    atol=0,
                    rtol=DEFAULT_RTOL,
                )
                np.testing.assert_allclose(
                    np.exp(event.logprobs.logprob), probabilities[event.token_id], atol=0, rtol=DEFAULT_RTOL
                )
                assert probabilities[event.token_id] >= probabilities.max() * (1 - DEFAULT_RTOL)
                if step + 1 < len(generated):
                    result = model.decoder(
                        jnp.asarray([[event.token_id]], dtype=jnp.int32),
                        jnp.asarray([[len(prompt) + step]], dtype=jnp.int32),
                        result.updated_state,
                        return_updated_state=True,
                        forward_pass_config=ModelForwardPassConfig.for_inference(ForwardPassMode.SINGLE_TOKEN),
                        keychain=keychain,
                    )
            assert received[-1] == SequenceFinished(FinishReason.LENGTH, 4)
        if round_index == 0:
            (cached_prefix,) = [
                prefix
                for prefix in engine._prefix_cache  # noqa: SLF001
                if prefix.token_ids == first_prompt[: -batching.uncached_prompt_tail]
            ]


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("qwen38_decoder", ["gpu"], indirect=True)
def test_qwen38_native_checkpoint_openai_service(qwen38_model: LanguageModel, tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    import httpx2  # noqa: PLC0415
    from openai import AsyncOpenAI  # noqa: PLC0415

    from lalamo.server import create_app  # noqa: PLC0415

    model = qwen38_model
    generation = GenerationConfig(temperature=0.0)
    prompts = ((1, 2, 31, *range(4, 18)), (1, 2, 31, *range(4, 12)))
    expected_texts = []
    for prompt in prompts:
        tokens = model.stream_tokens(
            jnp.array(prompt, dtype=jnp.int32),
            generation,
            4,
            keychain=Keychain.init(0, sharding_config=model.sharding_config),
        )
        expected_texts.append(model.token_codec.decode_tokens([int(token) for token in tokens]))
    directory = tmp_path / "qwen38"
    model.save(directory)
    restored = LanguageModel.load(directory, model.sharding_config)

    async def run() -> None:
        api = create_app(
            restored,
            "qwen38",
            ContinuousBatchingConfig(
                total_pages=4,
                slot_count=2,
                prefill_batch_size=2,
                prefill_chunk_size=4,
            ),
        )
        async with (
            api.router.lifespan_context(api),
            httpx2.AsyncClient(transport=httpx2.ASGITransport(app=api)) as http,
        ):
            client = AsyncOpenAI(api_key="test", base_url="http://test/v1", http_client=http)
            replies = await asyncio.gather(
                *(
                    client.chat.completions.create(
                        model="qwen38",
                        messages=[{"role": "user", "content": " ".join(f"t{token}" for token in prompt)}],
                        max_completion_tokens=4,
                        temperature=0,
                    )
                    for prompt in prompts
                )
            )
            for reply, prompt, expected in zip(replies, prompts, expected_texts, strict=True):
                (choice,) = reply.choices
                assert choice.message.content == expected
                assert choice.finish_reason == "length"
                assert reply.usage is not None
                assert reply.usage.prompt_tokens == len(prompt)
                assert reply.usage.completion_tokens == 4

    asyncio.run(run())


def test_qwen38_qsa_rejects_unsupported_attention_semantics(qwen38_decoder: Decoder) -> None:
    *_, config = (step for step in leaves(qwen38_decoder.config.program) if isinstance(step, QSAConfig))
    for attention in (
        replace(config.attention_config, scale=2.0),
        replace(config.attention_config, logit_soft_cap=30.0),
        replace(config.attention_config, sliding_window_size=4),
    ):
        with pytest.raises(ValueError, match="standard scaled attention"):
            replace(config, attention_config=attention)
    with pytest.raises(ValueError, match="whole selected-block budget"):
        replace(config, token_budget=5)
    with pytest.raises(ValueError, match="rotary width"):
        replace(config, rope_config=replace(config.rope_config, head_dim=8))


@pytest.mark.parametrize("qwen38_decoder", ["cpu"], indirect=True)
@pytest.mark.parametrize("resave_config", [False, True], ids=["published", "resaved"])
@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [(DType.FLOAT32, 3e-5, 3e-4), (DType.BFLOAT16, DEFAULT_ATOL, DEFAULT_RTOL)],
)
def test_qwen38_from_pretrained_trace(
    qwen38_decoder: Decoder,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dtype: DType,
    resave_config: bool,
    atol: float,
    rtol: float,
) -> None:
    import torch  # noqa: PLC0415
    from safetensors.torch import save_file  # noqa: PLC0415
    from transformers import AutoConfig  # noqa: PLC0415

    from tests.common import tolerance  # noqa: PLC0415
    from tests.tracer.tracer_huggingface import HFQwen38Tracer  # noqa: PLC0415

    directory = tmp_path / "published-checkpoint"
    directory.mkdir()
    with np.load(Path(__file__).parents[1] / "data" / "qwen38_reference.npz") as reference:
        (directory / "config.json").write_text(str(reference["config"]))
        if resave_config:
            AutoConfig.from_pretrained(directory).save_pretrained(directory)
        weights: dict[str, torch.Tensor] = {}
        for name in reference.files:
            if not name.startswith(("model.", "lm_head.")):
                continue
            tensor = torch.from_numpy(reference[name].copy())
            if tensor.is_floating_point():
                tensor = tensor.to(dtype.torch_dtype)
            weights[name] = tensor
        save_file(weights, directory / "model.safetensors", metadata={"format": "pt"})
        monkeypatch.setenv("LALAMO_HF_DEVICE_MAP", "cpu")
        tracer = HFQwen38Tracer.load(str(directory), dtype)
        from lalamo.utils.torch_interop import torch_to_jax  # noqa: PLC0415

        with jax.enable_x64():
            native_weights = {name: torch_to_jax(tensor) for name, tensor in weights.items()}
        foreign = HFQwen38Config.from_json(directory / "config.json")
        assert foreign.to_decoder_config().to_json() == qwen38_decoder.config.to_json()
        decoder, sources = _load_decoder(
            foreign,
            EmptyInitializer(dtype.jax_dtype, qwen38_decoder.sharding_config),
            native_weights,
        )
        forward = ModelForwardPassConfig.for_tracer_tests()
        forward = replace(
            forward,
            embedding_forward_pass_config=replace(
                forward.embedding_forward_pass_config,
                activation_dtype=dtype.jax_dtype,
            ),
        )
        result = decoder(
            jnp.asarray(reference["tokens"]),
            jnp.asarray(reference["positions"]),
            return_activation_trace=True,
            forward_pass_config=forward,
            keychain=Keychain.init(0, sharding_config=qwen38_decoder.sharding_config),
        )
        with tolerance(atol=atol, rtol=rtol):
            tracer.match_activations(result, decoder.program, sources)
