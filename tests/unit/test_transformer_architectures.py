from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import DTypeLike
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from lalamo.inference.continuous_batching import (
    ContinuousBatchingConfig,
    ContinuousBatchingEngine,
    FinishReason,
    GeneratedToken,
    SequenceFinished,
    TokenEvent,
)
from lalamo.initializer import RandomInitializer
from lalamo.model_import.loaders.binding import Binder
from lalamo.model_import.model_configs.huggingface.gemma4 import (
    Gemma4RopeParameters,
    HFGemma4Config,
    HFGemma4TextConfig,
    RopeParameters,
)
from lalamo.model_import.model_configs.huggingface.qwen3_next import HFQwen3NextConfig
from lalamo.models import GenerationConfig, LanguageModel
from lalamo.models.chat_codec import ChatCodecConfig
from lalamo.models.language_model import LanguageModelConfig
from lalamo.module import ForwardPassMode, Keychain, ShardingConfig
from lalamo.modules.decoder import DecoderConfig
from lalamo.modules.flow import SequenceConfig, validate_program
from lalamo.modules.forward_pass import ModelForwardPassConfig
from lalamo.safetensors import safe_read
from tests.helpers import build_tiny_attention_decoder_config, tiny_attention_config


def gemma4_text_config(*, moe: bool = False, ple_dim: int = 4) -> HFGemma4TextConfig:
    return HFGemma4TextConfig(
        hidden_size=8,
        intermediate_size=16,
        model_type="gemma4_text",
        num_hidden_layers=4,
        sliding_window=4,
        rms_norm_eps=1e-6,
        attention_bias=False,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_global_key_value_heads=1,
        head_dim=4,
        global_head_dim=4,
        max_position_embeddings=32,
        rope_parameters=Gemma4RopeParameters(
            full_attention=RopeParameters(rope_theta=10000.0, rope_type="proportional", partial_rotary_factor=0.5),
            sliding_attention=RopeParameters(rope_theta=100.0, rope_type="default"),
        ),
        final_logit_softcapping=30.0,
        vocab_size=16,
        layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        hidden_activation="gelu_pytorch_tanh",
        hidden_size_per_layer_input=ple_dim,
        vocab_size_per_layer_input=16,
        num_kv_shared_layers=2,
        use_double_wide_mlp=True,
        tie_word_embeddings=True,
        attention_k_eq_v=moe,
        enable_moe_block=moe,
        num_experts=4 if moe else None,
        top_k_experts=2 if moe else None,
        moe_intermediate_size=12 if moe else None,
    )


def gemma4_config() -> DecoderConfig:
    return gemma4_text_config().build_decoder(Binder())


def qwen_next_config() -> DecoderConfig:
    return HFQwen3NextConfig(
        torch_dtype="float32",
        architectures=["Qwen3NextForCausalLM"],
        attention_dropout=0.0,
        bos_token_id=0,
        decoder_sparse_step=1,
        eos_token_id=15,
        full_attention_interval=2,
        head_dim=4,
        hidden_act="silu",
        hidden_size=8,
        initializer_range=0.02,
        intermediate_size=16,
        linear_conv_kernel_dim=2,
        linear_key_head_dim=4,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_value_head_dim=4,
        max_position_embeddings=32,
        mlp_only_layers=[0],
        model_type="qwen3_next",
        moe_intermediate_size=12,
        norm_topk_prob=True,
        num_attention_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=3,
        num_key_value_heads=1,
        output_router_logits=False,
        partial_rotary_factor=0.5,
        rms_norm_eps=1e-6,
        rope_scaling=None,
        rope_theta=10000.0,
        router_aux_loss_coef=0.001,
        shared_expert_intermediate_size=12,
        tie_word_embeddings=True,
        transformers_version="5.5.0",
        use_cache=True,
        use_sliding_window=False,
        vocab_size=16,
    ).to_decoder_config()


@pytest.mark.parametrize("backend", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
def test_gemma4_full_width_suffix_keeps_padded_ple_inputs_aligned(backend: str) -> None:
    sharding = ShardingConfig.replicated(jax.devices(backend)[:1])
    with (
        jax.default_device(jax.devices(backend)[0]),
        jax.set_mesh(sharding.mesh),
        jax.default_matmul_precision("highest"),
    ):
        decoder = gemma4_config().init(RandomInitializer(jnp.float32, sharding, key=jax.random.key(0)))
        tokens = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=jnp.int32)
        positions = jnp.broadcast_to(jnp.arange(4, dtype=jnp.int32), tokens.shape)
        lengths = jnp.array([4, 3], dtype=jnp.int32)
        forward = ModelForwardPassConfig.for_tracer_tests()
        keychain = Keychain.init(0, sharding_config=sharding)
        state = decoder.init_static_state(2, 8, jnp.float32)
        full = decoder(
            tokens, positions, state, lengths_without_padding=lengths, forward_pass_config=forward, keychain=keychain
        )
        suffix = decoder(
            tokens,
            positions,
            state,
            lengths_without_padding=lengths,
            return_suffix_tokens=tokens.shape[1],
            forward_pass_config=forward,
            keychain=keychain,
        )
        for row, length in enumerate(lengths.tolist()):
            np.testing.assert_allclose(suffix.logits[row, -length:], full.logits[row, :length], atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize(
    "make_config", [gemma4_config, qwen_next_config], ids=["gemma4_shared_kv_ple", "qwen_next_hybrid_moe"]
)
@pytest.mark.parametrize("backend", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
def test_architecture_cached_and_suffix_outputs_match_full(
    make_config: Callable[[], DecoderConfig],
    backend: str,
) -> None:
    config = DecoderConfig.from_json(make_config().to_json())
    sharding = ShardingConfig.replicated(jax.devices(backend)[:8])
    with (
        jax.default_device(jax.devices(backend)[0]),
        jax.set_mesh(sharding.mesh),
        jax.default_matmul_precision("highest"),
    ):
        decoder = config.init(RandomInitializer(jnp.float32, sharding, key=jax.random.key(0)))
        token_ids = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=jnp.int32)
        positions = jnp.broadcast_to(jnp.arange(4, dtype=jnp.int32), (2, 4))
        forward = ModelForwardPassConfig.for_tracer_tests()
        keychain = Keychain.init(0, sharding_config=sharding)
        initial_state = decoder.init_static_state(2, 8, jnp.float32)
        full = decoder(
            token_ids,
            positions,
            state=initial_state,
            return_updated_state=True,
            forward_pass_config=forward,
            keychain=keychain,
        )
        suffix = decoder(
            token_ids,
            positions,
            state=initial_state,
            return_updated_state=True,
            return_suffix_tokens=2,
            forward_pass_config=forward,
            keychain=keychain,
        )
        np.testing.assert_allclose(suffix.logits, full.logits[:, -2:], rtol=3e-5, atol=3e-5)
        state = initial_state
        chunks = []
        for start in (0, 2):
            chunk = decoder(
                token_ids[:, start : start + 2],
                positions[:, start : start + 2],
                state=state,
                return_updated_state=True,
                forward_pass_config=forward,
                keychain=keychain,
            )
            chunks.append(chunk.logits)
            assert chunk.updated_state is not None
            state = chunk.updated_state
        np.testing.assert_allclose(jnp.concatenate(chunks, axis=1), full.logits, rtol=3e-5, atol=3e-5)
        for full_leaf, suffix_leaf, chunk_leaf in zip(
            jax.tree.leaves(full.updated_state),
            jax.tree.leaves(suffix.updated_state),
            jax.tree.leaves(state),
            strict=True,
        ):
            np.testing.assert_allclose(suffix_leaf, full_leaf, rtol=3e-5, atol=3e-5)
            np.testing.assert_allclose(chunk_leaf, full_leaf, rtol=3e-5, atol=3e-5)

        lengths = jnp.array([4, 3], dtype=jnp.int32)
        padded_full = decoder(
            token_ids,
            positions,
            state=initial_state,
            return_updated_state=True,
            lengths_without_padding=lengths,
            forward_pass_config=forward,
            keychain=keychain,
        )
        padded_suffix = decoder(
            token_ids,
            positions,
            state=initial_state,
            return_updated_state=True,
            return_suffix_tokens=2,
            lengths_without_padding=lengths,
            forward_pass_config=forward,
            keychain=keychain,
        )
        for row, length in enumerate(lengths.tolist()):
            np.testing.assert_allclose(
                padded_suffix.logits[row], padded_full.logits[row, length - 2 : length], rtol=3e-5, atol=3e-5
            )
        for suffix_leaf, full_leaf in zip(
            jax.tree.leaves(padded_suffix.updated_state), jax.tree.leaves(padded_full.updated_state), strict=True
        ):
            np.testing.assert_allclose(suffix_leaf, full_leaf, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("backend", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("moe", [False, True], ids=["shared_kv_ple", "parallel_moe"])
def test_continuous_batching_attention_requires_gpu_and_matches_dense(backend: str, moe: bool) -> None:
    sharding = ShardingConfig.replicated(jax.devices(backend)[:1])
    with (
        jax.default_device(jax.devices(backend)[0]),
        jax.set_mesh(sharding.mesh),
        jax.default_matmul_precision("highest"),
    ):
        ple_dim = 4
        if moe:
            ple_dim = 0
        config = LanguageModelConfig(
            token_codec_config=ChatCodecConfig("", None, "system", "user", "assistant", None, None),
            decoder_config=gemma4_text_config(moe=moe, ple_dim=ple_dim).build_decoder(Binder()),
            generation_config=GenerationConfig(),
        )
        tokenizer = Tokenizer(
            WordLevel({"[UNK]": 0, **{f"t{index}": index for index in range(1, 16)}}, unk_token="[UNK]")
        )
        model = config.init(tokenizer, RandomInitializer(jnp.bfloat16, sharding, key=jax.random.key(0)))
        batching = ContinuousBatchingConfig(
            total_pages=4, slot_count=2, max_context_length=32, prefill_batch_size=2, prefill_chunk_size=4
        )
        if backend == "cpu":
            with pytest.raises(ValueError, match="Paged batching requires causal GPU attention"):
                ContinuousBatchingEngine.init(model, batching)
            return
        engine = ContinuousBatchingEngine.init(model, batching)
        generation = GenerationConfig(temperature=0.0)
        prompts = ((1, 2, 3), (3, 2))
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


@pytest.mark.parametrize("backend", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("moe", [False, True], ids=["dense_ple", "parallel_moe"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["f32", "bf16"])
def test_gemma4_layer_paths_match_transformers(backend: str, moe: bool, dtype: DTypeLike, tmp_path: Path) -> None:
    import torch  # noqa: PLC0415
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig  # noqa: PLC0415

    text_config = replace(
        gemma4_text_config(moe=moe, ple_dim=0 if moe else 4),
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        head_dim=8,
        global_head_dim=8,
        moe_intermediate_size=32,
    )
    if moe:
        text_config = replace(
            text_config,
            num_kv_shared_layers=0,
            use_double_wide_mlp=False,
            num_key_value_heads=2,
            num_global_key_value_heads=1,
            global_head_dim=16,
            rope_parameters=Gemma4RopeParameters(
                full_attention=RopeParameters(
                    rope_theta=10000.0, rope_type="proportional", partial_rotary_factor=0.25
                ),
                sliding_attention=text_config.rope_parameters.sliding_attention,
            ),
        )
    if dtype == jnp.float32:
        text_config = replace(text_config, rms_norm_eps=1e-3)
    foreign = HFGemma4Config(text_config, "float32", "gemma4", [15])
    config = LanguageModelConfig(
        token_codec_config=ChatCodecConfig("", None, "system", "user", "assistant", None, None),
        decoder_config=text_config.build_decoder(Binder()),
        generation_config=GenerationConfig(),
    )
    assert DecoderConfig.from_json(config.decoder_config.to_json()).to_json() == config.decoder_config.to_json()
    torch.manual_seed(7)
    reference_config = Gemma4TextConfig(**asdict(text_config))
    reference = Gemma4ForCausalLM(reference_config).eval()
    reference.set_attn_implementation("eager")
    reference.set_experts_implementation("eager")
    torch_dtype = torch.float32 if dtype == jnp.float32 else torch.bfloat16
    reference = reference.to(torch_dtype)
    with torch.no_grad():
        if dtype == jnp.float32:
            for name, parameter in reference.named_parameters():
                if name.endswith("norm.weight"):
                    parameter.uniform_(0.5, 1.5)
        for layer in reference.model.layers:
            if moe:
                layer.router.scale.uniform_(0.5, 1.5)
                layer.router.per_expert_scale.uniform_(0.25, 1.75)
            layer.layer_scalar.fill_(0.9)
    reference.save_pretrained(tmp_path)
    tokenizer = Tokenizer(WordLevel({f"t{index}": index for index in range(16)}, unk_token="t0"))
    sharding = ShardingConfig.replicated(jax.devices(backend)[:1])
    with (
        jax.default_device(jax.devices(backend)[0]),
        jax.set_mesh(sharding.mesh),
        jax.default_matmul_precision("highest"),
        (tmp_path / "model.safetensors").open("rb") as weights_file,
    ):
        _, weights, _ = safe_read(weights_file)
        model = foreign.load(
            config,
            tokenizer,
            dtype,
            weights,
            sharding_config=sharding,
        )
        assert isinstance(model, LanguageModel)
        model.save(tmp_path / "native")
        model = LanguageModel.load(tmp_path / "native", sharding)
        tokens = jnp.array([[1, 2, 3, 4], [3, 4, 5, 6]], dtype=jnp.int32)
        positions = jnp.broadcast_to(jnp.arange(4, dtype=jnp.int32), tokens.shape)
        keychain = Keychain.init(0, sharding_config=sharding)
        forward = ModelForwardPassConfig.for_tracer_tests()
        forward = replace(
            forward,
            embedding_forward_pass_config=replace(forward.embedding_forward_pass_config, activation_dtype=dtype),
        )
        with torch.no_grad():
            expected = reference(torch.tensor(np.asarray(tokens)), use_cache=False).logits.float().numpy()
        actual = model.decoder(
            tokens, positions, return_updated_state=True, forward_pass_config=forward, keychain=keychain
        )
        tolerance = 3e-5 if dtype == jnp.float32 else 0.02
        np.testing.assert_allclose(actual.logits, expected, atol=tolerance, rtol=tolerance)
        single_token = replace(
            forward,
            flow_forward_pass_config=replace(
                forward.flow_forward_pass_config,
                mlp_forward_pass_config=replace(
                    forward.flow_forward_pass_config.mlp_forward_pass_config,
                    mode=ForwardPassMode.SINGLE_TOKEN,
                ),
            ),
        )
        decoded = model.decoder(tokens, positions, forward_pass_config=single_token, keychain=keychain)
        np.testing.assert_allclose(decoded.logits, actual.logits, atol=tolerance, rtol=tolerance)
        uncached = model.decoder(tokens, positions, forward_pass_config=forward, keychain=keychain)
        np.testing.assert_allclose(uncached.logits, expected, atol=tolerance, rtol=tolerance)
        state = model.decoder.init_static_state(2, 8, dtype)
        chunks = []
        for start, end in ((0, 2), (2, 4)):
            result = model.decoder(
                tokens[:, start:end],
                positions[:, start:end],
                state,
                return_updated_state=True,
                forward_pass_config=forward,
                keychain=keychain,
            )
            state = result.updated_state
            chunks.append(result.logits)
        np.testing.assert_allclose(jnp.concatenate(chunks, axis=1), expected, atol=tolerance, rtol=tolerance)
        suffix = model.decoder(
            tokens, positions, return_suffix_tokens=1, forward_pass_config=forward, keychain=keychain
        )
        np.testing.assert_allclose(suffix.logits, expected[:, -1:], atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("sources", [(1, None), (None, 2, None), (None, 0, 1)])
def test_attention_kv_sources_require_earlier_cache_owner(sources: tuple[int | None, ...]) -> None:
    with pytest.raises(ValueError, match="KV source"):
        build_tiny_attention_decoder_config(sources)


def test_attention_kv_source_must_be_in_program_once() -> None:
    owner = tiny_attention_config()
    for program in (
        SequenceConfig((tiny_attention_config(owner),)),
        SequenceConfig((owner, owner, tiny_attention_config(owner))),
    ):
        with pytest.raises(ValueError, match="KV source"):
            validate_program(program)
    with pytest.raises(ValueError, match="KV source"):
        replace(tiny_attention_config(owner), num_groups=1)


@pytest.mark.parametrize("variant", ["26B-A4B", "26B-A4B-it", "31B", "31B-it"])
def test_published_gemma4_configs_roundtrip(variant: str) -> None:
    foreign = HFGemma4Config.from_json(Path(__file__).parents[1] / "data" / f"gemma4_{variant}_config.json")
    config = foreign.to_decoder_config()
    assert DecoderConfig.from_json(config.to_json()).to_json() == config.to_json()
