from dataclasses import dataclass, replace
from enum import Enum

import jax
import jax.numpy as jnp
import pytest
from einops import rearrange
from hypothesis import given, settings
from hypothesis import strategies as st
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from lalamo.inference.continuous_batching import (
    CompletedRequest,
    ContinuousBatchingConfig,
    ContinuousBatchingEngine,
    TokenizedRequest,
    _decode_step,
    _DecodeReadOnly,
)
from lalamo.inference.paged_state import init_paged_state, insert_prefill
from lalamo.initializer import RandomInitializer
from lalamo.models import GenerationConfig, GenerationResults, LanguageModel, LanguageModelConfig
from lalamo.models.chat_codec import ChatCodecConfig
from lalamo.models.language_model import _top_logits_with_remainder
from lalamo.module import ForwardPassMode, Keychain
from lalamo.modules import (
    Decoder,
    DecoderForwardPassConfig,
    DeltaNetConfig,
    LinearConfig,
    Mamba2Config,
    NormalizationConfig,
    ShortConvConfig,
    SiLU,
    State,
    TokenMixerConfig,
)
from lalamo.modules.token_mixers.convolutions import SeparableCausalConvConfig
from lalamo.modules.token_mixers.kv_cache import PagedKVCacheLayer, StaticKVCacheLayer
from lalamo.modules.token_mixers.ssm_state import SSMStateLayer
from lalamo.sampling import SamplingPolicy
from tests.common import assert_close
from tests.helpers import build_tiny_attention_decoder, make_test_sharding_config


class _StatefulMixer(Enum):
    DELTANET = "deltanet-attention"
    MAMBA = "mamba-attention"
    SHORT_CONV = "short-conv-attention"


def _stateful_mixer_config(kind: _StatefulMixer, norm_config: NormalizationConfig) -> TokenMixerConfig:
    match kind:
        case _StatefulMixer.DELTANET:
            return DeltaNetConfig(
                in_proj_config=LinearConfig(),
                conv_config=SeparableCausalConvConfig(has_biases=False),
                out_proj_config=LinearConfig(),
                norm_config=norm_config,
                num_heads=2,
                num_groups=2,
                head_dim=2,
                value_head_dim=2,
                kernel_size=3,
            )
        case _StatefulMixer.MAMBA:
            return Mamba2Config(
                in_projection_config=LinearConfig(),
                out_projection_config=LinearConfig(),
                conv_config=SeparableCausalConvConfig(has_biases=False),
                activation=SiLU(),
                kernel_size=3,
                num_heads=2,
                num_groups=1,
                head_dim=4,
                state_dim=3,
                has_in_biases=False,
                has_out_biases=False,
            )
        case _StatefulMixer.SHORT_CONV:
            return ShortConvConfig(
                in_projection_config=LinearConfig(),
                conv_config=SeparableCausalConvConfig(has_biases=False),
                out_projection_config=LinearConfig(),
                kernel_size=3,
            )


def _hybrid_decoder(stateful_mixer: _StatefulMixer = _StatefulMixer.DELTANET) -> Decoder:
    attention_decoder = build_tiny_attention_decoder((None,))
    (attention_layer,) = attention_decoder.config.transformer_config.layer_configs
    assert attention_layer.pre_mixer_norm_config is not None
    delta_layer = replace(
        attention_layer,
        mixer_config=_stateful_mixer_config(stateful_mixer, attention_layer.pre_mixer_norm_config),
        rope_config=None,
    )
    decoder_config = replace(
        attention_decoder.config,
        transformer_config=replace(
            attention_decoder.config.transformer_config,
            layer_configs=(delta_layer, attention_layer),
        ),
    )
    return decoder_config.init(
        RandomInitializer(
            default_dtype=jnp.bfloat16,
            sharding_config=attention_decoder.sharding_config,
            key=jax.random.key(40),
        )
    )


@pytest.fixture(
    scope="module",
    params=tuple(_StatefulMixer),
    ids=lambda kind: kind.value,
)
def hybrid_language_model(request: pytest.FixtureRequest) -> LanguageModel:
    return _language_model(_hybrid_decoder(request.param))


def _language_model(decoder: Decoder) -> LanguageModel:
    codec_config = ChatCodecConfig(
        prompt_template="",
        output_parser_regex=None,
        system_role_name="system",
        user_role_name="user",
        assistant_role_name="assistant",
        eos_token=None,
        bos_token=None,
        end_of_thinking_tag=None,
    )
    config = LanguageModelConfig(
        token_codec_config=codec_config,
        decoder_config=decoder.config,
        generation_config=GenerationConfig(stop_token_ids=(), temperature=0.0),
    )
    tokenizer = Tokenizer(
        WordLevel(vocab={f"token_{token_id}": token_id for token_id in range(32)}, unk_token="token_0")
    )
    return LanguageModel(
        config=config,
        sharding_config=decoder.sharding_config,
        token_codec=codec_config.init(tokenizer),
        decoder=decoder,
    )


def test_mixed_paged_state_matches_dense_decode_across_slot_and_batch_changes() -> None:
    decoder = _hybrid_decoder()
    prompt_tokens = jnp.array([[1, 2, 3, 0], [4, 5, 6, 7]], dtype=jnp.int32)
    prompt_positions = jnp.broadcast_to(jnp.arange(4, dtype=jnp.int32), prompt_tokens.shape)
    prompt_lengths = jnp.array([3, 4], dtype=jnp.int32)
    initial_dense_state = decoder.init_static_state(batch_size=2, capacity=8, dtype=jnp.bfloat16)
    prefill_result = decoder(
        prompt_tokens,
        prompt_positions,
        state=initial_dense_state,
        return_updated_state=True,
        lengths_without_padding=prompt_lengths,
        return_suffix_tokens=1,
        forward_pass_config=DecoderForwardPassConfig.for_inference(),
        keychain=Keychain.init(40, sharding_config=make_test_sharding_config()),
    )
    assert prefill_result.updated_state is not None

    next_tokens = jnp.argmax(prefill_result.logits[:, 0], axis=-1).astype(jnp.int32)[:, None]
    next_positions = prompt_lengths[:, None]
    decode_config = DecoderForwardPassConfig.for_inference(ForwardPassMode.SINGLE_TOKEN)
    dense_result = decoder(
        next_tokens,
        next_positions,
        state=prefill_result.updated_state,
        return_updated_state=True,
        forward_pass_config=decode_config,
        keychain=Keychain.init(41, sharding_config=make_test_sharding_config()),
    )
    assert dense_result.updated_state is not None

    slot_ids = jnp.array([3, 1], dtype=jnp.int32)
    block_tables = jnp.array([[5, 1, 6], [2, 7, 0]], dtype=jnp.int32)
    paged_state = insert_prefill(
        init_paged_state(decoder, slot_count=4, total_pages=8, page_size=2, dtype=jnp.bfloat16),
        prefill_result.updated_state,
        slot_ids,
        block_tables,
    )
    _, paged_attention_pool = paged_state
    assert isinstance(paged_attention_pool, PagedKVCacheLayer)
    assert paged_attention_pool.block_tables.shape == (0, 0)
    assert paged_attention_pool.lengths.shape == (0,)
    last_logits = jnp.zeros((4, decoder.vocab_size), dtype=jnp.float32).at[slot_ids].set(prefill_result.logits[:, 0])
    paged_result = _decode_step(
        _DecodeReadOnly(
            decoder=decoder,
            slot_ids=slot_ids,
            block_tables=block_tables,
            lengths=prompt_lengths,
            sampling_policy=SamplingPolicy.init(temperature=0.0).broadcast(2),
            sampling_keys=jax.random.split(jax.random.key(0), 2),
        ),
        paged_state,
        last_logits,
        num_top_logits=4,
    )

    assert jnp.array_equal(paged_result.token_ids, next_tokens[:, 0])
    assert_close(
        result=paged_result.last_logits[slot_ids],
        reference=dense_result.logits[:, 0],
        operation_name="hybrid paged logits",
    )
    with jax.set_mesh(decoder.sharding_config.mesh):
        expected_top_ids, expected_top_logits, expected_remainder = _top_logits_with_remainder(
            prefill_result.logits[:, 0],
            4,
        )
    assert paged_result.top_k_token_ids is not None
    assert paged_result.top_k_token_logits is not None
    assert paged_result.remainder_logits is not None
    assert jnp.array_equal(paged_result.top_k_token_ids, expected_top_ids)
    assert_close(result=paged_result.top_k_token_logits, reference=expected_top_logits)
    assert_close(result=paged_result.remainder_logits, reference=expected_remainder)
    paged_ssm, paged_attention = paged_result.state
    dense_ssm, dense_attention = dense_result.updated_state
    assert isinstance(paged_ssm, SSMStateLayer)
    assert isinstance(dense_ssm, SSMStateLayer)
    assert isinstance(paged_attention, PagedKVCacheLayer)
    assert isinstance(dense_attention, StaticKVCacheLayer)
    assert paged_attention.block_tables.shape == (0, 0)
    assert paged_attention.lengths.shape == (0,)
    assert_close(
        result=paged_ssm.conv_state[slot_ids],
        reference=dense_ssm.conv_state,
        operation_name="paged conv state",
    )
    assert_close(
        result=paged_ssm.ssm_state[slot_ids],
        reference=dense_ssm.ssm_state,
        operation_name="paged SSM state",
    )
    assert_close(
        result=rearrange(
            paged_attention.keys[:, block_tables, :, :],
            "groups batch pages page_size channels -> batch (pages page_size) groups channels",
        ),
        reference=dense_attention.keys[:, :6],
        operation_name="paged keys",
    )
    assert_close(
        result=rearrange(
            paged_attention.values[:, block_tables, :, :],
            "groups batch pages page_size channels -> batch (pages page_size) groups channels",
        ),
        reference=dense_attention.values[:, :6],
        operation_name="paged values",
    )

    dense_second_state = State(jax.tree.map(lambda array: array[1:2], layer) for layer in dense_result.updated_state)
    second_token = jnp.argmax(dense_result.logits[1:2, 0], axis=-1).astype(jnp.int32)[:, None]
    second_position = jnp.array([[5]], dtype=jnp.int32)
    dense_second_result = decoder(
        second_token,
        second_position,
        state=dense_second_state,
        return_updated_state=True,
        forward_pass_config=decode_config,
        keychain=Keychain.init(42, sharding_config=make_test_sharding_config()),
    )
    paged_second_result = _decode_step(
        _DecodeReadOnly(
            decoder=decoder,
            slot_ids=slot_ids[1:],
            block_tables=block_tables[1:],
            lengths=second_position[:, 0],
            sampling_policy=SamplingPolicy.init(temperature=0.0).broadcast(1),
            sampling_keys=jax.random.split(jax.random.key(0), 1),
        ),
        paged_result.state,
        paged_result.last_logits,
        num_top_logits=None,
    )

    assert jnp.array_equal(paged_second_result.token_ids, second_token[:, 0])
    assert_close(
        result=paged_second_result.last_logits[slot_ids[1:]],
        reference=dense_second_result.logits[:, 0],
        operation_name="changed-membership paged logits",
    )


@dataclass(frozen=True)
class _EngineScenario:
    config: ContinuousBatchingConfig
    requests: tuple[TokenizedRequest, ...]
    arrival_steps: tuple[int, ...]


@st.composite
def _engine_scenarios(draw: st.DrawFn) -> _EngineScenario:
    request_count = draw(st.integers(min_value=1, max_value=4))
    page_size = draw(st.sampled_from((1, 2, 4)))
    prompt_lengths = draw(
        st.lists(st.integers(min_value=1, max_value=8), min_size=request_count, max_size=request_count)
    )
    output_lengths = draw(
        st.lists(st.integers(min_value=1, max_value=4), min_size=request_count, max_size=request_count)
    )
    prompts = tuple(
        tuple(
            draw(
                st.lists(
                    st.integers(min_value=1, max_value=31),
                    min_size=prompt_length,
                    max_size=prompt_length,
                )
            )
        )
        for prompt_length in prompt_lengths
    )
    num_top_logits = draw(st.lists(st.sampled_from((None, 1, 4)), min_size=request_count, max_size=request_count))
    generation_configs = draw(
        st.lists(
            st.sampled_from(
                (
                    GenerationConfig(),
                    GenerationConfig(temperature=0.0),
                    GenerationConfig(temperature=0.0, presence_penalty=0.2),
                    GenerationConfig(temperature=0.7),
                    GenerationConfig(temperature=0.8, top_k=4),
                    GenerationConfig(temperature=0.8, presence_penalty=0.2),
                )
            ),
            min_size=request_count,
            max_size=request_count,
        )
    )
    page_counts = tuple(
        (prompt_length + output_length + page_size - 1) // page_size
        for prompt_length, output_length in zip(prompt_lengths, output_lengths, strict=True)
    )
    largest_request_page_count = max(page_counts)
    total_pages = draw(st.integers(min_value=largest_request_page_count, max_value=sum(page_counts)))
    slot_count = draw(st.integers(min_value=1, max_value=request_count))
    requests = tuple(
        TokenizedRequest(
            request_id=f"request_{request_index}",
            sequence_id=f"request_{request_index}",
            prompt_token_ids=prompt,
            max_output_length=output_length,
            generation_config=generation_config,
            seed=request_index + 10,
            num_top_logits=request_num_top_logits,
        )
        for request_index, (prompt, output_length, generation_config, request_num_top_logits) in enumerate(
            zip(prompts, output_lengths, generation_configs, num_top_logits, strict=True)
        )
    )
    later_arrivals = draw(
        st.lists(st.integers(min_value=0, max_value=6), min_size=request_count - 1, max_size=request_count - 1)
    )
    return _EngineScenario(
        config=ContinuousBatchingConfig(
            page_size=page_size,
            total_pages=total_pages,
            slot_count=slot_count,
            max_decode_batch_size=draw(st.integers(min_value=1, max_value=slot_count)),
            prefill_batch_size=draw(st.integers(min_value=1, max_value=slot_count)),
            prefill_chunk_size=512,
            decode_steps_per_prefill=draw(st.integers(min_value=1, max_value=3)),
            decode_block_size=draw(st.integers(min_value=1, max_value=4)),
        ),
        requests=requests,
        arrival_steps=(0, *later_arrivals),
    )


def _run_scenario(model: LanguageModel, scenario: _EngineScenario) -> dict[str, CompletedRequest]:
    engine = ContinuousBatchingEngine(model, scenario.config)
    completed: dict[str, CompletedRequest] = {}
    for step in range(256):
        for request, arrival_step in zip(scenario.requests, scenario.arrival_steps, strict=True):
            if arrival_step == step:
                engine.submit(request)
        engine.step()
        while result := engine.pop_completed():
            completed[result.request_id] = result
        if len(completed) == len(scenario.requests):
            break
    return completed


def _assert_matches_generation(
    actual: CompletedRequest,
    reference: GenerationResults,
    request_id: str,
) -> None:
    assert actual.output_token_ids == tuple(reference.token_ids[0].tolist())
    if reference.top_k_token_ids is None:
        assert actual.logits is None
        return
    assert actual.logits is not None
    assert reference.top_k_token_logits is not None
    assert reference.remainder_logits is not None
    assert jnp.array_equal(jnp.asarray([step.top_token_ids for step in actual.logits]), reference.top_k_token_ids[0])
    assert_close(
        result=jnp.asarray([step.top_raw_logits for step in actual.logits]),
        reference=reference.top_k_token_logits[0],
        operation_name=f"{request_id} top logits",
    )
    assert_close(
        result=jnp.asarray([step.remainder_logit for step in actual.logits]),
        reference=reference.remainder_logits[0],
        operation_name=f"{request_id} remainder logits",
    )


@pytest.mark.slow
def test_continuous_engine_matches_language_model_with_shared_kv_cache() -> None:
    model = _language_model(build_tiny_attention_decoder((None, 0)))
    request = TokenizedRequest(
        request_id="shared_kv",
        sequence_id="shared_kv",
        prompt_token_ids=(1, 2, 3),
        max_output_length=3,
        generation_config=GenerationConfig(temperature=0.0),
        seed=0,
        num_top_logits=4,
    )
    scenario = _EngineScenario(
        config=ContinuousBatchingConfig(
            page_size=2,
            total_pages=4,
            slot_count=1,
            max_decode_batch_size=1,
            prefill_batch_size=1,
            prefill_chunk_size=2,
            decode_steps_per_prefill=1,
            decode_block_size=2,
        ),
        requests=(request,),
        arrival_steps=(0,),
    )
    reference = model.generate_tokens(
        jnp.asarray(request.prompt_token_ids, dtype=jnp.int32)[None, :],
        generation_config=request.generation_config,
        max_output_length=request.max_output_length,
        num_top_logits_to_return=request.num_top_logits,
        keychain=Keychain.init(request.seed, sharding_config=model.sharding_config),
    )

    actual = _run_scenario(model, scenario)[request.request_id]

    _assert_matches_generation(actual, reference, request.request_id)


@pytest.mark.slow
def test_continuous_engine_preempts_when_remaining_pages_cannot_fit_a_decode_block(
    hybrid_language_model: LanguageModel,
) -> None:
    requests = tuple(
        TokenizedRequest(
            request_id=f"request_{request_index}",
            sequence_id=f"request_{request_index}",
            prompt_token_ids=(request_index + 1,),
            max_output_length=4,
            generation_config=GenerationConfig(temperature=0.0),
            seed=request_index,
        )
        for request_index in range(2)
    )
    scenario = _EngineScenario(
        config=ContinuousBatchingConfig(
            page_size=1,
            total_pages=6,
            slot_count=2,
            max_decode_batch_size=2,
            prefill_batch_size=2,
            prefill_chunk_size=512,
            decode_steps_per_prefill=1,
            decode_block_size=4,
        ),
        requests=requests,
        arrival_steps=(0, 0),
    )

    assert set(_run_scenario(hybrid_language_model, scenario)) == {request.request_id for request in requests}


@pytest.mark.slow
@pytest.mark.filterwarnings(
    "ignore:Do not use the `random` module inside strategies:hypothesis.errors.HypothesisDeprecationWarning"
)
@settings(max_examples=40, deadline=None, derandomize=True)
@given(scenario=_engine_scenarios())
def test_continuous_engine_matches_language_model_across_scheduler_and_arrival_settings(
    hybrid_language_model: LanguageModel,
    scenario: _EngineScenario,
) -> None:
    completed = _run_scenario(hybrid_language_model, scenario)

    assert set(completed) == {request.request_id for request in scenario.requests}
    for request in scenario.requests:
        reference = hybrid_language_model.generate_tokens(
            jnp.asarray(request.prompt_token_ids, dtype=jnp.int32)[None, :],
            generation_config=request.generation_config,
            max_output_length=request.max_output_length,
            num_top_logits_to_return=request.num_top_logits,
            keychain=Keychain.init(request.seed, sharding_config=hybrid_language_model.sharding_config),
        )
        _assert_matches_generation(completed[request.request_id], reference, request.request_id)


def test_continuous_engine_stops_at_eos_for_every_stateful_layer(hybrid_language_model: LanguageModel) -> None:
    prompt = (1, 2, 3)
    first_token = int(
        hybrid_language_model.generate_tokens(
            jnp.asarray(prompt, dtype=jnp.int32)[None, :],
            generation_config=GenerationConfig(temperature=0.0),
            max_output_length=1,
            keychain=Keychain.init(10, sharding_config=hybrid_language_model.sharding_config),
        ).token_ids[0, 0]
    )
    model = LanguageModel(
        config=replace(
            hybrid_language_model.config,
            generation_config=GenerationConfig(stop_token_ids=(first_token,), temperature=0.0),
        ),
        sharding_config=hybrid_language_model.sharding_config,
        token_codec=hybrid_language_model.token_codec,
        decoder=hybrid_language_model.decoder,
    )
    engine = ContinuousBatchingEngine(
        model,
        ContinuousBatchingConfig(
            page_size=2,
            total_pages=4,
            slot_count=1,
            max_decode_batch_size=1,
            prefill_batch_size=1,
            prefill_chunk_size=2,
            decode_steps_per_prefill=1,
        ),
    )
    engine.submit(
        TokenizedRequest(
            request_id="eos",
            sequence_id="eos",
            prompt_token_ids=prompt,
            max_output_length=4,
            generation_config=GenerationConfig(temperature=0.0),
            seed=10,
        )
    )

    assert engine.step()
    assert engine.step()
    result = engine.pop_completed()

    assert result is not None
    assert result.output_token_ids == (first_token,)
