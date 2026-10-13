import jax
import jax.numpy as jnp
import pytest

from lalamo.model_registry import ModelRegistry
from lalamo.models import LanguageModel
from lalamo.models.chat_codec import UserMessage
from lalamo.models.language_model import GenerationConfig
from lalamo.module import Keychain, LogicalAxis, ShardingConfig
from tests.conftest import ConvertModel, RunLalamo, load_converted_model, strip_ansi_escape

MODELS = ["google/gemma-3-1b-it", "mlx-community/LFM2-350M-8bit"]

CAPITAL_PROMPT = "What's the capital of the United Kingdom? No thinking, answer right away."
APPLES_PROMPT = "Are apples fruits? Answer only yes or no, without thinking, answer right away."


def _assert_has_london_and_yes(texts: list[str]) -> None:
    joined = " ".join(texts).lower()
    assert "london" in joined, f"Expected 'london' in {texts!r}"
    assert "yes" in joined, f"Expected 'yes' in {texts!r}"


def _load_language_model(convert_model: ConvertModel, model_repo: str) -> LanguageModel:
    converted_model_dir = convert_model(model_repo, cached=True)
    model = load_converted_model(converted_model_dir, ShardingConfig.replicated())
    assert isinstance(model, LanguageModel)
    return model


def _generate_texts(model: LanguageModel, prompts: list[str]) -> list[str]:
    rendered_prompts = [model.token_codec.render_request([UserMessage(prompt)]) for prompt in prompts]
    encoded_prompts = [jnp.asarray(model.token_codec.encode_text(prompt)) for prompt in rendered_prompts]
    max_prompt_length = max(prompt.size for prompt in encoded_prompts)
    prompt_lengths = jnp.asarray([prompt.size for prompt in encoded_prompts], dtype=jnp.int32)
    token_ids = jnp.asarray(
        [jnp.pad(prompt, (0, max_prompt_length - prompt.size), constant_values=0) for prompt in encoded_prompts],
    )
    prompt_lengths = jax.device_put(
        prompt_lengths,
        model.sharding_config.resolve_sharding((LogicalAxis.BATCH,)),
    )
    token_ids = jax.device_put(
        token_ids,
        model.sharding_config.resolve_sharding((LogicalAxis.BATCH, None)),
    )
    generation_config = model.config.generation_config.override_with(GenerationConfig(temperature=0.0))
    generated = model.generate_tokens(
        token_ids,
        generation_config=generation_config,
        prompt_lengths_without_padding=prompt_lengths,
        max_output_length=64,
        keychain=Keychain.init(0, sharding_config=model.sharding_config),
    )
    return [
        model.token_codec.parse_response(
            model.token_codec.decode_tokens(generation_config.trim_stop_tokens(response_ids.tolist())), prompt=prompt
        ).response
        for response_ids, prompt in zip(generated.token_ids, rendered_prompts, strict=True)
    ]


def test_list_models_plain_and_no_plain(run_lalamo: RunLalamo, model_registry: ModelRegistry) -> None:
    plain_output = strip_ansi_escape(run_lalamo("list-models", "--plain"))
    plain_repos = [line.strip() for line in plain_output.splitlines() if line.strip()]
    assert plain_repos
    assert all("/" in repo for repo in plain_repos)
    assert "│" not in plain_output

    fancy_output = strip_ansi_escape(run_lalamo("list-models", "--no-plain"))
    assert "│" in fancy_output
    fancy_repos = [repo for repo in plain_repos if repo in fancy_output]

    local_repos = set(model_registry.repo_to_model)
    assert local_repos.issubset(set(plain_repos))
    assert local_repos.issubset(set(fancy_repos))


@pytest.mark.fast
def test_chat_non_interactive(
    run_lalamo: RunLalamo,
    convert_model: ConvertModel,
) -> None:
    converted_model_dir = convert_model(MODELS[0], cached=True)
    output = strip_ansi_escape(
        run_lalamo(
            "chat",
            str(converted_model_dir),
            "--message",
            CAPITAL_PROMPT,
            "--max-tokens",
            "64",
            "--temperature",
            "0",
        ),
    )
    assert "london" in output.lower(), f"Expected 'london' in {output!r}"


@pytest.mark.fast
@pytest.mark.parametrize("model_repo", MODELS)
def test_converted_model_generates_batch(
    convert_model: ConvertModel,
    model_repo: str,
) -> None:
    model = _load_language_model(convert_model, model_repo)
    _assert_has_london_and_yes(_generate_texts(model, [CAPITAL_PROMPT, APPLES_PROMPT]))


@pytest.mark.parametrize("model_repo", MODELS)
def test_converted_model_streams_reply(
    convert_model: ConvertModel,
    model_repo: str,
) -> None:
    model = _load_language_model(convert_model, model_repo)
    capital_output = "".join(
        model.stream_text(
            model.token_codec.render_request([UserMessage(CAPITAL_PROMPT)]),
            generation_config=model.config.generation_config.override_with(GenerationConfig(temperature=0.0)),
            max_output_length=64,
            keychain=Keychain.init(1, sharding_config=model.sharding_config),
        ),
    )
    assert "london" in capital_output.lower(), f"Expected 'london' in {capital_output!r}"

    apples_output = "".join(
        model.stream_text(
            model.token_codec.render_request([UserMessage(APPLES_PROMPT)]),
            generation_config=model.config.generation_config.override_with(GenerationConfig(temperature=0.0)),
            max_output_length=64,
            keychain=Keychain.init(2, sharding_config=model.sharding_config),
        ),
    )
    assert "yes" in apples_output.lower(), f"Expected 'yes' in {apples_output!r}"
