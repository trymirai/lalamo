import jax.numpy as jnp
import pytest

from lalamo.model_registry import ModelRegistry
from lalamo.models import LanguageModel
from lalamo.models.chat_codec import UserMessage
from lalamo.models.language_model import GenerationConfig
from lalamo.module import Keychain
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
    model = load_converted_model(converted_model_dir)
    assert isinstance(model, LanguageModel)
    return model


def _generate_texts(model: LanguageModel, prompts: list[str]) -> list[str]:
    def trim_at_stop_token(token_ids: list[int]) -> list[int]:
        stop_token_ids = set(model.config.generation_config.stop_token_ids)
        response_length = next(
            (idx + 1 for idx, token_id in enumerate(token_ids) if token_id in stop_token_ids),
            len(token_ids),
        )
        return token_ids[:response_length]

    encoded_prompts = [jnp.asarray(model.token_codec.encode_request([UserMessage(prompt)])) for prompt in prompts]
    max_prompt_length = max(prompt.size for prompt in encoded_prompts)
    prompt_lengths = jnp.asarray([prompt.size for prompt in encoded_prompts], dtype=jnp.int32)
    token_ids = jnp.asarray(
        [jnp.pad(prompt, (0, max_prompt_length - prompt.size), constant_values=0) for prompt in encoded_prompts],
    )
    generated = model.generate_tokens(
        token_ids,
        generation_config=GenerationConfig(temperature=0.0),
        prompt_lengths_without_padding=prompt_lengths,
        max_output_length=64,
        keychain=Keychain.init(0),
    )
    return [
        model.token_codec.decode_response(trim_at_stop_token(response_ids.tolist())).response
        for response_ids in generated.token_ids
    ]


@pytest.mark.fast
@pytest.mark.parametrize("model_repo", MODELS)
def test_convert(convert_model: ConvertModel, model_repo: str) -> None:
    converted_model_dir = convert_model(model_repo, cached=True)
    assert (converted_model_dir / "model.safetensors").exists() or any(converted_model_dir.glob("model*.safetensors"))
    assert (converted_model_dir / "config.json").exists()
    assert (converted_model_dir / "tokenizer.json").exists()


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
        model.stream_reply_text(
            [UserMessage(CAPITAL_PROMPT)],
            generation_config=GenerationConfig(temperature=0.0),
            max_output_length=64,
            keychain=Keychain.init(1),
        ),
    )
    assert "london" in capital_output.lower(), f"Expected 'london' in {capital_output!r}"

    apples_output = "".join(
        model.stream_reply_text(
            [UserMessage(APPLES_PROMPT)],
            generation_config=GenerationConfig(temperature=0.0),
            max_output_length=64,
            keychain=Keychain.init(2),
        ),
    )
    assert "yes" in apples_output.lower(), f"Expected 'yes' in {apples_output!r}"


@pytest.mark.parametrize("model_repo", MODELS)
def test_converted_model_returns_top_logits(
    convert_model: ConvertModel,
    model_repo: str,
) -> None:
    model = _load_language_model(convert_model, model_repo)
    token_ids = jnp.asarray(model.token_codec.encode_request([UserMessage(CAPITAL_PROMPT)]), dtype=jnp.int32)[None, :]
    result = model.generate_tokens(
        token_ids,
        generation_config=GenerationConfig(temperature=0.0),
        max_output_length=8,
        num_top_logits_to_return=4,
        keychain=Keychain.init(3),
    )
    assert result.top_k_token_ids is not None
    assert result.top_k_token_logits is not None
    assert result.top_k_token_ids.shape == (1, 8, 4)
    assert result.top_k_token_logits.shape == (1, 8, 4)
