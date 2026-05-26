import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from lalamo.commands import trace, trace_tokenization
from lalamo.initializer import RandomInitializer
from lalamo.models import GenerationConfig, LanguageModel, LanguageModelConfig
from lalamo.models.chat_codec import ChatCodecConfig, UserMessage
from lalamo.modules import (
    DecoderConfig,
    DenseMLPConfig,
    Identity,
    LinearConfig,
    NormalizationConfig,
    TiedEmbeddingConfig,
    TransformerConfig,
    TransformerLayerConfig,
    UnscaledRoPEConfig,
    UpcastMode,
)
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.safetensors import safe_read, safe_write
from lalamo.tracer import (
    export_trace_result,
    load_traceable_model,
    load_traceable_token_codec,
    record_message_trace,
    record_tokenization_trace,
    trace_sharding_config,
)
from lalamo.utils.json import JSON
from tests.conftest import RunLalamo


def _language_model() -> LanguageModel:
    model_dim = 8
    context_length = 16
    norm_config = NormalizationConfig(
        epsilon=1e-5,
        scale_offset=None,
        upcast_mode=UpcastMode.ONLY_NORMALIZATION,
        subtract_mean=False,
    )
    layer_config = TransformerLayerConfig(
        pre_mixer_norm_config=norm_config,
        mixer_config=AttentionConfig(
            qkv_projection_config=LinearConfig(),
            out_projection_config=LinearConfig(),
            query_norm_config=None,
            key_norm_config=None,
            num_heads=2,
            num_groups=2,
            head_dim=4,
            is_causal=True,
            scale=None,
            sliding_window_size=None,
            logit_soft_cap=None,
            has_sinks=False,
            has_qkv_biases=False,
            has_out_biases=False,
        ),
        post_mixer_norm_config=norm_config,
        pre_mlp_norm_config=norm_config,
        mlp_config=DenseMLPConfig(
            linear_config=LinearConfig(),
            activation=Identity(),
            has_up_biases=False,
            has_down_biases=False,
            gate_clipping=None,
            up_clipping=None,
        ),
        post_mlp_norm_config=norm_config,
        rope_config=UnscaledRoPEConfig(
            base=10_000.0,
            max_sequence_length=context_length,
            head_dim=4,
        ),
    )
    config = LanguageModelConfig(
        token_codec_config=ChatCodecConfig(
            prompt_template="{{ messages[0]['content'] }}",
            output_parser_regex=None,
            system_role_name="system",
            user_role_name="user",
            assistant_role_name="assistant",
            eos_token=None,
            bos_token=None,
        ),
        decoder_config=DecoderConfig(
            embedding_config=TiedEmbeddingConfig(
                input_scale=None,
                logit_soft_cap=None,
            ),
            transformer_config=TransformerConfig(
                layer_configs=(layer_config,),
                output_norm_config=norm_config,
                model_dim=model_dim,
                hidden_dim=16,
            ),
            vocab_size=32,
        ),
        generation_config=GenerationConfig(),
    )
    sharding_config = trace_sharding_config()
    tokenizer = Tokenizer(WordLevel(vocab={"[UNK]": 0, "trace": 7}, unk_token="[UNK]"))
    return config.init(
        tokenizer,
        RandomInitializer(
            default_dtype=jnp.float32,
            sharding_config=sharding_config,
            key=jax.random.key(4),
        ),
    )


def _assert_uzu_trace_paths(arrays: Mapping[str, jax.Array]) -> None:
    assert "activation_trace.token_ids" in arrays
    assert "activation_trace.token_positions" in arrays
    assert "activation_trace.layer_results.0.activation_trace.inputs" in arrays
    assert "activation_trace.layer_results.0.activation_trace.pre_mixer_norm" in arrays
    assert "activation_trace.layer_results.0.activation_trace.mixer" in arrays
    assert "activation_trace.layer_results.0.activation_trace.post_mixer_norm" in arrays
    assert "activation_trace.layer_results.0.activation_trace.mlp_inputs" in arrays
    assert "activation_trace.layer_results.0.activation_trace.pre_mlp_norm" in arrays
    assert "activation_trace.layer_results.0.activation_trace.mlp" in arrays
    assert "activation_trace.layer_results.0.activation_trace.post_mlp_norm" in arrays
    assert "activation_trace.layer_results.0.outputs" in arrays
    assert "activation_trace.layer_results.0.updated_state.keys" in arrays
    assert "activation_trace.layer_results.0.updated_state.values" in arrays
    assert "activation_trace.output_norm" in arrays
    assert "logits" in arrays


def _assert_tokenization_trace(metadata: dict[str, str] | None, arrays: Mapping[str, jax.Array]) -> None:
    assert metadata is not None
    assert metadata["add_special_tokens"] == "false"
    assert metadata["prompt_template"] == "{{ messages[0]['content'] }}"
    assert metadata["rendered_request"] == "trace"
    assert json.loads(metadata["request"]) == {
        "add_generation_prompt": True,
        "messages": [{"role": "user", "content": "trace"}],
        "bos_token": None,
        "eos_token": None,
        "enable_thinking": True,
    }
    assert json.loads(metadata["tokens"]) == ["trace"]
    assert arrays["activation_trace.token_ids"].tolist() == [[7]]
    assert arrays["activation_trace.token_positions"].tolist() == [[0]]


def _write_uzu_language_model(model: LanguageModel, model_dir: Path) -> None:
    model_dir.mkdir(parents=True)
    config = cast("dict[str, JSON]", model.config.to_json())
    message_config = cast("dict[str, JSON]", config["token_codec_config"])
    generation_config = cast("dict[str, JSON]", config["generation_config"])
    uzu_config = {
        "toolchain_version": "test",
        "vendor": "test",
        "family": "test",
        "name": "test",
        "size": "tiny",
        "quantization": None,
        "repo": "test",
        "use_cases": [],
        "model_type": "language_model",
        "model_config": {
            "model_config": config["decoder_config"],
            "message_processor_config": {
                "prompt_template": message_config["prompt_template"],
                "output_parser_regex": message_config["output_parser_regex"],
                "system_role_name": message_config["system_role_name"],
                "user_role_name": message_config["user_role_name"],
                "assistant_role_name": message_config["assistant_role_name"],
                "eos_token": message_config["eos_token"],
                "bos_token": message_config["bos_token"],
                "default_system_prompt": message_config["default_system_prompt"],
            },
            "generation_config": {
                "stop_token_ids": generation_config["stop_token_ids"],
                "temperature": generation_config["temperature"],
                "top_k": generation_config["top_k"],
                "top_p": generation_config["top_p"],
                "min_p": generation_config["min_p"],
                "banned_tokens": generation_config["banned_tokens"],
            },
        },
    }
    with (model_dir / "config.json").open("w") as fd:
        json.dump(uzu_config, fd, indent=4)
    model.token_codec.tokenizer.save(str(model_dir / "tokenizer.json"))

    arrays: dict[str, jax.Array] = {}
    for path, array in model.export().arrays.items():
        if path == "decoder.embedding.embedding.weights":
            arrays["embedding.weights"] = array
        elif path.startswith("decoder.transformer.ropes.0."):
            arrays["transformer.global_rope." + path.removeprefix("decoder.transformer.ropes.0.")] = array
        elif path.startswith("decoder."):
            uzu_path = path.removeprefix("decoder.")
            if uzu_path.endswith(".weights.weights"):
                uzu_path = uzu_path.removesuffix(".weights")
            arrays[uzu_path] = array

    with (model_dir / "model.safetensors").open("wb") as fd:
        safe_write(fd, arrays)


def _write_token_trace(trace_path: Path) -> None:
    _write_token_trace_arrays(
        trace_path,
        jnp.asarray([[0, 0, 0]], dtype=jnp.int32),
        jnp.asarray([[0, 1, 2]], dtype=jnp.int32),
    )


def _write_token_trace_arrays(trace_path: Path, token_ids: jax.Array, token_positions: jax.Array) -> None:
    with trace_path.open("wb") as fd:
        safe_write(
            fd,
            {
                "activation_trace.token_ids": token_ids,
                "activation_trace.token_positions": token_positions,
            },
            metadata={"source": "fixture"},
        )


def test_record_message_trace_exports_uzu_trace_paths() -> None:
    result = record_message_trace(_language_model(), [UserMessage("trace")])

    arrays = export_trace_result(result)

    _assert_uzu_trace_paths(arrays)
    assert arrays["activation_trace.token_ids"].dtype == jnp.int32
    assert arrays["activation_trace.token_ids"].shape == (1, 1)
    assert arrays["logits"].dtype == arrays["activation_trace.output_norm"].dtype


def test_record_tokenization_trace_exports_request_boundary() -> None:
    trace_result = record_tokenization_trace(_language_model().token_codec, [UserMessage("trace")])

    _assert_tokenization_trace(trace_result.metadata, trace_result.arrays)


def test_load_traceable_model_loads_uzu_language_model(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_uzu_language_model(_language_model(), model_dir)

    result = record_message_trace(load_traceable_model(model_dir), [UserMessage("trace")])

    _assert_uzu_trace_paths(export_trace_result(result))


def test_load_traceable_token_codec_loads_uzu_language_model(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_uzu_language_model(_language_model(), model_dir)

    trace_result = record_tokenization_trace(load_traceable_token_codec(model_dir), [UserMessage("trace")])

    _assert_tokenization_trace(trace_result.metadata, trace_result.arrays)


def test_trace_command_writes_safetensors(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    trace_path = tmp_path / "traces.safetensors"
    _language_model().save(model_dir)

    trace(model_dir, trace_path, [UserMessage("trace")])

    with trace_path.open("rb") as fd:
        metadata, arrays = safe_read(fd)
        _assert_uzu_trace_paths(arrays)
        assert metadata is not None
        assert metadata["rendered_request"] == "trace"


def test_trace_command_writes_safetensors_for_uzu_model(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    trace_path = tmp_path / "traces.safetensors"
    _write_uzu_language_model(_language_model(), model_dir)

    trace(model_dir, trace_path, [UserMessage("trace")])

    with trace_path.open("rb") as fd:
        metadata, arrays = safe_read(fd)
        _assert_uzu_trace_paths(arrays)
        assert metadata is not None
        assert metadata["rendered_request"] == "trace"


def test_trace_command_replays_token_trace(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    input_trace_path = tmp_path / "input-trace.safetensors"
    trace_path = tmp_path / "traces.safetensors"
    _language_model().save(model_dir)
    _write_token_trace(input_trace_path)

    trace(model_dir, trace_path, (), input_trace_path)

    with trace_path.open("rb") as fd:
        metadata, arrays = safe_read(fd)
        _assert_uzu_trace_paths(arrays)
        assert metadata == {"source": "fixture"}
        assert arrays["activation_trace.token_ids"].tolist() == [[0, 0, 0]]
        assert arrays["activation_trace.token_positions"].tolist() == [[0, 1, 2]]


def test_trace_command_rejects_bad_token_trace_shape(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    input_trace_path = tmp_path / "input-trace.safetensors"
    trace_path = tmp_path / "traces.safetensors"
    _language_model().save(model_dir)
    with input_trace_path.open("wb") as fd:
        safe_write(
            fd,
            {
                "activation_trace.token_ids": jnp.asarray([0, 0, 0], dtype=jnp.int32),
                "activation_trace.token_positions": jnp.asarray([0, 1, 2], dtype=jnp.int32),
            },
        )

    with pytest.raises(ValueError, match=r"\[1, suffix_tokens\]"):
        trace(model_dir, trace_path, (), input_trace_path)


def test_trace_command_rejects_bad_token_trace_dtype(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    input_trace_path = tmp_path / "input-trace.safetensors"
    trace_path = tmp_path / "traces.safetensors"
    _language_model().save(model_dir)
    with input_trace_path.open("wb") as fd:
        safe_write(
            fd,
            {
                "activation_trace.token_ids": jnp.asarray([[0, 0, 0]], dtype=jnp.uint32),
                "activation_trace.token_positions": jnp.asarray([[0, 1, 2]], dtype=jnp.int32),
            },
        )

    with pytest.raises(TypeError, match=r"token_ids.*int32"):
        trace(model_dir, trace_path, (), input_trace_path)


@pytest.mark.parametrize(
    ("token_ids", "token_positions", "message"),
    (
        (jnp.asarray([[-1]], dtype=jnp.int32), jnp.asarray([[0]], dtype=jnp.int32), r"token_ids.*\[0, 32\)"),
        (jnp.asarray([[32]], dtype=jnp.int32), jnp.asarray([[0]], dtype=jnp.int32), r"token_ids.*\[0, 32\)"),
        (jnp.asarray([[0]], dtype=jnp.int32), jnp.asarray([[-1]], dtype=jnp.int32), r"nonnegative"),
        (jnp.asarray([[0]], dtype=jnp.int32), jnp.asarray([[16]], dtype=jnp.int32), r"positions.*\[0, 16\)"),
    ),
)
def test_trace_command_rejects_bad_token_trace_values(
    tmp_path: Path,
    token_ids: jax.Array,
    token_positions: jax.Array,
    message: str,
) -> None:
    model_dir = tmp_path / "model"
    input_trace_path = tmp_path / "input-trace.safetensors"
    trace_path = tmp_path / "traces.safetensors"
    _language_model().save(model_dir)
    _write_token_trace_arrays(input_trace_path, token_ids, token_positions)

    with pytest.raises(ValueError, match=message):
        trace(model_dir, trace_path, (), input_trace_path)


def test_trace_command_rejects_messages_with_input_trace(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    input_trace_path = tmp_path / "input-trace.safetensors"
    trace_path = tmp_path / "traces.safetensors"
    _language_model().save(model_dir)
    _write_token_trace(input_trace_path)

    with pytest.raises(ValueError, match="messages cannot be used"):
        trace(model_dir, trace_path, [UserMessage("trace")], input_trace_path)


def test_trace_command_replays_tokenization_trace(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    tokenization_trace_path = tmp_path / "tokenization-trace.safetensors"
    trace_path = tmp_path / "traces.safetensors"
    _language_model().save(model_dir)
    trace_tokenization(model_dir, tokenization_trace_path, [UserMessage("trace")])

    trace(model_dir, trace_path, (), tokenization_trace_path)

    with trace_path.open("rb") as fd:
        metadata, arrays = safe_read(fd)
        assert metadata is not None
        assert metadata["rendered_request"] == "trace"
        assert arrays["activation_trace.token_ids"].tolist() == [[7]]
        assert arrays["activation_trace.token_positions"].tolist() == [[0]]


def test_record_tokenization_trace_rejects_empty_encoding() -> None:
    token_codec = ChatCodecConfig(
        prompt_template="",
        output_parser_regex=None,
        system_role_name="system",
        user_role_name="user",
        assistant_role_name="assistant",
        eos_token=None,
        bos_token=None,
    ).init(Tokenizer(WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]")))

    with pytest.raises(ValueError, match="at least one token"):
        record_tokenization_trace(token_codec, [UserMessage("trace")])


def test_trace_tokenization_command_writes_safetensors(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    trace_path = tmp_path / "tokenization-trace.safetensors"
    _language_model().save(model_dir)

    trace_tokenization(model_dir, trace_path, [UserMessage("trace")])

    with trace_path.open("rb") as fd:
        metadata, arrays = safe_read(fd)
        _assert_tokenization_trace(metadata, arrays)


def test_trace_cli_writes_safetensors(tmp_path: Path, run_lalamo: RunLalamo) -> None:
    model_dir = tmp_path / "model"
    trace_path = tmp_path / "traces.safetensors"
    _language_model().save(model_dir)

    run_lalamo("trace", str(model_dir), "--output-path", str(trace_path), "--message", "trace")

    with trace_path.open("rb") as fd:
        _, arrays = safe_read(fd)
        _assert_uzu_trace_paths(arrays)


def test_trace_cli_replays_token_trace(tmp_path: Path, run_lalamo: RunLalamo) -> None:
    model_dir = tmp_path / "model"
    input_trace_path = tmp_path / "input-trace.safetensors"
    trace_path = tmp_path / "traces.safetensors"
    _language_model().save(model_dir)
    _write_token_trace(input_trace_path)

    run_lalamo(
        "trace",
        str(model_dir),
        "--output-path",
        str(trace_path),
        "--input-trace-path",
        str(input_trace_path),
    )

    with trace_path.open("rb") as fd:
        metadata, arrays = safe_read(fd)
        assert metadata == {"source": "fixture"}
        assert arrays["activation_trace.token_ids"].tolist() == [[0, 0, 0]]


def test_trace_tokenization_cli_writes_safetensors(tmp_path: Path, run_lalamo: RunLalamo) -> None:
    model_dir = tmp_path / "model"
    trace_path = tmp_path / "tokenization-trace.safetensors"
    _language_model().save(model_dir)

    run_lalamo("trace-tokenization", str(model_dir), "--output-path", str(trace_path), "--message", "trace")

    with trace_path.open("rb") as fd:
        metadata, arrays = safe_read(fd)
        _assert_tokenization_trace(metadata, arrays)
