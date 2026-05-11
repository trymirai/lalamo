from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jaxtyping import Array
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from lalamo.checkpoint_manager import CheckpointManager
from lalamo.initializer import Initializer
from lalamo.model import Model, ModelConfig
from lalamo.module import ShardingAxis
from lalamo.token_codec import TokenCodec, TokenCodecConfig
from lalamo.utils.sharding import make_sharding


@dataclass(frozen=True)
class ExampleBaseTokenCodecConfig(TokenCodecConfig):
    pass


@dataclass(frozen=True)
class ExampleTokenCodecConfig(ExampleBaseTokenCodecConfig):
    def init(self, tokenizer: Tokenizer) -> "ExampleTokenCodec":
        return ExampleTokenCodec(config=self, tokenizer=tokenizer)


@dataclass(frozen=True)
class ExampleTokenCodec(TokenCodec[list[int], list[int], ExampleTokenCodecConfig]):
    def encode_request(self, request: list[int]) -> list[int]:
        return request

    def decode_response(self, response: list[int]) -> list[int]:
        return response


@dataclass(frozen=True)
class ExampleBaseModelConfig(ModelConfig[ExampleBaseTokenCodecConfig]):
    pass


@dataclass(frozen=True)
class ExampleModelConfig(ExampleBaseModelConfig):
    token_codec_config: ExampleBaseTokenCodecConfig
    width: int

    def init(self, tokenizer: Tokenizer, initializer: Initializer) -> "ExampleModel":
        return ExampleModel(
            config=self,
            token_codec=self.token_codec_config.init(tokenizer),
            dense_weight=initializer.ones((self.width,), partition=None),
            sharded_weight=initializer.zeros(
                (self.width, self.width),
                partition=(ShardingAxis.TENSOR, None),
            ),
        )


class ExampleModel(Model[ExampleTokenCodecConfig, ExampleModelConfig, ExampleTokenCodec]):
    dense_weight: Array
    sharded_weight: Array


def _tokenizer() -> Tokenizer:
    return Tokenizer(
        WordLevel(
            vocab={
                "hello": 0,
                "world": 1,
                "[UNK]": 2,
            },
            unk_token="[UNK]",
        ),
    )


def _model(width: int = 3) -> ExampleModel:
    config = ExampleModelConfig(token_codec_config=ExampleTokenCodecConfig(), width=width)
    return ExampleModel(
        config=config,
        token_codec=config.token_codec_config.init(_tokenizer()),
        dense_weight=jnp.arange(width, dtype=jnp.float32),
        sharded_weight=jnp.arange(width * width, dtype=jnp.float32).reshape(width, width),
    )


def _single_device_tensor_mesh() -> Mesh:
    return Mesh(np.array(jax.devices()[:1]), (ShardingAxis.TENSOR,))


def _save_model(tmp_path: Path, model: ExampleModel) -> CheckpointManager:
    manager = CheckpointManager.init(tmp_path)
    manager.save("example", model)
    return manager


def test_checkpoint_restore_roundtrips_config_tokenizer_and_arrays(tmp_path: Path) -> None:
    model = _model()
    manager = _save_model(tmp_path, model)

    with jax.set_mesh(_single_device_tensor_mesh()):
        restored = manager.restore(ExampleBaseModelConfig, "example", dtype=jnp.float32)

    assert restored.config == model.config
    assert restored.token_codec.tokenizer.get_vocab() == model.token_codec.tokenizer.get_vocab()
    assert jnp.array_equal(restored.dense_weight, model.dense_weight)
    assert jnp.array_equal(restored.sharded_weight, model.sharded_weight)


def test_checkpoint_restore_casts_arrays_to_requested_dtype(tmp_path: Path) -> None:
    model = _model()
    manager = _save_model(tmp_path, model)

    with jax.set_mesh(_single_device_tensor_mesh()):
        restored = manager.restore(ExampleBaseModelConfig, "example", dtype=jnp.bfloat16)

    assert restored.dense_weight.dtype == jnp.bfloat16
    assert restored.sharded_weight.dtype == jnp.bfloat16
    assert jnp.array_equal(restored.dense_weight, model.dense_weight.astype(jnp.bfloat16))
    assert jnp.array_equal(restored.sharded_weight, model.sharded_weight.astype(jnp.bfloat16))


def test_checkpoint_restore_uses_template_sharding_not_saved_sharding(tmp_path: Path, fake_mesh: Mesh) -> None:
    model = _model(width=4)
    saved_sharding = model.sharded_weight.sharding
    manager = _save_model(tmp_path, model)

    with jax.set_mesh(fake_mesh):
        expected_sharding = make_sharding((ShardingAxis.TENSOR, None))
        restored = manager.restore(ExampleBaseModelConfig, "example", dtype=jnp.float32)

    assert restored.sharded_weight.sharding == expected_sharding
    assert restored.sharded_weight.sharding != saved_sharding
    assert jnp.array_equal(restored.sharded_weight, model.sharded_weight)
