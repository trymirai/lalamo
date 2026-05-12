from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from jax.sharding import NamedSharding
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from lalamo.initializer import Initializer
from lalamo.model import Model, ModelConfig
from lalamo.models.chat_codec import ChatCodec, ChatCodecConfig
from lalamo.module import LalamoConfig, LalamoModule, ShardingAxis
from lalamo.preconditioner import Preconditioner, PreconditionerDict
from lalamo.utils.sharding import is_sharded, make_sharding
from lalamo.weight_matrix import FullPrecisionSpec, WeightMatrix
from tests.common import assert_close


def _assert_close(result: jax.Array, reference: jax.Array) -> None:
    assert_close(result=jnp.asarray(jax.device_get(result)), reference=jnp.asarray(jax.device_get(reference)))


@dataclass(frozen=True)
class _BlockConfig(LalamoConfig):
    pass


class _Block(LalamoModule[_BlockConfig]):
    matrix: WeightMatrix


@dataclass(frozen=True)
class _PreconditionerModelConfig(ModelConfig[ChatCodecConfig]):
    def init(self, tokenizer: Tokenizer, initializer: Initializer) -> "_PreconditionerModel":
        return _PreconditionerModel(
            config=self,
            token_codec=self.token_codec_config.init(tokenizer),
            first=initializer.weight_matrix(output_dim=2, input_dim=3),
            block=_Block(
                config=_BlockConfig(),
                matrix=initializer.weight_matrix(output_dim=3, input_dim=4),
            ),
        )


class _PreconditionerModel(Model[ChatCodecConfig, _PreconditionerModelConfig, ChatCodec]):
    token_codec: ChatCodec
    first: WeightMatrix
    block: _Block


def _chat_codec_config() -> ChatCodecConfig:
    return ChatCodecConfig(
        prompt_template="",
        output_parser_regex=None,
        system_role_name="system",
        user_role_name="user",
        assistant_role_name="assistant",
        eos_token=None,
        bos_token=None,
    )


def _tokenizer() -> Tokenizer:
    return Tokenizer(WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]"))


def _model() -> _PreconditionerModel:
    config = _PreconditionerModelConfig(token_codec_config=_chat_codec_config())
    return _PreconditionerModel(
        config=config,
        token_codec=config.token_codec_config.init(_tokenizer()),
        first=FullPrecisionSpec().compress(jnp.ones((2, 3), dtype=jnp.float32)),
        block=_Block(
            config=_BlockConfig(),
            matrix=FullPrecisionSpec().compress(jnp.ones((3, 4), dtype=jnp.float32)),
        ),
    )


def test_preconditioner_preserves_symmetric_blocks() -> None:
    input_block = jnp.array(
        [
            [2.0, -1.0, 0.5],
            [-1.0, 3.0, 4.0],
            [0.5, 4.0, 5.0],
        ],
        dtype=jnp.float32,
    )
    output_block = jnp.array(
        [
            [7.0, 2.0],
            [2.0, 11.0],
        ],
        dtype=jnp.float32,
    )

    preconditioner = Preconditioner.init(input_block=input_block, output_block=output_block)

    assert preconditioner.input_block_tril is not None
    assert preconditioner.input_block_tril.shape == (6,)
    assert preconditioner.output_block_tril is not None
    assert preconditioner.output_block_tril.shape == (3,)
    assert preconditioner.input_block is not None
    assert preconditioner.output_block is not None
    _assert_close(result=preconditioner.input_block, reference=input_block)
    _assert_close(result=preconditioner.output_block, reference=output_block)
    _assert_close(
        result=jnp.asarray(preconditioner.magnitude),
        reference=jnp.trace(input_block) * jnp.trace(output_block),
    )


def test_preconditioner_identity_has_no_blocks() -> None:
    preconditioner = Preconditioner.identity()

    assert preconditioner.input_block is None
    assert preconditioner.output_block is None
    _assert_close(result=preconditioner.magnitude, reference=jnp.asarray(1.0))


def test_preconditioner_magnitude_uses_only_present_blocks() -> None:
    input_block = jnp.array(
        [
            [2.0, -1.0, 0.5],
            [-1.0, 3.0, 4.0],
            [0.5, 4.0, 5.0],
        ],
        dtype=jnp.float32,
    )
    output_block = jnp.array(
        [
            [7.0, 2.0],
            [2.0, 11.0],
        ],
        dtype=jnp.float32,
    )

    input_preconditioner = Preconditioner.init(input_block=input_block)
    output_preconditioner = Preconditioner.init(output_block=output_block)

    _assert_close(result=jnp.asarray(input_preconditioner.magnitude), reference=jnp.trace(input_block))
    _assert_close(result=jnp.asarray(output_preconditioner.magnitude), reference=jnp.trace(output_block))


def test_preconditioner_supports_batched_symmetric_blocks() -> None:
    input_block = jnp.array(
        [
            [2.0, -1.0, 0.5],
            [-1.0, 3.0, 4.0],
            [0.5, 4.0, 5.0],
        ],
        dtype=jnp.float32,
    )
    output_block = jnp.array(
        [
            [7.0, 2.0],
            [2.0, 11.0],
        ],
        dtype=jnp.float32,
    )
    input_blocks = jnp.stack([input_block, input_block + jnp.identity(3, dtype=jnp.float32)])
    output_blocks = jnp.stack([output_block, output_block + jnp.identity(2, dtype=jnp.float32)])

    preconditioner = Preconditioner.init(input_block=input_blocks, output_block=output_blocks)

    assert preconditioner.input_block_tril is not None
    assert preconditioner.input_block_tril.shape == (2, 6)
    assert preconditioner.output_block_tril is not None
    assert preconditioner.output_block_tril.shape == (2, 3)
    assert preconditioner.input_block is not None
    assert preconditioner.output_block is not None
    _assert_close(result=preconditioner.input_block, reference=input_blocks)
    _assert_close(result=preconditioner.output_block, reference=output_blocks)
    _assert_close(
        result=jnp.asarray(preconditioner.magnitude),
        reference=jnp.mean(
            jnp.trace(input_blocks, axis1=-2, axis2=-1) * jnp.trace(output_blocks, axis1=-2, axis2=-1),
        ),
    )


def test_preconditioner_dict_init_for_model_creates_empty_preconditioners_for_each_weight_matrix() -> None:
    preconditioners = PreconditionerDict.init_for_model(_model())

    assert set(preconditioners) == {
        (".block", ".matrix"),
        (".first",),
    }
    assert all(preconditioner.input_block is None for preconditioner in preconditioners.values())
    assert all(preconditioner.output_block is None for preconditioner in preconditioners.values())


def test_preconditioner_dict_save_restore_roundtrips_preconditioners(tmp_path: Path) -> None:
    input_block = jnp.array(
        [
            [2.0, -1.0],
            [-1.0, 3.0],
        ],
        dtype=jnp.float32,
    )
    output_block = jnp.array(
        [
            [5.0, 0.5],
            [0.5, 7.0],
        ],
        dtype=jnp.float32,
    )
    preconditioners = PreconditionerDict(
        {
            (".block", ".matrix"): Preconditioner.init(input_block=input_block),
            (".empty",): Preconditioner.identity(),
            (".first",): Preconditioner.init(output_block=output_block),
        },
    )

    preconditioners.save(tmp_path / "preconditioners")
    restored = PreconditionerDict.restore(tmp_path / "preconditioners")

    assert set(restored) == set(preconditioners)
    assert restored[(".empty",)].input_block is None
    assert restored[(".empty",)].output_block is None
    restored_input_block = restored[(".block", ".matrix")].input_block
    restored_output_block = restored[(".first",)].output_block
    assert restored_input_block is not None
    assert restored_output_block is not None
    _assert_close(
        result=restored_input_block,
        reference=input_block,
    )
    _assert_close(
        result=restored_output_block,
        reference=output_block,
    )


def test_preconditioner_dict_save_restores_unsharded_preconditioners(
    tmp_path: Path,
    fake_mesh: jax.sharding.Mesh,
) -> None:
    input_block = jax.device_put(
        jnp.stack(
            [
                jnp.eye(2, dtype=jnp.float32),
                2 * jnp.eye(2, dtype=jnp.float32),
            ],
        ),
        make_sharding((ShardingAxis.DATA, None, None)),
    )
    assert is_sharded(input_block.sharding)
    assert input_block.sharding.mesh == fake_mesh

    preconditioners = PreconditionerDict(
        {
            (".first",): Preconditioner.init(input_block=input_block),
        },
    )

    preconditioners.save(tmp_path / "preconditioners")
    metadata = ocp.StandardCheckpointer().metadata(tmp_path / "preconditioners")
    assert metadata.item_metadata is not None
    input_block_metadata = metadata.item_metadata.tree[0]["input_block_tril"]
    assert input_block_metadata.sharding is not None
    metadata_sharding = input_block_metadata.sharding.to_jax_sharding()
    assert isinstance(metadata_sharding, NamedSharding)
    assert metadata_sharding.mesh == fake_mesh
    assert tuple(metadata_sharding.spec) == (None, None)

    restored_preconditioner = PreconditionerDict.restore(tmp_path / "preconditioners")[(".first",)]

    assert restored_preconditioner.input_block_tril is not None
    assert isinstance(restored_preconditioner.input_block_tril, jax.Array)
    restored_sharding = restored_preconditioner.input_block_tril.sharding
    assert isinstance(restored_sharding, NamedSharding)
    assert restored_sharding.mesh == fake_mesh
    assert tuple(restored_sharding.spec) == (None, None)
    restored_input_block = restored_preconditioner.input_block
    assert restored_input_block is not None
    _assert_close(
        result=restored_input_block,
        reference=input_block,
    )
