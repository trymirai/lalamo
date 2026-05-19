import json
from dataclasses import dataclass, field
from pathlib import Path

import equinox as eqx
import orbax.checkpoint as ocp
from jax import ShapeDtypeStruct
from jaxtyping import Array, DTypeLike, PyTree
from tokenizers import Tokenizer

from lalamo.model import Model, ModelConfig

from .initializer import EmptyInitializer
from .module import LalamoModule

__all__ = ["CheckpointManager"]


@dataclass(frozen=True)
class CheckpointManager:
    _directory: Path = field(default=Path("checkpoints"))

    @classmethod
    def init(cls, directory: Path | str = "checkpoints") -> "CheckpointManager":
        return cls(Path(directory))

    def save(self, name: str, model: Model) -> None:
        arrays = eqx.filter(model, eqx.is_array)
        checkpointer = ocp.StandardCheckpointer()
        checkpointer.save(self._directory / name / "state", arrays)
        checkpointer.wait_until_finished()
        with open(self._directory / name / "config.json", "w") as config_file:
            json.dump(model.config.to_json(), config_file)
        model.token_codec.tokenizer.save(str(self._directory / name / "tokenizer.json"))

    def restore[M: LalamoModule](
        self,
        base_config_class: type[ModelConfig],
        name: str,
        dtype: DTypeLike,
    ) -> M:
        initializer = EmptyInitializer(dtype)
        config_path = self._directory / name / "config.json"
        with open(config_path) as config_file:
            config_json = json.load(config_file)
            config = base_config_class.from_json(config_json)
        tokenizer = Tokenizer.from_file(str(self._directory / name / "tokenizer.json"))
        empty = config.init(tokenizer, initializer)

        def is_restore_leaf(node: PyTree) -> bool:
            return isinstance(node, (ShapeDtypeStruct, Array))

        template_arrays, non_arrays = eqx.partition(empty, is_restore_leaf)
        restored_arrays = ocp.StandardCheckpointer().restore(self._directory / name / "state", target=template_arrays)

        return eqx.combine(restored_arrays, non_arrays)
