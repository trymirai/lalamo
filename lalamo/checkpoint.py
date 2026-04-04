import importlib
import json
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from lalamo.common import stringify_path
from lalamo.model_import.loaders.common import _apply_parameter_sharding, find_field_sharding
from lalamo.modules.common import EmptyInitializer, LalamoModule, ShardingConfig, config_converter

__all__ = ["CheckpointManager"]


@dataclass(frozen=True)
class CheckpointManager:
    directory: Path = Path("checkpoints")

    def save(self, name: str, model: LalamoModule) -> None:
        arrays = eqx.filter(model, eqx.is_array)
        checkpointer = ocp.StandardCheckpointer()
        checkpointer.save(self.directory / name / "state", arrays)
        checkpointer.wait_until_finished()
        flat_with_path, _ = jax.tree_util.tree_flatten_with_path(model)
        array_dtypes = {stringify_path(path): str(leaf.dtype) for path, leaf in flat_with_path if eqx.is_array(leaf)}
        config = model.config
        config_json = {
            "_type": f"{type(config).__module__}.{type(config).__qualname__}",
            "_array_dtypes": array_dtypes,
            **config_converter.unstructure(config),
        }
        (self.directory / name / "config.json").write_text(json.dumps(config_json))

    def restore[M: LalamoModule](
        self,
        name: str,
        sharding_config: ShardingConfig | None = None,
    ) -> M:
        config_path = self.directory / name / "config.json"
        config_json = json.loads(config_path.read_text())
        module_name, class_name = config_json.pop("_type").rsplit(".", 1)
        array_dtypes: dict[str, str] = config_json.pop("_array_dtypes")
        config_type = getattr(importlib.import_module(module_name), class_name)
        config = config_converter.structure(config_json, config_type)
        empty = config.init(EmptyInitializer(precision=jnp.float32))

        def is_restore_leaf(node: object) -> bool:
            return isinstance(node, jax.ShapeDtypeStruct) or eqx.is_array(node)

        target = jax.tree_util.tree_map_with_path(
            lambda path, leaf: (
                _apply_parameter_sharding(
                    jnp.zeros(
                        leaf.shape,
                        config_converter.structure(array_dtypes[stringify_path(path)], jnp.dtype),
                    ),
                    find_field_sharding(empty, leaf),
                    sharding_config,
                )
                if is_restore_leaf(leaf)
                else leaf
            ),
            empty,
            is_leaf=is_restore_leaf,
        )
        arrays, non_arrays = eqx.partition(target, eqx.is_array)
        restored = ocp.StandardCheckpointer().restore(self.directory / name / "state", target=arrays)
        return eqx.combine(restored, non_arrays)
