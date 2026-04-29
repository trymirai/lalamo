import gc
import weakref
from dataclasses import dataclass

import jax.numpy as jnp

from lalamo.utils.registry_abc import RegistryABC, make_registry_abc_converter


def test_registry_tracks_descendants_by_root() -> None:
    class Root(RegistryABC):
        pass

    class OtherRoot(RegistryABC):
        pass

    class Child(Root):
        pass

    class Grandchild(Child):
        pass

    class OtherChild(OtherRoot):
        pass

    assert set(Root.__descendants__()) == {Child, Grandchild}
    assert set(Child.__descendants__()) == {Grandchild}
    assert set(OtherRoot.__descendants__()) == {OtherChild}


def test_registry_excludes_classes_that_directly_inherit_registry_abc() -> None:
    class Root(RegistryABC):
        pass

    class Child(Root):
        pass

    class Hybrid(Root, RegistryABC):
        pass

    assert Child in Root.__descendants__()
    assert Hybrid not in Root.__descendants__()


def test_registry_releases_unreferenced_descendants() -> None:
    class Root(RegistryABC):
        pass

    def build_descendant() -> type[Root]:
        class Temporary(Root):
            pass

        return Temporary

    temporary = build_descendant()
    temporary_ref = weakref.ref(temporary)

    assert temporary in Root.__descendants__()

    del temporary
    gc.collect()

    assert temporary_ref() is None
    assert {descendant.__name__ for descendant in Root.__descendants__()} == set()


def test_registry_converter_roundtrips_subclass_with_type_tag() -> None:
    class OptimizerConfig(RegistryABC):
        pass

    @dataclass(frozen=True)
    class AdamConfig(OptimizerConfig):
        learning_rate: float
        dtype: jnp.dtype

    converter = make_registry_abc_converter()
    config = AdamConfig(learning_rate=0.001, dtype=jnp.dtype(jnp.bfloat16))

    raw_config = converter.unstructure(config, unstructure_as=OptimizerConfig)
    restored_config = converter.structure(raw_config, OptimizerConfig)

    assert raw_config == {
        "type": "AdamConfig",
        "learning_rate": 0.001,
        "dtype": "bfloat16",
    }
    assert restored_config == config


def test_registry_converter_applies_nested_registered_field_hooks() -> None:
    class OptimizerConfig(RegistryABC):
        pass

    @dataclass(frozen=True)
    class LearningRate:
        value: float

    @dataclass(frozen=True)
    class ScheduleConfig:
        learning_rate: LearningRate

    @dataclass(frozen=True)
    class AdamConfig(OptimizerConfig):
        schedule: ScheduleConfig

    converter = make_registry_abc_converter()
    converter.register_unstructure_hook(LearningRate, lambda learning_rate: f"{learning_rate.value:.3f}")
    converter.register_structure_hook(LearningRate, lambda value, _: LearningRate(value=float(value)))

    config = AdamConfig(schedule=ScheduleConfig(learning_rate=LearningRate(value=0.001)))

    raw_config = converter.unstructure(config, unstructure_as=OptimizerConfig)
    restored_config = converter.structure(raw_config, OptimizerConfig)

    assert raw_config == {
        "type": "AdamConfig",
        "schedule": {
            "learning_rate": "0.001",
        },
    }
    assert restored_config == config


def test_registry_converter_handles_none_for_optional_registry_values() -> None:
    class OptimizerConfig(RegistryABC):
        pass

    converter = make_registry_abc_converter()

    assert converter.unstructure(None, unstructure_as=OptimizerConfig | None) is None
    assert converter.structure(None, OptimizerConfig | None) is None


def test_registry_converter_roundtrips_registry_type_fields() -> None:
    class OptimizerConfig(RegistryABC):
        pass

    @dataclass(frozen=True)
    class AdamConfig(OptimizerConfig):
        learning_rate: float

    @dataclass(frozen=True)
    class TrainingConfig:
        optimizer_config_type: type[OptimizerConfig]
        fallback_optimizer_config_type: type[OptimizerConfig] | None = None

    converter = make_registry_abc_converter()
    config = TrainingConfig(
        optimizer_config_type=AdamConfig,
        fallback_optimizer_config_type=AdamConfig,
    )

    raw_config = converter.unstructure(config)
    restored_config = converter.structure(raw_config, TrainingConfig)

    assert raw_config == {
        "optimizer_config_type": "AdamConfig",
        "fallback_optimizer_config_type": "AdamConfig",
    }
    assert restored_config == config


def test_registry_converter_roundtrips_registry_typevar_fields() -> None:
    class OptimizerConfig(RegistryABC):
        pass

    @dataclass(frozen=True)
    class AdamConfig(OptimizerConfig):
        learning_rate: float

    @dataclass(frozen=True)
    class TrainingConfig[OptimizerConfigT: OptimizerConfig]:
        optimizer_config: OptimizerConfigT

    @dataclass(frozen=True)
    class AdamTrainingConfig(TrainingConfig[AdamConfig]):
        num_steps: int

    converter = make_registry_abc_converter()
    config = AdamTrainingConfig(
        optimizer_config=AdamConfig(learning_rate=0.001),
        num_steps=100,
    )

    raw_config = converter.unstructure(config)
    restored_config = converter.structure(raw_config, AdamTrainingConfig)

    assert raw_config == {
        "optimizer_config": {
            "type": "AdamConfig",
            "learning_rate": 0.001,
        },
        "num_steps": 100,
    }
    assert restored_config == config
