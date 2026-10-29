import copy
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

import jax
from jaxtyping import Array

from lalamo.module import LalamoConfig, LalamoModule
from lalamo.modules.flow import Emit, EmitConfig, Sequence, SequenceConfig, Sink, SinkConfig, StepConfig, leaves
from lalamo.utils.parameter_path import ParameterPath
from lalamo.weight_matrix import CompressionImplementation

__all__ = ["Binder", "Binding", "CheckpointView", "Loader"]


@runtime_checkable
class Loader(Protocol):
    # Dispatch pairs a concrete module with its loader; Python cannot express that dependent type.
    def __call__(
        self,
        module: Any,  # noqa: ANN401
        weights: Mapping[str, Array],
        path: ParameterPath,
        /,
        *,
        implementation: CompressionImplementation,
    ) -> Any: ...  # noqa: ANN401


@dataclass(frozen=True)
class Binding:
    path: ParameterPath
    load: Loader | None


@dataclass
class Binder:
    """Records which checkpoint path holds the weights of each config built by an importer.

    `at(path, config)` returns a fresh copy of `config` bound to `path`; the loader then walks the
    built config tree alongside the initialized module and loads every bound primitive.
    """

    default_loader: Callable[[LalamoModule], Loader] | None = None
    _bindings: dict[int, tuple[LalamoConfig, Binding]] = field(default_factory=dict)

    def __call__[ConfigT: LalamoConfig](self, path: str, config: ConfigT, load: Loader | None = None) -> ConfigT:
        bound = copy.copy(config)
        self._bindings[id(bound)] = (bound, Binding(ParameterPath(path), load))
        return bound

    def binding(self, config: LalamoConfig) -> Binding | None:
        entry = self._bindings.get(id(config))
        if entry is None:
            return None
        _, binding = entry
        return binding

    def sources(self, root: StepConfig) -> tuple[Binding | None, ...]:
        """Bindings aligned with `leaves(root)`, i.e. with program traces."""
        return tuple(self.binding(primitive) for primitive in leaves(root))

    def load[ModuleT](
        self,
        module: ModuleT,
        config: LalamoConfig,
        weights: Mapping[str, Array],
        *,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
    ) -> ModuleT:
        binding = self.binding(config)
        module_config = module.config if isinstance(module, LalamoModule) else module
        if type(module_config) is not type(config):
            raise TypeError(f"Module {type(module).__name__} was not built from {type(config).__name__}.")
        if binding is not None:
            load = binding.load
            if load is None:
                assert self.default_loader is not None and isinstance(module, LalamoModule)
                load = self.default_loader(module)
            return load(module, weights, binding.path, implementation=implementation)
        match module, config:
            case Sequence(), SequenceConfig():
                steps = tuple(
                    self.load(step, step_config, weights, implementation=implementation)
                    for step, step_config in zip(module.steps, config.steps, strict=True)
                )
                return replace(module, steps=steps)
            case Emit(), EmitConfig():
                if module.body is None:
                    return module
                assert config.body is not None
                return replace(
                    module, body=self.load(module.body, config.body, weights, implementation=implementation)
                )
            case Sink(), SinkConfig():
                return replace(
                    module,
                    combine=self.load(module.combine, config.combine, weights, implementation=implementation),
                )
        if jax.tree.leaves(module):
            raise ValueError(f"No checkpoint binding for {type(module).__name__} weights.")
        return module


_LANGUAGE_MODEL_PREFIXES = {"model.language_model.": "model.", "language_model.": ""}


@dataclass(frozen=True)
class CheckpointView(Mapping[str, Array]):
    """Presents a multimodal checkpoint's language model under canonical `model.*` / `lm_head` paths."""

    weights: Mapping[str, Array]
    prefix: str = ""

    @classmethod
    def of(cls, weights: Mapping[str, Array]) -> "CheckpointView":
        for prefix in _LANGUAGE_MODEL_PREFIXES:
            if any(key.startswith(prefix) for key in weights):
                return cls(weights, prefix)
        return cls(weights)

    def _actual(self, key: str) -> str:
        canonical = _LANGUAGE_MODEL_PREFIXES.get(self.prefix)
        if canonical is None or not key.startswith(canonical):
            return key
        mapped = f"{self.prefix}{key.removeprefix(canonical)}"
        return mapped if mapped in self.weights else key

    def __getitem__(self, key: str) -> Array:
        return self.weights[self._actual(key)]

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self._actual(key) in self.weights

    def __iter__(self) -> Iterator[str]:
        canonical = _LANGUAGE_MODEL_PREFIXES.get(self.prefix)
        for key in self.weights:
            if canonical is not None and key.startswith(self.prefix):
                yield f"{canonical}{key.removeprefix(self.prefix)}"
            else:
                yield key

    def __len__(self) -> int:
        return len(self.weights)
