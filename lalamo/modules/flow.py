from abc import abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from enum import Enum
from typing import Any, ClassVar, NamedTuple, Protocol, Self, overload, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.lax import DotAlgorithmPreset
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, DTypeLike, Float, Int, PyTree

from lalamo.exportable import Exportable
from lalamo.initializer import Initializer
from lalamo.module import ForwardPassMode, Keychain, LalamoConfig, LalamoModule, LogicalAxis, ShardingConfig
from lalamo.utils.registry_abc import RegistryABC
from lalamo.weight_matrix import GradientEstimator, MatmulConfig

from .rope import PositionalEmbeddings, RoPEConfig
from .utils import call_vmapped, gather_suffix_tokens

__all__ = [
    "Add",
    "AttentionImplementation",
    "Context",
    "Emit",
    "EmitConfig",
    "Expand",
    "FlowForwardPassConfig",
    "FlowResult",
    "MLPForwardPassConfig",
    "MixerForwardPassConfig",
    "Op",
    "Sequence",
    "SequenceConfig",
    "Sink",
    "SinkConfig",
    "State",
    "StateLayerBase",
    "StepConfig",
    "StepTrace",
    "SumStreams",
    "TokenMixerBase",
    "TokenMixerConfig",
    "TokenMixerResult",
    "Value",
    "init_state",
    "leaves",
    "residual",
    "run",
    "state_owners",
    "validate_program",
    "walk",
]


type Value = PyTree[Array]


class StateLayerBase(Exportable, eqx.Module):
    pass


@register_pytree_node_class
class State(tuple[StateLayerBase, ...]):
    __slots__ = ()

    def tree_flatten(self) -> tuple[tuple[StateLayerBase, ...], None]:
        return (tuple(self), None)

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[StateLayerBase, ...]) -> Self:  # noqa: ARG003
        return cls(children)


class AttentionImplementation(Enum):
    PALLAS = "pallas"
    STABLE_REDUCTION = "stable_reduction"
    XLA = "xla"


@dataclass(frozen=True)
class MixerForwardPassConfig:
    attention_implementation: AttentionImplementation = AttentionImplementation.PALLAS
    attention_accumulation_dtype: DTypeLike | None = jnp.float32
    rope_dtype: DTypeLike | None = jnp.float32
    attention_tile_size: int = 128
    ssm_chunk_size: int = 32
    ssm_min_tail_size_to_chunk: int = 16
    matmul_config: MatmulConfig = dataclass_field(default_factory=MatmulConfig)

    @classmethod
    def for_tracer_tests(cls) -> Self:
        return cls(
            attention_implementation=AttentionImplementation.STABLE_REDUCTION,
            matmul_config=MatmulConfig.for_tracer_tests(),
        )

    @classmethod
    def for_inference(cls, precision: DotAlgorithmPreset = DotAlgorithmPreset.DEFAULT) -> Self:
        return cls(
            attention_implementation=AttentionImplementation.PALLAS,
            matmul_config=MatmulConfig.for_inference(precision),
        )

    @classmethod
    def for_training(
        cls,
        gradient_estimator: GradientEstimator = GradientEstimator.DETERMINISTIC_ROUNDING,
        precision: DotAlgorithmPreset = DotAlgorithmPreset.DEFAULT,
    ) -> Self:
        return cls(
            attention_implementation=AttentionImplementation.XLA,
            ssm_min_tail_size_to_chunk=0,
            matmul_config=MatmulConfig.for_training(gradient_estimator, precision),
        )


@dataclass(frozen=True)
class MLPForwardPassConfig:
    mode: ForwardPassMode = ForwardPassMode.MULTI_TOKEN
    moe_chunk_size_ratio: float = 0.2
    matmul_config: MatmulConfig = dataclass_field(default_factory=MatmulConfig)

    @classmethod
    def for_tracer_tests(cls) -> Self:
        return cls(matmul_config=MatmulConfig.for_tracer_tests())

    @classmethod
    def for_inference(
        cls,
        mode: ForwardPassMode = ForwardPassMode.MULTI_TOKEN,
        precision: DotAlgorithmPreset = DotAlgorithmPreset.DEFAULT,
    ) -> Self:
        return cls(mode=mode, matmul_config=MatmulConfig.for_inference(precision))

    @classmethod
    def for_training(
        cls,
        gradient_estimator: GradientEstimator = GradientEstimator.DETERMINISTIC_ROUNDING,
        precision: DotAlgorithmPreset = DotAlgorithmPreset.DEFAULT,
    ) -> Self:
        return cls(matmul_config=MatmulConfig.for_training(gradient_estimator, precision))


@dataclass(frozen=True)
class FlowForwardPassConfig:
    mixer_forward_pass_config: MixerForwardPassConfig = dataclass_field(default_factory=MixerForwardPassConfig)
    # Every primitive other than a token mixer uses the MLP settings.
    mlp_forward_pass_config: MLPForwardPassConfig = dataclass_field(default_factory=MLPForwardPassConfig)

    @property
    def matmul_config(self) -> MatmulConfig:
        return self.mlp_forward_pass_config.matmul_config

    @classmethod
    def for_tracer_tests(cls) -> Self:
        return cls(MixerForwardPassConfig.for_tracer_tests(), MLPForwardPassConfig.for_tracer_tests())

    @classmethod
    def for_inference(
        cls,
        mode: ForwardPassMode = ForwardPassMode.MULTI_TOKEN,
        precision: DotAlgorithmPreset = DotAlgorithmPreset.DEFAULT,
    ) -> Self:
        return cls(
            MixerForwardPassConfig.for_inference(precision), MLPForwardPassConfig.for_inference(mode, precision)
        )

    @classmethod
    def for_training(
        cls,
        gradient_estimator: GradientEstimator = GradientEstimator.DETERMINISTIC_ROUNDING,
        precision: DotAlgorithmPreset = DotAlgorithmPreset.DEFAULT,
    ) -> Self:
        return cls(
            MixerForwardPassConfig.for_training(gradient_estimator, precision),
            MLPForwardPassConfig.for_training(gradient_estimator, precision),
        )


class Context(eqx.Module):
    # Absent when the stream does not come from tokens, e.g. in speculative draft blocks.
    token_ids: Int[Array, "batch tokens"] | None
    token_positions: Int[Array, "batch tokens"]
    lengths: Int[Array, " batch"] | None
    parents: Int[Array, "batch tokens"] | None
    forward_pass_config: FlowForwardPassConfig = eqx.field(static=True)
    sharding_config: ShardingConfig = eqx.field(static=True)
    return_updated_state: bool = eqx.field(static=True, default=False)

    def positional_embeddings(self, rope_config: RoPEConfig) -> PositionalEmbeddings:
        return call_vmapped(
            rope_config.compute_positional_embeddings,
            self.token_positions,
            added_sharding_axis=self.sharding_config.resolve_axis(LogicalAxis.BATCH),
        ).astype(self.forward_pass_config.mixer_forward_pass_config.rope_dtype)


@dataclass(frozen=True)
class StepConfig(LalamoConfig, RegistryABC):
    @abstractmethod
    def init(self, initializer: Initializer, model_dim: int) -> "Node": ...


@dataclass(frozen=True)
class Op(StepConfig):
    """A parameterless primitive; the config is its own (static) node."""

    temporal: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init_subclass__(**kwargs)
        jax.tree_util.register_static(cls)

    def init(self, initializer: Initializer, model_dim: int) -> Self:  # noqa: ARG002
        return self

    @abstractmethod
    def forward(self, x: Value, *operands: Value, ctx: Context, keychain: Keychain) -> Value: ...


@dataclass(frozen=True)
class Add(Op):
    def forward(self, x: Value, *operands: Value, ctx: Context, keychain: Keychain) -> Value:  # noqa: ARG002
        (operand,) = operands
        assert x.dtype == operand.dtype
        return x + operand


@dataclass(frozen=True)
class Expand(Op):
    streams: int

    def forward(self, x: Value, *operands: Value, ctx: Context, keychain: Keychain) -> Value:  # noqa: ARG002
        () = operands
        return jnp.tile(x, (1, 1, self.streams))


@dataclass(frozen=True)
class SumStreams(Op):
    streams: int

    def forward(self, x: Value, *operands: Value, ctx: Context, keychain: Keychain) -> Value:  # noqa: ARG002
        () = operands
        return x.reshape(*x.shape[:-1], self.streams, -1).sum(axis=-2)


@dataclass(frozen=True)
class SequenceConfig(StepConfig):
    steps: tuple[StepConfig, ...]

    def init(self, initializer: Initializer, model_dim: int) -> "Sequence":
        return Sequence(
            self, initializer.sharding_config, tuple(step.init(initializer, model_dim) for step in self.steps)
        )


@dataclass(frozen=True, eq=False)
class EmitConfig(StepConfig):
    """Stores `body(x)` (or `x`) for later Sinks and passes `x` through. Compared by identity."""

    shared_by_reference: ClassVar[bool] = True
    __eq__ = object.__eq__
    __hash__ = object.__hash__

    name: str
    body: StepConfig | None = None

    def init(self, initializer: Initializer, model_dim: int) -> "Emit":
        body = self.body.init(initializer, model_dim) if self.body is not None else None
        return Emit(self, initializer.sharding_config, body)


@dataclass(frozen=True)
class SinkConfig(StepConfig):
    """Replaces `x` with `combine(x, *emitted values of sources)`."""

    sources: tuple[EmitConfig, ...]
    combine: StepConfig

    def init(self, initializer: Initializer, model_dim: int) -> "Sink":
        return Sink(self, initializer.sharding_config, self.combine.init(initializer, model_dim))


class Sequence(LalamoModule[SequenceConfig]):
    steps: tuple["Node", ...]


class Emit(LalamoModule[EmitConfig]):
    body: "Node | None"


class Sink(LalamoModule[SinkConfig]):
    combine: "Node"


@runtime_checkable
class Primitive(Protocol):
    temporal: ClassVar[bool]

    def forward(self, x: Value, *operands: Value, ctx: Context, keychain: Keychain) -> Value: ...


class TokenMixerResult[StateLayerT](NamedTuple):
    outputs: Float[Array, "*batch suffix_tokens channels"]
    state: StateLayerT | None = None


@dataclass(frozen=True)
class TokenMixerConfig(StepConfig):
    shared_by_reference: ClassVar[bool] = True

    def state_owner(self) -> "TokenMixerConfig":
        return self

    @abstractmethod
    def init(self, initializer: Initializer, model_dim: int) -> "TokenMixerBase": ...


class TokenMixerBase[ConfigT: TokenMixerConfig, StateLayerT: StateLayerBase](LalamoModule[ConfigT]):
    """A primitive that mixes information across tokens and owns its own slot of the model state."""

    @property
    @abstractmethod
    def model_dim(self) -> int: ...

    @abstractmethod
    def init_static_state(self, capacity: int, dtype: DTypeLike) -> StateLayerT: ...

    def requires_full_sequence(self, state: StateLayerT | None, ctx: Context, suffix: int) -> bool:  # noqa: ARG002
        return True

    def forward(
        self,
        x: Float[Array, "batch tokens channels"],
        *,
        state: StateLayerT | None,
        ctx: Context,
        keychain: Keychain,
    ) -> TokenMixerResult[StateLayerT]:
        return self.forward_rows(x, state, ctx, None, keychain=keychain)

    def forward_rows(
        self,
        x: Float[Array, "batch tokens channels"],
        state: StateLayerT | None,
        ctx: Context,
        positional_embeddings: PositionalEmbeddings | None,
        *,
        keychain: Keychain,
    ) -> TokenMixerResult[StateLayerT]:
        """Calls the per-sequence `__call__` of a mixer on every row of the batch."""
        mixer = self

        def call_row(
            inputs: Float[Array, "tokens channels"],
            embeddings: PositionalEmbeddings | None,
            cache: StateLayerT | None,
            length: Array | None,
            parents: Array | None,
            *,
            keychain: Keychain,
        ) -> TokenMixerResult[StateLayerT]:
            return mixer(  # type: ignore[call-arg]
                inputs,
                embeddings,
                cache,
                return_updated_state=ctx.return_updated_state,
                length_without_padding=length,
                forward_pass_config=ctx.forward_pass_config.mixer_forward_pass_config,
                attention_parent_indices=parents,
                keychain=keychain,
            )

        outputs, updated_state = call_vmapped(
            call_row,
            x,
            positional_embeddings,
            state,
            ctx.lengths,
            ctx.parents,
            keychain=keychain,
            added_sharding_axis=self.sharding_config.resolve_axis(LogicalAxis.BATCH),
        )
        return TokenMixerResult(outputs, updated_state)


type Node = Sequence | Emit | Sink | Primitive | TokenMixerBase


def residual(*body: StepConfig) -> SequenceConfig:
    branch = EmitConfig("branch", SequenceConfig(body))
    return SequenceConfig((branch, SinkConfig((branch,), Add())))


@overload
def walk(node: StepConfig) -> Iterator[StepConfig]: ...


@overload
def walk(node: Node) -> Iterator[Node]: ...


def walk(node: Node | StepConfig) -> Iterator[Node | StepConfig]:
    """Pre-order traversal of a program, given either as configs or as initialized modules."""
    yield node
    match node:
        case Sequence() | SequenceConfig():
            for step in node.steps:
                yield from walk(step)
        case Emit() | EmitConfig():
            if node.body is not None:
                yield from walk(node.body)
        case Sink() | SinkConfig():
            yield from walk(node.combine)


@overload
def leaves(node: StepConfig) -> tuple[StepConfig, ...]: ...


@overload
def leaves(node: Node) -> tuple[Primitive | TokenMixerBase, ...]: ...


def leaves(node: Node | StepConfig) -> tuple[Node | StepConfig, ...]:
    """Primitives in execution order; `trace[i]` and loader bindings line up with these."""
    return tuple(
        step
        for step in walk(node)
        if not isinstance(step, Sequence | Emit | Sink | SequenceConfig | EmitConfig | SinkConfig)
    )


def state_owners(program: Node) -> tuple[TokenMixerBase, ...]:
    return tuple(
        node
        for node in leaves(program)
        if isinstance(node, TokenMixerBase) and node.config.state_owner() is node.config
    )


def init_state(program: Node, batch_size: int, capacity: int, dtype: DTypeLike) -> State:
    def batched(owner: TokenMixerBase) -> StateLayerBase:
        return jax.tree.map(
            lambda array: jax.device_put(
                jnp.repeat(array[None, ...], batch_size, axis=0),
                owner.sharding_config.resolve_sharding((LogicalAxis.BATCH, *((None,) * array.ndim))),
            ),
            owner.init_static_state(capacity, dtype),
        )

    return State(batched(owner) for owner in state_owners(program))


def validate_program(program: StepConfig, taps: tuple[EmitConfig, ...] = ()) -> None:
    """Checks that emissions are defined once and before use, and that KV borrowers follow their owners."""
    seen: set[int] = set()
    emitted: set[int] = set()
    mixers: list[TokenMixerConfig] = []

    def visit(node: StepConfig, *, is_combine: bool = False) -> None:
        if is_combine and isinstance(node, SequenceConfig | EmitConfig | SinkConfig | TokenMixerConfig):
            raise ValueError("A sink combine must be a stateless primitive.")
        match node:
            case SequenceConfig():
                for step in node.steps:
                    visit(step)
            case EmitConfig():
                if id(node) in seen:
                    raise ValueError(f"Emitter {node.name!r} occurs twice.")
                seen.add(id(node))
                if node.body is not None:
                    visit(node.body)
                emitted.add(id(node))
            case SinkConfig():
                if any(id(source) not in emitted for source in node.sources):
                    raise ValueError("A sink requires earlier emitters.")
                visit(node.combine, is_combine=True)
            case TokenMixerConfig():
                owner = node.state_owner()
                if owner is not node and not any(earlier is owner for earlier in mixers):
                    raise ValueError("A KV source must be an earlier state-owning attention.")
                mixers.append(node)

    visit(program)
    sources = (node.state_owner() for node in mixers if node.state_owner() is not node)
    if any(sum(node is source for node in mixers) != 1 for source in sources):
        raise ValueError("A KV source must occur exactly once in the program.")
    if any(id(tap) not in emitted for tap in taps):
        raise ValueError("Every tap must be emitted by the program.")


class StepTrace(Exportable, eqx.Module):
    inputs: Value
    operands: tuple[Value, ...]
    outputs: Value


class FlowResult(eqx.Module):
    outputs: Value
    state: State | None
    taps: tuple[Value, ...]
    trace: tuple[StepTrace, ...] | None


def run(
    program: Sequence,
    x: Value,
    ctx: Context,
    state: State | None = None,
    *,
    keychain: Keychain,
    return_updated_state: bool = False,
    suffix: int | None = None,
    taps: tuple[EmitConfig, ...] = (),
    trace: bool = False,
) -> FlowResult:
    owners = state_owners(program)
    if state is not None and len(state) != len(owners):
        raise ValueError(f"Expected {len(owners)} state layers, got {len(state)}.")
    has_borrowers = any(
        isinstance(node, TokenMixerBase) and node.config.state_owner() is not node.config for node in leaves(program)
    )
    ctx = replace(ctx, return_updated_state=return_updated_state or has_borrowers)

    emitted: dict[EmitConfig, Value] = {}
    published: dict[int, StateLayerBase | None] = {}
    updated_states: list[StateLayerBase | None] = []
    traces: list[StepTrace] = []
    incoming_states = iter(state) if state is not None else None

    def apply(node: Node, x: Value, operands: tuple[Value, ...], ctx: Context, keychain: Keychain) -> Value:
        if isinstance(node, TokenMixerBase):
            () = operands
            owner = node.config.state_owner()
            if owner is not node.config:
                layer_state = published[id(owner)]
            elif incoming_states is not None:
                layer_state = next(incoming_states)
            else:
                layer_state = None
            outputs, layer_state = node.forward(x, state=layer_state, ctx=ctx, keychain=keychain)
            if owner is node.config:
                published[id(node.config)] = layer_state
                updated_states.append(layer_state)
        else:
            if not isinstance(node, Primitive):
                raise TypeError(f"{type(node).__name__} cannot be a program step.")
            outputs = node.forward(x, *operands, ctx=ctx, keychain=keychain)
        if trace:
            traces.append(StepTrace(x, operands, outputs))
        return outputs

    def call(node: Node, x: Value, ctx: Context, keychain: Keychain) -> Value:
        match node:
            case Sequence():
                for step, step_keychain in zip(node.steps, keychain.split(len(node.steps)), strict=True):
                    x = call(step, x, ctx, step_keychain)
                return x
            case Emit():
                if node.config in emitted:
                    raise ValueError(f"Emitter {node.config.name!r} occurs twice.")
                emitted[node.config] = x if node.body is None else call(node.body, x, ctx, keychain)
                return x
            case Sink():
                if any(source not in emitted for source in node.config.sources):
                    raise ValueError("A sink requires earlier emitters.")
                operands = tuple(emitted[source] for source in node.config.sources)
                return apply(node.combine, x, operands, ctx, keychain)
            case _:
                return apply(node, x, (), ctx, keychain)

    if suffix is None:
        x = call(program, x, ctx, keychain)
    else:
        if trace or ctx.parents is not None:
            raise ValueError("Suffix outputs cannot be combined with traces or attention parents.")
        cut = _suffix_cut(program, state, ctx, suffix)
        live = {
            *(
                source
                for step in program.steps[cut + 1 :]
                for node in walk(step)
                if isinstance(node, Sink)
                for source in node.config.sources
            ),
            *taps,
        }

        suffix_tokens = suffix

        def gather(value: Value) -> Value:
            return jax.tree.map(
                lambda leaf: gather_suffix_tokens(leaf, ctx.lengths, suffix_tokens, program.sharding_config),
                value,
            )

        step_keys = keychain.split(len(program.steps))
        for step, step_keychain in zip(program.steps[: cut + 1], step_keys[: cut + 1], strict=True):
            x = call(step, x, ctx, step_keychain)
        # Everything after the last temporal step only affects the returned suffix.
        x = gather(x)
        emitted = {emitter: gather(value) for emitter, value in emitted.items() if emitter in live}
        ctx = replace(
            ctx,
            token_ids=None if ctx.token_ids is None else gather(ctx.token_ids),
            token_positions=gather(ctx.token_positions),
            lengths=None,
            parents=None,
        )
        for step, step_keychain in zip(program.steps[cut + 1 :], step_keys[cut + 1 :], strict=True):
            x = call(step, x, ctx, step_keychain)

    if return_updated_state:
        returned_states = tuple(layer_state for layer_state in updated_states if layer_state is not None)
        if len(returned_states) != len(updated_states):
            raise ValueError("A state owner did not return its updated state.")
        updated = State(returned_states)
    else:
        updated = None
    return FlowResult(
        outputs=x,
        state=updated,
        taps=tuple(emitted[tap] for tap in taps),
        trace=tuple(traces) if trace else None,
    )


def _suffix_cut(program: Sequence, state: State | None, ctx: Context, suffix: int) -> int:
    owners = state_owners(program)
    caches = dict(zip((id(owner.config) for owner in owners), state or (None,) * len(owners), strict=True))
    temporal_steps = [
        index
        for index, step in enumerate(program.steps)
        if any(
            node.requires_full_sequence(caches[id(node.config.state_owner())], ctx, suffix)
            if isinstance(node, TokenMixerBase)
            else node.temporal
            for node in walk(step)
        )
    ]
    return max(temporal_steps, default=-1)
