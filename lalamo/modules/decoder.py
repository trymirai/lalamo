from dataclasses import dataclass

import equinox as eqx
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.exportable import Exportable
from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoConfig, LalamoModule, LogicalAxis

from .embedding import EmbeddingBase, EmbeddingConfig
from .flow import (
    Context,
    EmitConfig,
    Sequence,
    SequenceConfig,
    State,
    StepTrace,
    Value,
    init_state,
    run,
    validate_program,
)
from .forward_pass import ModelForwardPassConfig
from .utils import call_vmapped_twice

__all__ = [
    "Decoder",
    "DecoderActivationTrace",
    "DecoderConfig",
    "DecoderResult",
]


class DecoderActivationTrace(Exportable, eqx.Module):
    token_ids: Int[Array, "batch suffix_tokens"]
    token_positions: Int[Array, "batch suffix_tokens"]
    state: State | None
    embeddings: Float[Array, "batch suffix_tokens channels"]
    # One entry per program primitive, in execution (pre-)order.
    steps: tuple[StepTrace, ...]
    taps: tuple[Value, ...]
    outputs: Float[Array, "batch suffix_tokens channels"]


class DecoderResult(Exportable, eqx.Module):
    logits: Float[Array, "batch suffix_tokens vocabulary"]
    updated_state: State | None = None
    activation_trace: DecoderActivationTrace | None = None


@dataclass(frozen=True)
class DecoderConfig(LalamoConfig):
    embedding_config: EmbeddingConfig
    program: SequenceConfig
    model_dim: int
    vocab_size: int
    # Observation points at the checkpoint's hidden-state boundaries, used by traces and speculators.
    taps: tuple[EmitConfig, ...] = ()

    def __post_init__(self) -> None:
        validate_program(self.program, self.taps)

    def init(self, initializer: Initializer) -> "Decoder":
        return Decoder(
            config=self,
            sharding_config=initializer.sharding_config,
            embedding=self.embedding_config.init(initializer, model_dim=self.model_dim, vocab_size=self.vocab_size),
            program=self.program.init(initializer, self.model_dim),
        )


class Decoder(LalamoModule[DecoderConfig]):
    embedding: EmbeddingBase
    program: Sequence

    @property
    def vocab_size(self) -> int:
        return self.embedding.vocab_size

    @eqx.filter_jit
    def __call__(
        self,
        token_ids: Int[Array, "batch suffix_tokens"],
        token_positions: Int[Array, "batch suffix_tokens"],
        state: State | None = None,
        return_updated_state: bool = False,
        return_activation_trace: bool = False,
        lengths_without_padding: Int[Array, " batch"] | None = None,
        forward_pass_config: ModelForwardPassConfig = ModelForwardPassConfig(),
        attention_parent_indices: Int[Array, " batch suffix_tokens"] | None = None,
        return_suffix_tokens: int | None = None,
        *,
        keychain: Keychain,
    ) -> DecoderResult:
        if token_ids.ndim != 2:
            raise ValueError(
                f"token_ids must be a 2D array of size (batch_size, sequence_length), got {token_ids.shape}",
            )
        if token_positions.ndim != 2:
            raise ValueError(
                "token_positions must be a 2D array of size (batch_size, sequence_length),"
                f" got {token_positions.shape}",
            )
        if return_suffix_tokens is not None:
            _, sequence_length = token_ids.shape
            if not 1 <= return_suffix_tokens <= sequence_length:
                raise ValueError(
                    f"return_suffix_tokens must be between 1 and the sequence length {sequence_length},"
                    f" got {return_suffix_tokens}",
                )
            if return_activation_trace:
                raise ValueError("return_suffix_tokens cannot be combined with return_activation_trace.")
        embedding_keychain, program_keychain, readout_keychain = keychain.split(3)
        embeddings = self.embedding.embed(
            token_ids,
            forward_pass_config=forward_pass_config.embedding_forward_pass_config,
            keychain=embedding_keychain,
        )
        result = run(
            self.program,
            embeddings,
            Context(
                token_ids,
                token_positions,
                lengths_without_padding,
                attention_parent_indices,
                forward_pass_config.flow_forward_pass_config,
                self.sharding_config,
            ),
            state,
            keychain=program_keychain,
            return_updated_state=return_updated_state,
            suffix=return_suffix_tokens,
            taps=self.config.taps if return_activation_trace else (),
            trace=return_activation_trace,
        )
        logits = call_vmapped_twice(
            self.embedding.readout,
            result.outputs,
            forward_pass_config=forward_pass_config.embedding_forward_pass_config,
            keychain=readout_keychain,
            added_sharding_axes=(self.sharding_config.resolve_axis(LogicalAxis.BATCH), None),
        )
        activation_trace = None
        if return_activation_trace:
            assert result.trace is not None
            activation_trace = DecoderActivationTrace(
                token_ids=token_ids,
                token_positions=token_positions,
                state=state,
                embeddings=embeddings,
                steps=result.trace,
                taps=result.taps,
                outputs=result.outputs,
            )
        return DecoderResult(logits=logits, updated_state=result.state, activation_trace=activation_trace)

    def init_static_state(self, batch_size: int, capacity: int, dtype: DTypeLike) -> State:
        return init_state(self.program, batch_size, capacity, dtype)
