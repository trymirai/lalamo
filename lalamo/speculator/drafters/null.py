"""No-speculation baseline.

:class:`NullSpeculator` implements :class:`Speculator` with an empty
proposal: every step is one target decoder forward that commits a single
token, exactly what plain autoregressive decoding does. It exists so the
DFlash / NGram benchmarks can reuse the ``lalamo speculator eval`` harness
(same tokenizer, same chat template, same prompt set, same loop skeleton)
to measure the non-speculative denominator for speedup calculations.

The pointer file passed to ``lalamo speculator eval`` can be any file
(its contents are ignored):

    : > null.bin
    uv run lalamo speculator eval <target_dir> null.bin --drafter-name null \\
        --dataset gsm8k --num-questions 100 --max-tokens 2048
"""

import dataclasses
from dataclasses import dataclass
from typing import ClassVar, Self, cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from lalamo.modules.common import ForwardPassMode
from lalamo.modules.decoder import Decoder
from lalamo.speculator.common import LMState, SamplerConfig, SpeculationStep, Speculator

__all__ = ["NullSpeculator"]


@dataclass(frozen=True, kw_only=True)
class NullSpeculator(Speculator[None]):
    """No-op speculator — one target forward per step, zero drafts accepted.

    ``step()`` returns ``SpeculationStep(accepted=[], bonus=<next>)``, so
    ``tokens_per_step`` collapses to ``1.0`` and ``speculation_rate`` to 0 —
    exactly the shape a speedup denominator wants.
    """

    name: ClassVar[str] = "null"

    temperature: float = 0.0
    rng_key: Int[Array, "2"]

    @classmethod
    def create(
        cls,
        *,
        decoder: Decoder,
        config: SamplerConfig,
        eos_set: frozenset[int],
        temperature: float = 0.0,
    ) -> Self:
        return cls(
            decoder=decoder,
            config=config,
            eos_set=eos_set,
            temperature=temperature,
            rng_key=jax.random.PRNGKey(config.seed),
            trace_layer_outputs=None,
            trace_output_norm=False,
            prefill_hidden_range=None,
        )

    @classmethod
    def deserialize_impl(cls, data: bytes, **kwargs: object) -> Self:  # noqa: ARG003 — pointer payload ignored
        decoder = kwargs["decoder"]
        config = kwargs["config"]
        eos_set = kwargs["eos_set"]
        temperature = kwargs.get("temperature", 0.0)
        assert isinstance(decoder, Decoder)
        assert isinstance(config, SamplerConfig)
        assert isinstance(eos_set, frozenset)
        assert isinstance(temperature, (int, float))
        return cls.create(
            decoder=decoder,
            config=config,
            eos_set=cast("frozenset[int]", eos_set),
            temperature=float(temperature),
        )

    def serialize(self) -> bytes:
        return b""

    @property
    def generation_capacity(self) -> int:
        return self.config.max_tokens + 16

    def prefill(self, prompt_ids: list[int]) -> tuple[Self, LMState]:
        state = self.decoder.init_static_state(1, self.generation_capacity + len(prompt_ids))
        prefix = jnp.array([prompt_ids], dtype=jnp.int32)
        positions = jnp.arange(len(prompt_ids), dtype=jnp.int32)[None, :]
        fwd = self.decoder(
            prefix,
            positions,
            state,
            return_updated_state=True,
        )
        assert fwd.updated_state is not None
        logits = fwd.logits[0, -1]
        rng_key, sub_key = jax.random.split(self.rng_key)
        bonus = self.sample_token(logits, sub_key)
        new_self = dataclasses.replace(self, rng_key=rng_key)
        lm = LMState(
            kv_cache=fwd.updated_state,
            layer_outputs=(),
            output_norm=None,
            logits=logits,
            position=len(prompt_ids),
            bonus=bonus,
        )
        return new_self, lm

    def draft(self, lm: LMState) -> None:  # noqa: ARG002 — nothing to propose
        return None

    def step(self, lm: LMState) -> tuple[Self, LMState, SpeculationStep]:
        first_layer = lm.kv_cache[0]
        cache_len = first_layer.current_length[0]  # type: ignore[union-attr]
        tok_ids = jnp.array([[lm.bonus]], dtype=jnp.int32)
        positions = cache_len[None, None]
        fwd = self.decoder(
            tok_ids,
            positions,
            lm.kv_cache,
            return_updated_state=True,
            forward_pass_mode=ForwardPassMode.SINGLE_TOKEN,
        )
        assert fwd.updated_state is not None
        next_logits = fwd.logits[0, 0]
        rng_key, sub_key = jax.random.split(self.rng_key)
        next_bonus = self.sample_token(next_logits, sub_key)
        new_lm = LMState(
            kv_cache=fwd.updated_state,
            layer_outputs=(),
            output_norm=None,
            logits=next_logits,
            position=int(cache_len) + 1,
            bonus=next_bonus,
        )
        final_self = dataclasses.replace(self, rng_key=rng_key)
        return final_self, new_lm, SpeculationStep(accepted=[], bonus=next_bonus)

    def sample_token(self, logits: Float[Array, " vocab"], key: Int[Array, "2"]) -> int:
        if self.temperature < 1e-5:
            return int(jnp.argmax(logits))
        return int(jax.random.categorical(key, logits.astype(jnp.float32) / self.temperature))
