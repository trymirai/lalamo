from collections.abc import Iterator
from dataclasses import dataclass, field

from lalamo.speculator.common import LMState, SpeculationStep, Speculator


@dataclass
class SpeculativeDecodingResult:
    num_steps: int = 0
    total_accepted: int = 0
    generated: list[int] = field(default_factory=list)

    @property
    def mean_draft_accepted(self) -> float:
        return self.total_accepted / max(self.num_steps, 1)

    @property
    def speculation_rate(self) -> float:
        return self.total_accepted / max(len(self.generated), 1)

    @property
    def tokens_per_step(self) -> float:
        return len(self.generated) / max(self.num_steps, 1)


class SpeculationRun:
    speculator: Speculator
    lm_state: LMState
    result: SpeculativeDecodingResult

    def __init__(self, speculator: Speculator, prompt_ids: list[int]) -> None:
        self.speculator, self.lm_state = speculator.prefill(prompt_ids)
        self.result = SpeculativeDecodingResult()

    def done(self) -> bool:
        return len(self.result.generated) >= self.speculator.config.max_tokens

    def stopped(self) -> bool:
        return self.lm_state.bonus in self.speculator.eos_set or (
            bool(self.result.generated and self.result.generated[-1] in self.speculator.eos_set)
        )

    def __iter__(self) -> Iterator[SpeculationStep]:
        while not self.done() and not self.stopped():
            lm = self.lm_state
            self.speculator, new_lm, step = self.speculator.step(lm)
            emitted = [lm.bonus, *step.accepted]
            remaining = self.speculator.config.max_tokens - len(self.result.generated)
            self.result.generated.extend(emitted[:remaining])
            self.result.total_accepted += len(step.accepted)
            self.result.num_steps += 1
            self.lm_state = new_lm
            yield step

        if self.lm_state.bonus in self.speculator.eos_set:
            self.result.generated.append(self.lm_state.bonus)
