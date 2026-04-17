import dataclasses
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from lalamo.message_processor import MessageProcessor, UserMessage
from lalamo.speculator.common import Speculator
from lalamo.speculator.sampler import GumbelSeed
from lalamo.speculator.speculate import SpeculationRun, SpeculativeDecodingResult


@dataclass(frozen=True)
class EvalQuestion:
    id: int
    category: str
    prompt: str


@dataclass(frozen=True)
class CategoryStats:
    tokens: int
    steps: int
    accepted: int
    count: int
    elapsed_s: float

    @property
    def tokens_per_step(self) -> float:
        return self.tokens / max(self.steps, 1)

    @property
    def mean_draft_accepted(self) -> float:
        return self.accepted / max(self.steps, 1)

    @property
    def speculation_rate(self) -> float:
        return self.accepted / max(self.tokens, 1)

    @property
    def tokens_per_second(self) -> float:
        return self.tokens / self.elapsed_s if self.elapsed_s > 0 else 0.0


@dataclass(frozen=True)
class EvalResults:
    by_category: dict[str, CategoryStats]
    total_tokens: int
    total_steps: int
    total_accepted: int
    total_elapsed_s: float

    @property
    def total_count(self) -> int:
        return sum(stats.count for stats in self.by_category.values())

    @property
    def tokens_per_step(self) -> float:
        return self.total_tokens / max(self.total_steps, 1)

    @property
    def mean_draft_accepted(self) -> float:
        return self.total_accepted / max(self.total_steps, 1)

    @property
    def speculation_rate(self) -> float:
        return self.total_accepted / max(self.total_tokens, 1)

    @property
    def tokens_per_second(self) -> float:
        return self.total_tokens / self.total_elapsed_s if self.total_elapsed_s > 0 else 0.0


def evaluate_prompt(
    speculator: Speculator,
    mp: MessageProcessor,
    prompt: str,
) -> SpeculativeDecodingResult:
    prompt_ids = mp.tokenize_request([UserMessage(content=prompt)])
    session = SpeculationRun(speculator, prompt_ids)
    for _ in session:
        pass
    return session.result


def run_eval(
    speculator: Speculator,
    mp: MessageProcessor,
    questions: Iterable[EvalQuestion],
    on_question: Callable[[int, EvalQuestion, SpeculativeDecodingResult, float], None] | None = None,
) -> EvalResults:
    accum: dict[str, list[tuple[SpeculativeDecodingResult, float]]] = {}
    for i, question in enumerate(questions):
        per_question = dataclasses.replace(speculator, seed=GumbelSeed(42 + i))
        start = time.perf_counter()
        result = evaluate_prompt(per_question, mp, question.prompt)
        elapsed_s = time.perf_counter() - start
        accum.setdefault(question.category, []).append((result, elapsed_s))
        if on_question is not None:
            on_question(i, question, result, elapsed_s)

    by_category = {
        category: CategoryStats(
            tokens=sum(len(r.generated) for r, _ in entries),
            steps=sum(r.num_steps for r, _ in entries),
            accepted=sum(r.total_accepted for r, _ in entries),
            count=len(entries),
            elapsed_s=sum(t for _, t in entries),
        )
        for category, entries in accum.items()
    }
    return EvalResults(
        by_category=by_category,
        total_tokens=sum(stats.tokens for stats in by_category.values()),
        total_steps=sum(stats.steps for stats in by_category.values()),
        total_accepted=sum(stats.accepted for stats in by_category.values()),
        total_elapsed_s=sum(stats.elapsed_s for stats in by_category.values()),
    )


def print_results(results: EvalResults, label: str = "") -> None:
    prefix = f" [{label}]" if label else ""
    header_width = 90
    print(f"\n{'=' * header_width}{prefix}")
    print(
        f"{'Category':>15s}  {'tok/step':>10s}  {'tok/sec':>10s}  "
        f"{'draft_acc':>10s}  {'spec_rate':>10s}  {'questions':>10s}",
    )
    print(f"{'-' * header_width}")
    for category in sorted(results.by_category):
        stats = results.by_category[category]
        print(
            f"{category:>15s}  {stats.tokens_per_step:>10.2f}  {stats.tokens_per_second:>10.2f}  "
            f"{stats.mean_draft_accepted:>10.2f}  {stats.speculation_rate:>10.2%}  {stats.count:>10d}",
        )
    print(f"{'-' * header_width}")
    print(
        f"{'OVERALL':>15s}  {results.tokens_per_step:>10.2f}  {results.tokens_per_second:>10.2f}  "
        f"{results.mean_draft_accepted:>10.2f}  {results.speculation_rate:>10.2%}  {results.total_count:>10d}",
    )
    print(f"{'=' * header_width}")
