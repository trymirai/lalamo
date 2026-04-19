"""Small DFlash visualiser — step-by-step view of a single prompt generation.

Mirrors the shape of the retrieval-speculator interact script but drives
:class:`DFlashSpeculator` on top of a locally converted lalamo target and a
DFlash drafter pulled from the HuggingFace cache (the ``.bin`` pointer file
is just the repo id — see ``lalamo/speculator/drafters/dflash.py``).

Usage
-----
    uv run python dflash_interact.py \\
        --model-path ~/hikettei/Models/Qwen3-4B \\
        --dflash-repo z-lab/Qwen3-4B-DFlash-b16 \\
        --prompt "Tell me briefly about the Navier-Stokes equations." \\
        --max-tokens 256
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import typer

from lalamo.message_processor import UserMessage
from lalamo.models.language_model import LanguageModelConfig
from lalamo.speculator.common import SamplerConfig
from lalamo.speculator.drafters.dflash import DFlashSpeculator, load_from_hf
from lalamo.speculator.speculate import SpeculationRun


def run(
    model_path: str,
    dflash_repo: str = "z-lab/Qwen3-4B-DFlash-b16",
    prompt: str = "Tell me briefly about the Navier-Stokes equations.",
    max_tokens: int = 256,
    prompt_pad_length: int = 1024,
    temperature: float = 0.0,
    seed: int = 42,
) -> None:
    resolved = Path(model_path).expanduser()
    print(f"Loading target: {resolved}", file=sys.stderr)
    llm = LanguageModelConfig.load_model(resolved)
    eos_set = frozenset(int(e) for e in llm.stop_token_ids)

    print(f"Loading DFlash drafter: {dflash_repo}", file=sys.stderr)
    _, draft_model = load_from_hf(dflash_repo)

    config = SamplerConfig(max_tokens=max_tokens, seed=seed)
    speculator = DFlashSpeculator.create(
        decoder=llm.model,
        model=draft_model,
        config=config,
        eos_set=eos_set,
        temperature=temperature,
        prompt_pad_length=prompt_pad_length,
    )

    prompt_ids = llm.message_processor.tokenize_request([UserMessage(content=prompt)])
    print(f"Prompt: {prompt!r}  (len={len(prompt_ids)})", file=sys.stderr)
    print(
        f"Config: block_size={draft_model.config.block_size} max_tokens={max_tokens} "
        f"T={temperature} pad={prompt_pad_length}",
        file=sys.stderr,
    )

    tokenizer = llm.message_processor.tokenizer
    session = SpeculationRun(speculator, prompt_ids)
    prev_len = 0
    t0 = time.perf_counter()
    for step in session:
        emitted = session.result.generated[prev_len:]
        prev_len = len(session.result.generated)
        text = tokenizer.decode(emitted)
        print(
            f"[step {session.result.num_steps:3d}] "
            f"acc={len(step.accepted):>2d}/{len(emitted):>2d} "
            f"bonus_next={step.bonus:>6d}  {text!r}",
        )
    elapsed = time.perf_counter() - t0

    r = session.result
    n = len(r.generated)
    print(
        f"\n{n} tokens in {r.num_steps} steps | "
        f"mean_accept={r.mean_draft_accepted:.2f} | tok/step={r.tokens_per_step:.2f} | "
        f"tok/s={n / elapsed:.1f} | elapsed={elapsed:.2f}s",
    )
    print("\n" + tokenizer.decode(r.generated))


if __name__ == "__main__":
    typer.run(run)
