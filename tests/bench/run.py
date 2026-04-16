"""Run the typical-prompt bench against a converted lalamo model.

Loads the model once, streams each prompt, prints the response, and writes a
JSONL log for offline review.

Usage:
    uv run python tests/bench/run.py models/Qwen3.6-35B-A3B
    uv run python tests/bench/run.py models/Qwen3.6-35B-A3B --max-tokens 256
    uv run python tests/bench/run.py models/Qwen3.6-35B-A3B --only-category reasoning
"""

from __future__ import annotations

import argparse
import json
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.rule import Rule

from lalamo.message_processor import UserMessage
from lalamo.models import LanguageModelConfig

DEFAULT_SPEC = Path(__file__).parent / "prompts.toml"


@dataclass(frozen=True)
class Prompt:
    id: str
    category: str
    prompt: str
    expect: str


def load_prompts(spec_path: Path) -> list[Prompt]:
    with open(spec_path, "rb") as f:
        data = tomllib.load(f)
    return [
        Prompt(id=p["id"], category=p["category"], prompt=p["prompt"], expect=p["expect"]) for p in data["prompts"]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("model_path", type=Path, help="Path to converted lalamo model directory.")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC, help="Prompt spec TOML.")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens per reply.")
    parser.add_argument(
        "--only-category",
        type=str,
        default=None,
        help="Run only prompts in this category.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSONL output path (default: <model_path>/bench_<timestamp>.jsonl).",
    )
    args = parser.parse_args()

    console = Console()
    prompts = load_prompts(args.spec)
    if args.only_category is not None:
        prompts = [p for p in prompts if p.category == args.only_category]
    if not prompts:
        console.print("[red]No prompts matched.[/red]")
        return

    output_path = args.output or args.model_path / f"bench_{int(time.time())}.jsonl"

    console.print(f"[cyan]🚀 Loading {args.model_path}[/cyan]")
    model = LanguageModelConfig.load_model(args.model_path)
    console.print("[cyan]🔥 Warming up[/cyan]")
    list(model.stream_reply_text([UserMessage("")], max_output_length=1))

    console.print(f"[green]📝 Running {len(prompts)} prompts, logging to {output_path}[/green]")

    with open(output_path, "w") as log:
        for idx, p in enumerate(prompts, 1):
            console.print(Rule(f"[{idx}/{len(prompts)}] {p.id} ({p.category})"))
            console.print(f"[cyan]Q:[/cyan] {p.prompt}")
            console.print(f"[dim]expect: {p.expect}[/dim]")
            console.print("[red]A:[/red] ", end="")

            start = time.time()
            tokens: list[str] = []
            for tok in model.stream_reply_text([UserMessage(p.prompt)], max_output_length=args.max_tokens):
                console.print(tok, end="")
                tokens.append(tok)
            elapsed = time.time() - start
            console.print()

            response = "".join(tokens)
            num_tokens = len(tokens)
            tok_per_s = num_tokens / elapsed if elapsed > 0 else 0.0
            console.print(f"[dim]{num_tokens} tokens in {elapsed:.1f}s ({tok_per_s:.2f} tok/s)[/dim]")

            log.write(
                json.dumps(
                    {
                        "id": p.id,
                        "category": p.category,
                        "prompt": p.prompt,
                        "expect": p.expect,
                        "response": response,
                        "num_tokens": num_tokens,
                        "elapsed_s": elapsed,
                    },
                )
                + "\n",
            )
            log.flush()

    console.print(Rule(f"[green]✅ Done — log saved to {output_path}[/green]"))


if __name__ == "__main__":
    main()
