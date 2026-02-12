import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import requests
from typer.testing import CliRunner

from lalamo.main import app

DEFAULT_JUDGE_MODEL = "meta-llama/llama-3.3-70b-instruct"

TASK_PROMPT = "Implement a double-pivot quicksort in Rust and a small benchmark. Minimalistic implementation."

# Representative language models < 10B: smallest per family + one MLX 4bit each.
MODEL_REPOS = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "EssentialAI/rnj-1-instruct",
    "google/gemma-2-2b-it",
    "google/gemma-3-1b-it",
    # "google/functiongemma-270m-it",  # output is weird by default, can't verify coherence
    "mlx-community/gemma-3-1b-it-4bit",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "LiquidAI/LFM2-350M",
    "LiquidAI/LFM2.5-1.2B-Instruct",
    "mlx-community/LFM2-350M-4bit",
    "meta-llama/Llama-3.2-1B-Instruct",
    "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "cartesia-ai/Llamba-1B",
    "cartesia-ai/Llamba-1B-4bit-mlx",
    "POLARIS-Project/Polaris-4B-Preview",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-0.6B-MLX-4bit",
]

COHERENT_EXAMPLE = """\
```rust
use std::time::Instant;

fn double_pivot_quicksort(a: &mut [i32]) {
    if a.len() <= 1 { return; }
    if a[0] > a[a.len() - 1] { a.swap(0, a.len() - 1); }
    let (lp, rp) = (a[0], a[a.len() - 1]);
    let (mut lt, mut i, mut gt) = (1, 1, a.len() - 2);
    while i <= gt {
        if a[i] < lp { a.swap(i, lt); lt += 1; i += 1; }
        else if a[i] > rp { a.swap(i, gt); if gt == 0 { break; } gt -= 1; }
        else { i += 1; }
    }
    lt -= 1; gt += 1;
    a.swap(0, lt); a.swap(a.len() - 1, gt);
    if lt > 0 { double_pivot_quicksort(&mut a[..lt]); }
    if gt > lt + 1 { double_pivot_quicksort(&mut a[lt + 1..gt]); }
    double_pivot_quicksort(&mut a[gt + 1..]);
}

fn main() {
    let mut v = (0..20_000).rev().collect::<Vec<_>>();
    let t0 = Instant::now();
    double_pivot_quicksort(&mut v);
    println!("double-pivot quicksort: {:?}", t0.elapsed());
}
```"""

INCOHERENT_EXAMPLES = [
    (
        (
            "fndoublepivot(v:&mut[i32]){v.sort();}metalsandcorrosionmatter"
            "morethanrustalgorithms.aluminumoxidationexplainslatency."
        ),
        "<eos><eos><eos><eos>",
        {
            "coherent": False,
            "score": 0.05,
            "summary": "Garbled tokens with off-topic corrosion tangent and eos spam",
            "issues": ["unreadable", "off_topic", "eos_spam"],
        },
    ),
    (
        (
            "Rust?? benchmark?? no idea. copper rusts and steel passivates. "
            "The moon affects ocean tides and therefore sorting complexity."
        ),
        {
            "coherent": False,
            "score": 0.05,
            "summary": "No implementation, entirely off-topic rambling",
            "issues": ["no_implementation", "off_topic"],
        },
    ),
    (
        "<eos><eos><eos><eos><eos> double pivot double pivot double pivot lorem ipsum lorem ipsum lorem ipsum",
        {
            "coherent": False,
            "score": 0.0,
            "summary": "Degenerate repetition and eos spam with no code",
            "issues": ["eos_spam", "repetition", "no_implementation"],
        },
    ),
    (
        (
            'fn main(){println!("hello")} and now a tangent about battery chemistry, '
            "surface pitting, and galvanic series instead of sorting."
        ),
        {
            "coherent": False,
            "score": 0.1,
            "summary": "Trivial unrelated code followed by off-topic tangent",
            "issues": ["off_topic", "wrong_implementation"],
        },
    ),
]

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CoherenceVerdict:
    coherent: bool
    score: float
    summary: str
    issues: list[str]


def _strip_ansi(text: str) -> str:
    ansi_regex = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    return ansi_regex.sub("", text)


def _extract_output(raw: str) -> str:
    cleaned = _strip_ansi(raw)
    lines = [line.rstrip() for line in cleaned.splitlines() if line.strip()]
    return "\n".join(lines).strip()


def _candidate_message(output: str) -> str:
    return f"Task: {TASK_PROMPT}\n\nCandidate output:\n{output}"


def _build_messages(candidate_output: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You judge whether model outputs are coherent responses to a given task."
                " The output may be truncated due to a token limit, so partial"
                " implementations or thinking/planning without code are perfectly fine"
                " as long as the text is on-topic and coherent."
                " For each candidate output, return ONLY a JSON object with keys:"
                ' "coherent" (bool), "score" (0.0-1.0), "summary" (short string),'
                ' "issues" (array of strings).'
                " Mark as incoherent if the output is off-topic, unreadable,"
                " has repeated/degenerate tokens, or eos spam."
            ),
        },
    ]

    messages.append({"role": "user", "content": _candidate_message(COHERENT_EXAMPLE)})
    messages.append(
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "coherent": True,
                    "score": 0.95,
                    "summary": "Correct double-pivot quicksort in Rust with benchmark",
                    "issues": [],
                },
            ),
        },
    )

    for output, verdict in INCOHERENT_EXAMPLES:
        messages.append({"role": "user", "content": _candidate_message(output)})
        messages.append({"role": "assistant", "content": json.dumps(verdict)})

    messages.append({"role": "user", "content": _candidate_message(candidate_output)})
    return messages


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.index("\n") if "\n" in text else len(text)
        text = text[first_newline + 1 :]
    text = text.removesuffix("```")
    return text.strip()


def _parse_verdict(content: str) -> CoherenceVerdict:
    content = _strip_markdown_fences(content)
    parsed: dict[str, Any] = json.loads(content)

    raw_issues = parsed.get("issues")
    if isinstance(raw_issues, list):
        issues = [str(i) for i in raw_issues]
    elif raw_issues is None:
        issues = []
    else:
        issues = [str(raw_issues)]

    try:
        score = float(parsed.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0

    return CoherenceVerdict(
        coherent=bool(parsed.get("coherent", False)),
        score=score,
        summary=str(parsed.get("summary", "")).strip(),
        issues=issues,
    )


def _judge(*, api_key: str, model: str, candidate_output: str, timeout: int, max_retries: int = 3) -> CoherenceVerdict:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        resp = requests.post(
            OPENROUTER_ENDPOINT,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-Title": "lalamo-coherence-tests",
            },
            json={
                "model": model,
                "messages": _build_messages(candidate_output),
                "temperature": 0.0,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        try:
            return _parse_verdict(content)
        except json.JSONDecodeError as e:
            last_error = e
            log.warning("Judge returned invalid JSON (attempt %d/%d): %s", attempt + 1, max_retries, content[:200])
    raise last_error  # type: ignore[misc]


@pytest.fixture(params=MODEL_REPOS, ids=lambda repo: repo.split("/")[-1])
def converted_model_path(request: pytest.FixtureRequest, tmp_path: Path) -> Path:
    repo = request.param
    output_dir = tmp_path / "model"

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["convert", repo, "--output-dir", str(output_dir), "--overwrite"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, f"lalamo convert {repo} failed (exit {result.exit_code}): {result.output.strip()!r}"
    return output_dir


def test_model_coherent(converted_model_path: Path) -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("Set OPENROUTER_API_KEY to run coherence tests.")

    judge_model = os.getenv("COHERENCE_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["chat", str(converted_model_path), "--message", TASK_PROMPT, "--max-tokens", "256"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, f"lalamo chat failed (exit {result.exit_code}): {result.output.strip()!r}"

    output = _extract_output(result.stdout)
    assert output, "Model produced empty output"

    log.info("Model output:\n%s", output)

    verdict = _judge(
        api_key=api_key,
        model=judge_model,
        candidate_output=output,
        timeout=60,
    )

    log.info(
        "Judge verdict: coherent=%s, score=%.2f, issues=%s, summary=%s",
        verdict.coherent,
        verdict.score,
        verdict.issues,
        verdict.summary,
    )

    issues = ", ".join(verdict.issues) or "none"
    assert verdict.coherent, (
        f"The model output ({output}) was found to be incoherent "
        f"(score={verdict.score:.2f}, issues={issues}, summary={verdict.summary!r})"
    )
