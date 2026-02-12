import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import requests

DEFAULT_JUDGE_MODEL = "meta-llama/llama-3.3-70b-instruct"

TASK_PROMPT = "Implement a double-pivot quicksort in Rust and a small benchmark. Minimalistic implementation."

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

COHERENT_TRUNCATED_EXAMPLE = """\
fn partition<T: Ord + Copy>(arr: &mut [T], low: usize, high: usize) -> usize {
    let pivot_index = partition_helper(arr, low, high);
    pivot_index
}
fn partition_helper<T: Ord + Copy>(arr: &mut [T], low: usize, high: usize) -> usize {
    let pivot = arr[high];
    let i = low - 1;
    for j in low .. high {
        if arr[j] <= pivot {
            i += 1;
            arr.swap(i, j);
        }
    }
    arr.swap(i + 1, high);
    i + 1
}
fn quicksort<T: Ord + Copy>(arr: &mut [T], low: usize, high: usize) {
    if low < high {
        let pivot_index = partition(arr, low, high);
        quicksort(arr, low, pivot_index - 1);
        quicksort(arr, pivot_index + 1, high);
    }
}
fn main()"""

INCOHERENT_EXAMPLES = [
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
]

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CoherenceVerdict:
    coherent: bool
    score: float
    summary: str
    issues: list[str]


def strip_ansi(text: str) -> str:
    ansi_regex = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    return ansi_regex.sub("", text)


def extract_output(raw: str) -> str:
    cleaned = strip_ansi(raw)
    lines = [line.rstrip() for line in cleaned.splitlines() if line.strip()]
    return "\n".join(lines).strip()


def _candidate_message(task_prompt: str, output: str) -> str:
    return f"Task: {task_prompt}\n\nCandidate output:\n{output}"


def _build_messages(candidate_output: str, task_prompt: str = TASK_PROMPT) -> list[dict[str, str]]:
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

    messages.append({"role": "user", "content": _candidate_message(TASK_PROMPT, COHERENT_EXAMPLE)})
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

    messages.append({"role": "user", "content": _candidate_message(TASK_PROMPT, COHERENT_TRUNCATED_EXAMPLE)})
    messages.append(
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "coherent": True,
                    "score": 0.7,
                    "summary": "Truncated but coherent quicksort implementation in Rust",
                    "issues": [],
                },
            ),
        },
    )

    for output, verdict in INCOHERENT_EXAMPLES:
        messages.append({"role": "user", "content": _candidate_message(TASK_PROMPT, output)})
        messages.append({"role": "assistant", "content": json.dumps(verdict)})

    messages.append({"role": "user", "content": _candidate_message(task_prompt, candidate_output)})
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


def judge(*, api_key: str, model: str, candidate_output: str, timeout: int, max_retries: int = 3, task_prompt: str = TASK_PROMPT) -> CoherenceVerdict:
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
                "messages": _build_messages(candidate_output, task_prompt),
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
