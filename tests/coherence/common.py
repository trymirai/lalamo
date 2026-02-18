import json
import logging
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests

from tests.conftest import strip_ansi_escape

DEFAULT_JUDGE_MODEL = "meta-llama/llama-3.3-70b-instruct"

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CoherenceVerdict:
    coherent: bool
    score: float
    summary: str
    issues: tuple[str, ...]


@dataclass(frozen=True)
class FewShotExample:
    output: str
    verdict: CoherenceVerdict


def _load_examples(path: Path | str = Path(__file__).parent / "prompt_spec.toml") -> tuple[str, list[FewShotExample]]:
    with open(path, "rb") as f:
        data = tomllib.load(f)

    task_prompt: str = data["task_prompt"]

    examples = [
        FewShotExample(
            output=e["output"],
            verdict=CoherenceVerdict(coherent=True, score=e["score"], summary=e["summary"], issues=tuple(e["issues"])),
        )
        for e in data.get("coherent", [])
    ] + [
        FewShotExample(
            output=e["output"],
            verdict=CoherenceVerdict(coherent=False, score=e["score"], summary=e["summary"], issues=tuple(e["issues"])),
        )
        for e in data.get("incoherent", [])
    ]

    return task_prompt, examples


TASK_PROMPT, _EXAMPLES = _load_examples()


def extract_output(raw: str) -> str:
    cleaned = strip_ansi_escape(raw)
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
                " Coherence is about readability, relevance, and non-degenerate text,"
                " not factual correctness. If an answer is understandable and answers"
                " the prompt, treat it as coherent even when it contains factual errors,"
                " invented attributions, or extra details."
                " For each candidate output, return ONLY a JSON object with keys:"
                ' "coherent" (bool), "score" (0.0-1.0), "summary" (short string),'
                ' "issues" (array of strings).'
                " Mark as incoherent if the output is off-topic, unreadable,"
                " has repeated/degenerate tokens, or eos spam. Please be somewhat lax:"
                " we might be checking you against stupid models, which could be genuinely "
                " incapable of generating working code. Still, consider their output coherent."
                " If the output is merely too long, somewhat inaccurate, or mildly rambly"
                " but still relevant and readable, mark coherent=True with a lower score."
            ),
        },
    ]

    for example in _EXAMPLES:
        messages.append({"role": "user", "content": _candidate_message(TASK_PROMPT, example.output)})
        messages.append({"role": "assistant", "content": json.dumps(asdict(example.verdict))})

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
        issues = tuple(str(i) for i in raw_issues)
    elif raw_issues is None:
        issues = ()
    else:
        issues = (str(raw_issues),)

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


def judge(
    *,
    api_key: str,
    model: str,
    candidate_output: str,
    timeout: int,
    max_retries: int = 3,
    task_prompt: str = TASK_PROMPT,
) -> CoherenceVerdict:
    if max_retries < 1:
        raise ValueError(f"max_retries must be >= 1, got {max_retries}")
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
    raise last_error
