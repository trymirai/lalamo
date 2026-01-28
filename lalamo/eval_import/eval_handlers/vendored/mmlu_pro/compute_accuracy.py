"""
Vendored from: https://github.com/TIGER-AI-Lab/MMLU-Pro
Commit: main branch (2024)
License: MIT

Official MMLU-Pro evaluation code for computing accuracy.
"""

import glob
import json
import random
import re
from pathlib import Path


def extract_answer(text: str, level: str) -> str | None:
    if level == "l1":
        pattern = r"answer is \(?([A-J])\)?"
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        else:
            return None
    elif level == "l2":
        pattern = r"answer is \(?([A-J])\)?"
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        else:
            return extract_again(text)
    return None


def extract_again(text: str) -> str | None:
    match = re.search(r".*[aA]nswer:\s*([A-J])", text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)


def extract_final(text: str) -> str | None:
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None


def compute_accuracy_from_dir(directory: Path, level: str = "l2", seed: int = 12345) -> dict[str, dict]:
    """Compute accuracy for all JSON files in directory.

    Args:
        directory: Path to directory containing JSON files
        level: Extraction level ('l1' or 'l2')
        seed: Random seed for fallback selection

    Returns:
        Dictionary mapping category name to accuracy metrics
    """
    random.seed(seed)
    results = {}

    for file_path in directory.glob("*.json"):
        category = file_path.stem
        succ, fail = 0, 0

        with open(file_path) as f:
            entries = json.load(f)
            for e in entries:
                pred = extract_answer(e["model_outputs"], level)
                if pred is None:
                    pred = random.choice(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"])

                if pred == e["answer"]:
                    succ += 1
                else:
                    fail += 1

        total = succ + fail
        accuracy = succ / total if total > 0 else 0.0

        results[category] = {
            "accuracy": accuracy,
            "correct": succ,
            "total": total,
        }

    return results
