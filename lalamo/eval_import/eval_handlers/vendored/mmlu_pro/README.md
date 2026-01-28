# MMLU-Pro Official Evaluation Code

**Source:** https://github.com/TIGER-AI-Lab/MMLU-Pro
**License:** MIT
**Vendored from:** main branch (2024)

This code is vendored from the official MMLU-Pro repository to ensure
evaluation consistency with the official leaderboard.

## Files

- `compute_accuracy.py`: Official accuracy computation logic with 3-tier regex extraction

## Changes from Original

- Wrapped in function `compute_accuracy_from_dir()` for programmatic use
- Added type hints
- Removed command-line argument parsing (now accepts Path parameter)
- Returns structured dictionary instead of printing to stdout
