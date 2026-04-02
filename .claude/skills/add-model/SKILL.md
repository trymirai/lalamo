---
description: Add a new HuggingFace model to lalamo
match:
  - "add .*/.*"
  - "add .* model"
---

# Adding a model to lalamo

IMPORTANT: Do not propose drafts or ask for confirmation. Write the code directly and
proceed through all steps autonomously. Do not stop to ask questions — make your best
judgment call and keep going. Do not use worktrees or spawn sub-agents — make all
changes directly in the current working directory.

Fetch the model's `config.json` from HuggingFace. Use `model_type` to find a matching
config class in `lalamo/model_import/model_configs/huggingface/`. If none exists, create
one using the closest existing config as a template.

Add a `ModelSpec` in `lalamo/model_import/model_specs/` and register it in `ALL_MODEL_LISTS`.
Add the repo to `tests/model_test_tiers.py`.

## Validate

1. `uv run lalamo convert {repo}`
2. `uv run lalamo chat {repo}` — must produce coherent output.
3. Run pre-commit only on files you changed: `pre-commit run --files <file1> <file2> ...`
4. Run the fast test suite: `uv run pytest -m fast -n=auto`

## Common pitfalls

- Always read the model's HuggingFace README, it may document requirements not captured in config.json.
- Do not add defaults to required config fields — let deserialization fail loudly on missing data.
- Do not handle hypothetical edge cases the model doesn't actually have. If a field is always None/null for all requested models, assert that it's None rather than writing code to handle non-None values.
- Prefer the simplest representation that covers the actual use case. A boolean flag is better than a generic override mechanism when there are only two states.
- Fix bugs at the consumer, not the data layer. Config deserialization should faithfully represent what the model ships — fix bad values where they're used.
- When using an existing config as a template, strip all methods and fields that don't apply to the new model. Do not copy blindly.

## Review

Before finishing, review your changes against these criteria:
- No unnecessary defaults on config fields.
- No dead code paths for features the model doesn't use.
- Bug fixes are at the consumer, not the data layer.
- Simplest possible representation (no generic mechanisms for few-state choices).
