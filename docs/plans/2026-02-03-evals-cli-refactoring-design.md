# Evals CLI Refactoring Design

**Date:** 2026-02-03
**Status:** Approved
**Type:** Refactoring

## Overview

Refactor `lalamo/evals/cli.py` by splitting it into domain-specific CLI modules while maintaining backward compatibility with existing command names and behavior.

## Current State

Single monolithic file at `lalamo/evals/cli.py` (~380 lines) containing:
- Dataset conversion command and helpers (CliEvalConversionCallbacks, EvalParser)
- Inference command
- Benchmark command
- Shared utilities (console, DEFAULT_DATASETS_DIR)

## Target Architecture

### File Structure
```
lalamo/evals/
├── cli.py                    # Main entry point, eval_app, imports commands
├── dataset/
│   ├── __init__.py
│   └── cli.py               # convert_dataset_command + helpers
├── inference/
│   ├── __init__.py
│   └── cli.py               # infer_command
└── benchmark/
    ├── __init__.py
    └── cli.py               # benchmark_command
```

### Component Responsibilities

**Main `cli.py` (~50 lines):**
- Creates `eval_app = Typer()` instance
- Defines shared utilities: `console`, `DEFAULT_DATASETS_DIR`
- Imports command functions from submodules
- Registers commands with decorators

**`dataset/cli.py` (~140 lines):**
- `CliEvalConversionCallbacks` - progress bars and user prompts
- `EvalParser` - custom argument type for eval repo validation
- `convert_dataset_command` - main command function

**`inference/cli.py` (~105 lines):**
- `infer_command` - runs model inference on evaluation datasets
- Console output and progress display logic

**`benchmark/cli.py` (~135 lines):**
- `benchmark_command` - evaluates model predictions
- Result loading and display logic

## Design Decisions

**Option B: Flat CLI with imported commands**
- Maintains backward compatibility (command names unchanged)
- Commands remain: `lalamo eval convert-dataset`, `lalamo eval infer`, `lalamo eval benchmark`
- Alternative (rejected): Subgroups would change to `lalamo eval dataset convert`

**Keep it simple:**
- No further extraction of display/callback logic
- Commands are thin orchestration layers - business logic already in other modules
- Avoids over-engineering for relatively simple command functions

## Implementation Plan

1. Create directory structure: `dataset/`, `inference/`, `benchmark/` with `__init__.py`
2. Create `dataset/cli.py` - move `CliEvalConversionCallbacks`, `EvalParser`, `convert_dataset_command`
3. Create `inference/cli.py` - move `infer_command`
4. Create `benchmark/cli.py` - move `benchmark_command`
5. Update main `cli.py` - keep shared utilities, import and register commands
6. Verify: Run `lalamo eval --help` and each subcommand help

## Risks & Mitigation

- **Import errors:** Carefully track dependencies when moving code
- **Shared utilities:** Ensure console and constants accessible from submodules
- **Testing:** Manual verification via CLI help and command execution

## Success Criteria

- All three commands work identically to before
- `lalamo eval --help` lists all commands
- Each command's help text displays correctly
- Zero behavior changes - pure code organization refactoring
