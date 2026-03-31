# Quant Distill Merge Checklist

This is the only checklist I am using for the branch. I will mark items `x` only when the code and the relevant local checks are done. The merge-ready call still depends on the H100 items at the bottom.

## Core correctness

- [x] `lalamo/distillation.py`: reject impossible trace rows and remove fake clamp / zero-fill paths
- [x] `lalamo/distill_runner.py`: fix dataset accounting, tokenized example selection, and sharding semantics
- [x] `lalamo/modules/embedding.py`: fix wrong metadata, wrong shape annotation, bad error shape, and missing MLX import invariants
- [x] `lalamo/modules/linear.py`: add real group-size invariants and make legacy RHT checkpoint import clean
- [x] `lalamo/modules/common.py` + `lalamo/model_import/loaders/common.py`: simplify metadata/sharding handling and fail loudly on bad state
- [x] `lalamo/quantization.py`: remove dead enum state and make stochastic rounding sample in `float32`
- [x] `lalamo/model_import/model_specs/lfm2.py`: make vendor/repo truth consistent
- [x] `lalamo/model_import/loaders/huggingface.py` + `lalamo/model_import/model_configs/huggingface/lfm2.py`: keep quantized embedding/config loading honest

## Branch cleanup

- [x] Delete `scripts/run_distillation_experiment.py` if it is still wrapper residue
- [x] Simplify `scripts/run_distillation_sweep.py` until it is not pretending to be a giant configurable search framework
- [x] Decide whether `scripts/convert_tied_embedding_to_mlx_quantized.py` belongs in this PR; if it stays, keep it tiny and obvious
- [x] Shrink `lalamo/main.py` distill CLI and callback state
- [x] Cut hidden-cost APIs and dead branches that make the core path harder to read
- [x] Do a branch-wide cleanup pass focused only on skimmability, state reduction, and removing non-essential code

## Tests

- [x] Rewrite weak `distill_runner` tests so they prove real behavior
- [x] Rewrite the worst `distillation` tests so they stop depending on tree layout and helper machinery
- [x] Cut plumbing-only CLI tests and keep only behavior that matters
- [x] Tighten quantization / embedding / loader tests around the real risky paths

## Verification

- [x] Run focused local unit tests for the touched files
- [x] Run real H100 CLI smoke for simple distill mode and confirm KL goes down
- [x] Run real H100 CLI smoke for advanced distill mode and confirm KL goes down
- [x] Re-check final diff shape against the cleanup bar before calling the branch merge-ready
