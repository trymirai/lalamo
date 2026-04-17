#!/bin/bash
# DFlash vs NullSpeculator benchmark on Qwen3-4B / 8B.
#
# Default budget targets ~10 min on a single H100:
#   * DFlash: 50 prompts × 3 datasets (gsm8k, mtbench, math500) per size
#   * Null: 20 prompts × gsm8k per size (speedup denominator only)
# Override with env vars if you need a longer or shorter run.
#
# Run from the repo root after `git pull`. Models must be at
# $HIKETTEI_MODELS/Qwen3-{4B,8B}. DFlash pointer files `dflash_{4b,8b}.bin`
# and the empty `null.bin` are created in the CWD if missing.

set -euo pipefail

HIKETTEI_MODELS=${HIKETTEI_MODELS:-$HOME/hikettei/Models}
N_DFLASH=${N_DFLASH:-50}
N_NULL=${N_NULL:-20}
MAX_TOK=${MAX_TOK:-2048}
WARMUP=${WARMUP:-1}
DFLASH_DATASETS=${DFLASH_DATASETS:-"gsm8k mtbench math500"}

# Pointer files — .bin content is just the HF repo id / empty payload.
for f in dflash_4b.bin dflash_8b.bin null.bin; do
    if [[ ! -s "$f" ]]; then
        case "$f" in
            dflash_4b.bin) printf "z-lab/Qwen3-4B-DFlash-b16" > "$f" ;;
            dflash_8b.bin) printf "z-lab/Qwen3-8B-DFlash-b16" > "$f" ;;
            null.bin)      : > "$f" ;;
        esac
    fi
done

run_cell() {
    local size=$1 drafter=$2 dataset=$3 n=$4
    local target=$HIKETTEI_MODELS/Qwen3-$size
    local bin
    case "$drafter" in
        dflash) bin=dflash_${size,,}.bin ;;
        null)   bin=null.bin ;;
        *) echo "unknown drafter: $drafter"; exit 1 ;;
    esac
    echo ""
    echo "============================================================"
    echo " size=$size  drafter=$drafter  dataset=$dataset  n=$n"
    echo "============================================================"
    uv run lalamo speculator eval "$target" "$bin" \
        --drafter-name "$drafter" --dataset "$dataset" \
        --num-questions "$n" --max-tokens "$MAX_TOK" --warmup "$WARMUP"
}

for size in 4B 8B; do
    for dataset in $DFLASH_DATASETS; do
        run_cell "$size" dflash "$dataset" "$N_DFLASH"
    done
    run_cell "$size" null gsm8k "$N_NULL"
done

echo ""
echo "done."
