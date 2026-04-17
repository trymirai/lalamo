#!/bin/bash
# DFlash vs NullSpeculator benchmark on Qwen3-4B / 8B.
#
# Uses the "merged" dataset (gsm8k + mtbench + math500 round-robin with
# source-prefixed categories) so each config prints a single table with
# tok/sec, draft_acc, etc. broken down by source x sub-category.
#
# Default budget targets ~10 min on a single H100:
#   * DFlash: N_DFLASH prompts (default 150) from merged per size
#   * Null:   N_NULL prompts (default 60) from merged per size
# Override with env vars if you need a longer or shorter run.

set -euo pipefail

HIKETTEI_MODELS=${HIKETTEI_MODELS:-$HOME/hikettei/Models}
N_DFLASH=${N_DFLASH:-150}
N_NULL=${N_NULL:-60}
MAX_TOK=${MAX_TOK:-2048}
WARMUP=${WARMUP:-1}
DATASET=${DATASET:-merged}

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
    local size=$1 drafter=$2 n=$3
    local target=$HIKETTEI_MODELS/Qwen3-$size
    local bin
    case "$drafter" in
        dflash) bin=dflash_${size,,}.bin ;;
        null)   bin=null.bin ;;
        *) echo "unknown drafter: $drafter"; exit 1 ;;
    esac
    echo ""
    echo "============================================================"
    echo " size=$size  drafter=$drafter  dataset=$DATASET  n=$n"
    echo "============================================================"
    uv run lalamo speculator eval "$target" "$bin" \
        --drafter-name "$drafter" --dataset "$DATASET" \
        --num-questions "$n" --max-tokens "$MAX_TOK" --warmup "$WARMUP"
}

for size in 4B 8B; do
    run_cell "$size" dflash "$N_DFLASH"
    run_cell "$size" null "$N_NULL"
done

echo ""
echo "done."
