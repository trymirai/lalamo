#!/usr/bin/env python3
"""Benchmark inference_collect_traces on ultrachat_200k test_sft split."""

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".5"

import argparse
import shutil
import time
from pathlib import Path

import polars as pl
from huggingface_hub import hf_hub_download

from lalamo.commands import convert
from lalamo.model_import import REPO_TO_MODEL
from lalamo.models import LanguageModelConfig
from lalamo.speculator.inference import CollectTracesEvent, inference_collect_traces


def download_dataset(cache_dir: Path) -> Path:
    """Download ultrachat_200k test_sft and convert to expected format."""
    parquet_path = cache_dir / "ultrachat_test_sft.parquet"

    if parquet_path.exists():
        print(f"Using cached dataset: {parquet_path}")
        return parquet_path

    print("Downloading HuggingFaceH4/ultrachat_200k test_sft split...")
    # Download the parquet file directly from HF hub
    downloaded = hf_hub_download(
        repo_id="HuggingFaceH4/ultrachat_200k",
        filename="data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet",
        repo_type="dataset",
    )

    # ultrachat_200k has 'messages' column, we need 'conversation'
    df = pl.read_parquet(downloaded)
    df = df.rename({"messages": "conversation"})

    cache_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(parquet_path)
    print(f"Saved dataset to: {parquet_path}")

    return parquet_path


def convert_model(repo: str, output_dir: Path) -> None:
    """Convert a HuggingFace model to lalamo format."""
    if repo not in REPO_TO_MODEL:
        raise ValueError(f"Unknown model repo: {repo}. Available: {list(REPO_TO_MODEL.keys())}")

    model_spec = REPO_TO_MODEL[repo]

    # Remove existing model directory for clean conversion
    if output_dir.exists():
        print(f"Removing existing model directory: {output_dir}")
        shutil.rmtree(output_dir)

    print(f"Converting model: {repo}")
    convert(model_spec, output_dir)
    print(f"Model converted to: {output_dir}")


def run_benchmark(
    model_path: Path,
    dataset_path: Path,
    batch_size: int,
    max_input_length: int,
    max_output_length: int,
    tokens_to_generate: int,
) -> None:
    print(f"Loading model from: {model_path}")
    model = LanguageModelConfig.load_model(model_path)
    print("Model loaded.")

    print(f"Loading dataset from: {dataset_path}")
    from lalamo.data import import_hf_parquet

    dataset = import_hf_parquet(dataset_path)

    tokens_generated = 0
    prefill_tokens = 0
    sequences_processed = 0
    tokens_at_warmup = 0
    prefill_at_warmup = 0
    sequences_at_warmup = 0
    first_batch_start = time.perf_counter()
    first_batch_time: float | None = None
    start_time = time.perf_counter()
    last_report_time = start_time
    warmup_done = False

    def progress_callback(event: CollectTracesEvent) -> None:
        nonlocal tokens_generated, prefill_tokens, sequences_processed, last_report_time
        nonlocal warmup_done, start_time, tokens_at_warmup, prefill_at_warmup, sequences_at_warmup
        nonlocal first_batch_time, first_batch_start
        tokens_generated = event.tokens_generated
        prefill_tokens = event.prefill_tokens
        sequences_processed = event.sequences_processed

        # Record first batch timing (includes JIT compilation)
        if first_batch_time is None and sequences_processed >= 1:
            first_batch_time = time.perf_counter() - first_batch_start
            print(f"  First batch completed in {first_batch_time:.2f}s (includes JIT compilation)")

        # Reset timing after first batch to exclude JIT compilation
        if not warmup_done and sequences_processed >= 1:
            warmup_done = True
            tokens_at_warmup = tokens_generated
            prefill_at_warmup = prefill_tokens
            sequences_at_warmup = sequences_processed
            start_time = time.perf_counter()
            last_report_time = start_time
            print("  (Resetting timer for benchmark measurements)")

        now = time.perf_counter()
        if now - last_report_time >= 5.0:  # Report every 5 seconds
            elapsed = now - start_time
            gen_since_warmup = tokens_generated - tokens_at_warmup
            prefill_since_warmup = prefill_tokens - prefill_at_warmup
            total_tokens = gen_since_warmup + prefill_since_warmup
            tps = gen_since_warmup / elapsed if elapsed > 0 else 0
            print(
                f"  Progress: {prefill_since_warmup:,} prefill + {gen_since_warmup:,} gen = {total_tokens:,} tokens, "
                f"{sequences_processed} seq, "
                f"{tps:.1f} tok/s",
            )
            last_report_time = now

    print("\nStarting benchmark...")
    print(f"  batch_size={batch_size}")
    print(f"  max_input_length={max_input_length}")
    print(f"  max_output_length={max_output_length}")
    print(f"  tokens_to_generate={tokens_to_generate}")
    print()

    # Consume the generator to run inference
    traces = inference_collect_traces(
        model,
        dataset,
        batch_size=batch_size,
        max_input_length=max_input_length,
        max_output_length=max_output_length,
        tokens_to_generate=tokens_to_generate,
        progress_callback=progress_callback,
    )

    # Drain the iterator
    for _ in traces:
        pass

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    gen_since_warmup = tokens_generated - tokens_at_warmup
    prefill_since_warmup = prefill_tokens - prefill_at_warmup
    total_tokens = gen_since_warmup + prefill_since_warmup
    num_sequences = sequences_processed - sequences_at_warmup

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS (excluding first batch / JIT warmup)")
    print("=" * 60)
    if first_batch_time is not None:
        print(f"First batch time (JIT): {first_batch_time:.2f}s")
    print()
    print("Token counts:")
    print(f"  Prefill tokens:       {prefill_since_warmup:,}")
    print(f"  Generation tokens:    {gen_since_warmup:,}")
    print(f"  Total tokens:         {total_tokens:,}")
    print()
    print("Throughput:")
    print(f"  Total time:           {elapsed:.2f}s")
    print(f"  Overall:              {total_tokens / elapsed:.1f} tokens/sec")
    print(f"  Prefill:              {prefill_since_warmup / elapsed:.1f} tokens/sec")
    print(f"  Generation:           {gen_since_warmup / elapsed:.1f} tokens/sec")
    print()
    print("Sequences:")
    print(f"  Sequences processed:  {num_sequences}")
    print(f"  Avg prefill/seq:      {prefill_since_warmup / num_sequences:.1f}")
    print(f"  Avg generation/seq:   {gen_since_warmup / num_sequences:.1f}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark inference_collect_traces")
    parser.add_argument(
        "--repo",
        type=str,
        default="cartesia-ai/Llamba-1B",
        help="HuggingFace model repo to convert (default: cartesia-ai/Llamba-1B)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/Llamba-1B"),
        help="Path to save/load converted model (default: models/Llamba-1B)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (default: 1)",
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=2048,
        help="Max input sequence length (default: 2048)",
    )
    parser.add_argument(
        "--max-output-length",
        type=int,
        default=256,
        help="Max output tokens per sequence (default: 256)",
    )
    parser.add_argument(
        "--tokens-to-generate",
        type=int,
        default=42000,
        help="Total tokens to generate before stopping",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/benchmark"),
        help="Cache directory for dataset (default: .cache/benchmark)",
    )
    args = parser.parse_args()

    dataset_path = download_dataset(args.cache_dir)
    # convert_model(args.repo, args.model_path)

    run_benchmark(
        model_path=args.model_path,
        dataset_path=dataset_path,
        batch_size=args.batch_size,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length,
        tokens_to_generate=args.tokens_to_generate,
    )


if __name__ == "__main__":
    main()
