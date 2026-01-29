import struct
from array import array
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import chain, repeat, tee
from math import exp
from typing import Self

import xxhash

from .common import Speculator


# This is not exactly randomly distributed if size is not a power of two.
# Shouldn't matter in practice though because size <<< 2**64
def seqhash(tokens: Iterable[int], size: int) -> int:
    tokens = list(tokens)
    assert size <= 2**64
    if len(tokens) > 0:
        packed = struct.pack("<" + "I" * len(tokens), *tokens)
    else:
        packed = b""

    return xxhash.xxh3_64_intdigest(packed) % size


def padded_sliding_window(seq: Iterable[int], size: int, pad: int) -> Iterable[tuple[int, ...]]:
    seqs = tee(seq, size)
    pads = tuple(repeat(pad, size - i) for i in range(size))
    padded_seqs = tuple(chain(pad, seq) for pad, seq in zip(pads, seqs, strict=True))
    return zip(*padded_seqs, strict=False)


def softmax(logits: Iterable[float]) -> list[float]:
    logits = list(logits)
    log_max = max(logits)
    exp_logs = [exp(logit - log_max) for logit in logits]
    exp_log_sum = sum(exp_logs)
    return [exp_log / exp_log_sum for exp_log in exp_logs]


def online_mean(old_mean: float, sample: float, new_count: int) -> float:
    return old_mean + (sample - old_mean) / new_count


def update_probs(old_mean: dict[int, float], sample: dict[int, float], new_count: int, top_k: int) -> dict[int, float]:
    all_keys = set(old_mean.keys()).union(sample.keys())

    new_probs_all = {k: online_mean(old_mean.get(k, 0), sample.get(k, 0), new_count) for k in all_keys}

    new_probs_top_k = dict(sorted(new_probs_all.items(), key=lambda x: (-x[1], x[0]))[:top_k])

    new_probs_sum = sum(new_probs_top_k.values())

    new_probs_norm = {k: v / new_probs_sum for k, v in new_probs_top_k.items()}

    return new_probs_norm


@dataclass(frozen=True, eq=False)
class NGramSpeculator(Speculator):
    hashtable_size: int
    ngram_k: int
    ngram_n: int
    ngram_pad: int

    # "interior mutability" here.
    # dataclass field default_factory doesn't pass self so None here + post init as a workaround
    ngram_keys: array[int]
    ngram_values: array[float]
    ngram_counts: array[int]

    def __post_init__(self) -> None:
        if not self.hashtable_size > 0:
            raise ValueError(f"{self.hashtable_size=} (must be > 0)")
        if not self.ngram_k > 0:
            raise ValueError(f"{self.ngram_k=} (must be > 0)")
        if not self.ngram_n > 0:
            raise ValueError(f"{self.ngram_n=} (must be > 0)")

    @classmethod
    def new(cls, hashtable_size: int, ngram_k: int, ngram_n: int, ngram_pad: int = 2**32 - 1) -> Self:
        return cls(
            hashtable_size,
            ngram_k,
            ngram_n,
            ngram_pad,
            array("I", range(ngram_k)) * hashtable_size,
            array("f", repeat(0, ngram_k)) * hashtable_size,
            array("I", [0]) * hashtable_size,
        )

    def train(self, token_ids: Iterable[int], token_logits: Iterable[dict[int, float]]) -> None:
        ngram_ctx = self.ngram_n - 1
        if ngram_ctx > 0:
            contexts = padded_sliding_window(token_ids, ngram_ctx, self.ngram_pad)
        else:
            contexts = repeat(())

        for ctx, cur_logits in zip(contexts, token_logits, strict=False):
            ngram_keys, ngram_values, ngram_counts = self._seq_slice(ctx)

            ngram_counts[0] = new_count = ngram_counts[0] + 1

            old_mean = dict(zip(ngram_keys, ngram_values, strict=True))
            sample = dict(zip(cur_logits.keys(), softmax(cur_logits.values()), strict=True))

            new_probs = update_probs(old_mean, sample, new_count, self.ngram_k)

            ngram_keys[:] = array("I", new_probs.keys())
            ngram_values[:] = array("f", new_probs.values())

    def probs(self, seq: Iterable[int]) -> dict[int, float]:
        ngram_keys, ngram_values, _ = self._seq_slice(seq)
        return dict(zip(ngram_keys, ngram_values, strict=True))

    # python < 3.13 doesn't support memoryview[T], but if T is not specified typechecker incorrectly assumes it's int
    def _seq_slice(self, seq: Iterable[int]) -> tuple["memoryview[int]", "memoryview[float]", "memoryview[int]"]:
        seq = list(seq)
        ngram_ctx = self.ngram_n - 1
        if ngram_ctx > 0:
            padded_seq = [*repeat(self.ngram_pad, max(ngram_ctx - len(seq), 0)), *seq[-ngram_ctx:]]
        else:
            padded_seq = []

        seq_hash = seqhash(padded_seq, self.hashtable_size)
        idx_start = seq_hash * self.ngram_k
        idx_end = seq_hash * self.ngram_k + self.ngram_k

        return (
            memoryview(self.ngram_keys)[idx_start:idx_end],
            memoryview(self.ngram_values)[idx_start:idx_end].cast("c").cast("f"), # noop cast to make typechecker happy
            memoryview(self.ngram_counts)[seq_hash : (seq_hash + 1)],
        )

    def serialize(self) -> bytes:
        hdr = struct.pack("<4I", self.hashtable_size, self.ngram_k, self.ngram_n, self.ngram_pad)
        return hdr + bytes(self.ngram_keys) + bytes(self.ngram_values) + bytes(self.ngram_counts)

    @classmethod
    def deserialize(cls, blob: bytes) -> Self:
        offset = 16

        hashtable_size, ngram_k, ngram_len, ngram_pad = struct.unpack("<4I", blob[:offset])

        ngram_kv_len = 4 * ngram_k * hashtable_size

        ngram_keys = array("I", blob[offset : offset + ngram_kv_len])
        offset += ngram_kv_len

        ngram_values = array("f", blob[offset : offset + ngram_kv_len])
        offset += ngram_kv_len

        ngram_counts_len = 4 * hashtable_size
        ngram_counts = array("I", blob[offset : offset + ngram_counts_len])
        offset += ngram_counts_len

        return cls(hashtable_size, ngram_k, ngram_len, ngram_pad, ngram_keys, ngram_values, ngram_counts)
