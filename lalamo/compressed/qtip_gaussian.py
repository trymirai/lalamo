"""QTIP bitshift-trellis decode with a Gaussian codebook.

The variant the k3 checkpoint uses on 43 of its 272 leaves: (L, k, V) =
(16, 3, 2), codebook `randn(2**L, V)` from a seeded RNG.

The codebook is unstructured by design -- i.i.d. standard normals, no lattice,
no symmetry, no product form. That matches the incoherence-processed weight
distribution, and it is also why decode is a genuine random gather with no
locality to exploit.

The property that makes a kernel tractable:

    s_{t+1} = ((s_t << kV) mod 2**L) | c_t

means `s_t` is exactly the L bits of the code stream immediately preceding
group t. Nothing is carried between groups, so any group decodes independently
from its own bit offset. Decode is a gather, not a scan -- do not write it as a
sequential state machine.

Reference implementation. Correctness first; the shapes a kernel would want are
in bench/qtip_gaussian_decode_bench.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "QTIPGaussianParams",
    "codes_from_states",
    "build_gaussian_codebook",
    "states_from_codes",
    "decode",
    "recover_codes",
]


@dataclass(frozen=True)
class QTIPGaussianParams:
    """k3 ships L=16, k=3, V=2. kV=6 bits per V=2 weights -> 3.0 bits/weight."""

    L: int = 16
    k: int = 3
    V: int = 2
    seed: int = 1234

    @property
    def kV(self) -> int:
        return self.k * self.V

    @property
    def num_states(self) -> int:
        return 1 << self.L

    @property
    def bits_per_weight(self) -> float:
        return self.kV / self.V

    @property
    def codebook_bytes(self) -> int:
        return self.num_states * self.V * 4

    def __post_init__(self) -> None:
        if self.kV > self.L:
            raise ValueError(f"kV={self.kV} must not exceed L={self.L}")
        if (1 << self.L) % (1 << self.kV):
            raise ValueError("2**L must be divisible by 2**kV")


def build_gaussian_codebook(p: QTIPGaussianParams = QTIPGaussianParams()) -> np.ndarray:
    """[2**L, V] fp32.

    MUST match sbt_og_qtip2.build_table bit for bit, which uses torch's
    Philox/MT stream -- a numpy RNG with the same integer seed produces
    DIFFERENT numbers. Tests import the torch builder and compare; this is the
    documentation of intent, and `from_torch` is the source of truth.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "the canonical codebook is defined by torch's RNG stream; "
            "numpy cannot reproduce it") from exc
    g = torch.Generator(device="cpu").manual_seed(p.seed)
    return torch.randn(1 << p.L, p.V, generator=g, dtype=torch.float32).numpy()


def states_from_codes(init_state: np.ndarray, codes: np.ndarray,
                      p: QTIPGaussianParams) -> np.ndarray:
    """(init_state [rows], codes [rows, steps-1]) -> states [rows, steps].

    Each row carries a FREE L-bit initial state -- that is the `(L - kV)/cols`
    term in QTIP's rate, and omitting it was a real bug here: seeding the
    window with 0 made every recovered state disagree with the fitted stream.

    After the initial state, each group injects kV new bits:

        s_{t+1} = ((s_t << kV) mod 2**L) | c_t

    verified to hold on 100% of transitions in the shipped k3 fit, including
    across the fitter's 64-column block boundaries. So `s_t` is exactly the L
    bits of the stream ending at t, and any group decodes independently from
    its own bit offset -- the kernel is a gather, not a scan.
    """
    rows = init_state.shape[0]
    steps = codes.shape[1] + 1
    mask = (1 << p.L) - 1
    states = np.empty((rows, steps), dtype=np.int64)
    states[:, 0] = init_state
    s = init_state.astype(np.int64)
    for t in range(1, steps):
        s = ((s << p.kV) & mask) | codes[:, t - 1].astype(np.int64)
        states[:, t] = s
    return states


def codes_from_states(states: np.ndarray, p: QTIPGaussianParams
                      ) -> tuple[np.ndarray, np.ndarray]:
    """states -> (init_state [rows], codes [rows, steps-1]). The stored form."""
    return states[:, 0].copy(), (states[:, 1:] & ((1 << p.kV) - 1))


def decode(
    init_state: np.ndarray,     # [rows] the free L-bit start state
    codes: np.ndarray,          # [rows, steps-1] int, each in [0, 2**kV)
    row_scale: np.ndarray,      # [rows] fp32 base scale
    table: np.ndarray,          # [2**L, V] fp32
    p: QTIPGaussianParams = QTIPGaussianParams(),
    gain: np.ndarray | None = None,   # per-row Hessian gain, applied second
) -> np.ndarray:
    """-> [rows, steps * V] fp32."""
    states = states_from_codes(init_state, codes, p)   # bit window
    vals = table[states].astype(np.float32)       # [rows, steps, V] gather
    sc = row_scale[:, None, None].astype(np.float32)
    out = (vals * sc).astype(np.float32)
    if gain is not None:
        out = (out * gain[:, None, None].astype(np.float32)).astype(np.float32)
    return out.reshape(init_state.shape[0], -1)


def _emit(table32: np.ndarray, base_scale: np.float32,
          gain: np.float32) -> np.ndarray:
    """Reproduce the fitter's TWO-STEP emit exactly.

    qtip_quantize returns `table * base_scale`; hessian_gain_refit then
    multiplies by a per-row gain. In fp32 `(t*s)*g != t*(s*g)` for most
    values, so a single combined scale does not reproduce the stored bits.
    This cost a debugging cycle: row 0 verified on all 2,560 groups purely
    because its gain was exactly 1.0, and row 1 (gain 1.015625) failed at the
    first group.
    """
    return ((table32 * base_scale).astype(np.float32) * gain).astype(np.float32)


def solve_row_scale(row: np.ndarray, table: np.ndarray, p: "QTIPGaussianParams",
                    gain: float = 1.0, probes: int = 6,
                    verify_groups: int = 64) -> float:
    """Recover a row's BASE scale (the per-row fp16 std) from its dequant.

    The artifact does not store it: `s_row` in the .pt is the Hessian gain,
    and the pre-feedback weights it was derived from are not saved.

    Propose, then verify. Probe agreement alone picks wrong candidates, so a
    candidate is accepted only if the exact two-step emit reproduces the row's
    first `verify_groups` pairs bit for bit.
    """
    tb = np.ascontiguousarray(table.astype(np.float32))
    g = np.float32(gain)
    pre = (row.astype(np.float32) / g)          # approximate; only for proposing
    cands: list[np.float32] = []
    for t in range(probes):
        pair = pre[2 * t:2 * t + 2]
        if not np.any(pair):
            continue
        r0 = pair[0] / tb[:, 0]
        r1 = pair[1] / tb[:, 1]
        hit = np.where(np.abs(r0 - r1) <= 1e-5 * np.maximum(np.abs(r0), 1e-12))[0]
        for i in hit:
            cands.append(np.float32(r0[i]))
    if not cands:
        raise ValueError("no consistent scale — wrong table or not a QTIP-gauss leaf")

    n = min(verify_groups, len(row) // p.V)
    probe_rows = row[: n * p.V].astype(np.float32).reshape(n, p.V)
    seen: set[bytes] = set()
    for sc in cands:
        kb = sc.tobytes()
        if kb in seen:
            continue
        seen.add(kb)
        scaled = _emit(tb, sc, g)
        lut = {x.tobytes() + y.tobytes() for x, y in scaled}
        if all(probe_rows[i, 0].tobytes() + probe_rows[i, 1].tobytes() in lut
               for i in range(n)):
            return float(sc)
    raise ValueError(
        f"no candidate reproduced the first {n} groups exactly "
        f"({len(seen)} candidate(s) tried)")


def recover_codes(
    dequant: np.ndarray,        # [rows, cols] fp32, as fitted
    table: np.ndarray,          # [2**L, V] fp32
    p: QTIPGaussianParams = QTIPGaussianParams(),
    gains: np.ndarray | None = None,      # the .pt's `s_row` (Hessian gain)
    base_scale: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (states [rows, steps], row_scale [rows]).

    Exact-integer inversion via a hash of the codebook's float bit patterns.
    An earlier version searched only coordinate 0 and then verified; with
    65,536 i.i.d. normals the nearest entry by one coordinate is usually the
    WRONG entry, so it failed on every real leaf. Match on both coordinates.
    """
    rows, cols = dequant.shape
    steps = cols // p.V
    tb = np.ascontiguousarray(table.astype(np.float32))
    lut = {a.tobytes() + b.tobytes(): i for i, (a, b) in enumerate(tb)}

    scales = np.empty(rows, dtype=np.float32)
    gain_arr = np.empty(rows, dtype=np.float32)
    states = np.empty((rows, steps), dtype=np.int64)
    for r in range(rows):
        g = np.float32(1.0 if gains is None else gains[r])
        sc = np.float32(float(base_scale[r]) if base_scale is not None
                        else solve_row_scale(dequant[r], table, p, float(g)))
        scales[r] = sc
        gain_arr[r] = g
        # Hash the SCALED table, not the divided data. fp32 division does not
        # exactly invert the fitter's multiply, so `dequant / sc` differs from
        # the codeword in the last ulp for a minority of groups -- which made
        # an exact-equality lookup fail sporadically (row 0 matched through
        # group 7 and died at group 8). Reproducing the fitter's operation
        # order makes the comparison exact by construction.
        scaled = _emit(tb, sc, g)
        lut_r = {a.tobytes() + b.tobytes(): i for i, (a, b) in enumerate(scaled)}
        row = dequant[r].astype(np.float32).reshape(steps, p.V)
        for t in range(steps):
            key = row[t, 0].tobytes() + row[t, 1].tobytes()
            idx = lut_r.get(key)
            if idx is None:
                raise ValueError(
                    f"row {r} group {t} is not a codebook entry — the leaf does "
                    f"not decode under this (L,k,V,seed)")
            states[r, t] = idx
    return states, scales, gain_arr


def _bench() -> None:
    """Isolate the gather so its memory behaviour can be measured, not argued.

    Shapes are k3's real ones (16480 x 5120 delta_in). The loop below is what a
    Metal kernel has to do; everything else in decode is bookkeeping. States are
    drawn uniformly, which is also what the real access pattern looks like --
    the codebook is i.i.d. normals, so there is no locality either way.

    Run:  python -m lalamo.compressed.qtip_gaussian
    """
    import time

    p = QTIPGaussianParams()
    rows_full, cols = 16480, 5120
    steps = cols // p.V
    print(f"QTIP-gauss  L={p.L} k={p.k} V={p.V}  "
          f"{p.bits_per_weight} bits/weight  states={p.num_states}")
    print(f"full leaf {rows_full}x{cols} = {rows_full * steps / 1e6:.1f}M gathers, "
          f"43 such leaves in k3")
    print("access pattern: uniform over all 65,536 entries — no locality")
    table = build_gaussian_codebook(p)
    rng = np.random.default_rng(7)
    rows = 256
    states = rng.integers(0, 1 << p.L, size=(rows, steps), dtype=np.int64)
    scale = rng.standard_normal(rows).astype(np.float32)
    for dtype in (np.float32, np.float16):
        tb = table.astype(dtype)
        t0 = time.perf_counter()
        out = tb[states].reshape(rows, -1) * scale[:, None]
        dt = time.perf_counter() - t0
        print(f"  table {str(np.dtype(dtype)):>8}  {tb.nbytes / 1024:6.0f} KiB  "
              f"{rows * steps / dt / 1e6:7.1f} M gathers/s  "
              f"{out.nbytes / dt / 1e9:6.2f} GB/s out")


if __name__ == "__main__":
    _bench()
