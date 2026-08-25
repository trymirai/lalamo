"""Bit-exactness against real k3 leaves from both shipped checkpoints.

Not approximate. Every fitted value is exactly `(table[state]*scale)*gain`, so
a wrong codebook, scale convention, gain order or transition fails outright
rather than degrading. That is the property a Metal port must preserve.

Qwen3.6 k3 and Qwen3.8 t0data-k3 decode under the SAME (L,k,V,seed) and the
same table. T0 is a per-row bf16 multiply applied at compose time, after
dequant -- it never enters the trellis, so it does not fork the decode path.
"""

import os

import numpy as np
import pytest

from lalamo.compressed.qtip_gaussian import (
    QTIPGaussianParams,
    build_gaussian_codebook,
    codes_from_states,
    decode,
    recover_codes,
    states_from_codes,
)

_R = "/data/ry2009/naq_research_20260726"
FIXTURES = [
    ("qwen3.6-k3", os.environ.get(
        "QTIP_K3_FIT_Q36", f"{_R}/sbt-air-qtip-k3-fit-v1/l10_delta_in_qtipsym.pt")),
    ("qwen3.8-k3", os.environ.get(
        "QTIP_K3_FIT_Q38", f"{_R}/sbt-air-q38-k3-fit-v1/l0_delta_in_qtipsym.pt")),
    ("qwen3.8-k3-mlp", os.environ.get(
        "QTIP_K3_FIT_Q38_MLP", f"{_R}/sbt-air-q38-k3-fit-v1/l0_mlp_down_qtipsym.pt")),
]
P = QTIPGaussianParams(L=16, k=3, V=2, seed=1234)
present = [f for f in FIXTURES if os.path.exists(f[1])]
pytestmark = pytest.mark.skipif(not present, reason="k3 fixtures absent")
leaf_param = pytest.mark.parametrize(
    "path", [f[1] for f in present], ids=[f[0] for f in present])


def _leaf(path: str, n_rows: int):
    import torch

    d = torch.load(path, map_location="cpu", weights_only=False)
    return (d["dequant_rot"][:n_rows].float().numpy(),
            d["s_row"][:n_rows].float().numpy())      # s_row is the GAIN


def test_codebook_is_512_kib():
    t = build_gaussian_codebook(P)
    assert t.shape == (1 << 16, 2) and t.dtype == np.float32
    assert P.codebook_bytes == 512 * 1024          # the number that decides metal feasibility
    assert P.bits_per_weight == 3.0


@leaf_param
def test_transition_is_a_sliding_bit_window(path):
    """The claim the whole kernel design rests on.

    Holds for 100% of transitions on both checkpoints, including across the
    fitter's 64-column block boundaries. If it were false, decode would be a
    serial scan instead of a gather.
    """
    dq, gains = _leaf(path, 4)
    st, _, _ = recover_codes(dq, build_gaussian_codebook(P), P, gains=gains)
    base = 1 << (P.L - P.kV)
    for r in range(st.shape[0]):
        s = st[r]
        assert all((s[i + 1] >> P.kV) == (s[i] & (base - 1))
                   for i in range(len(s) - 1)), f"row {r}"


@leaf_param
def test_roundtrip_is_bit_exact(path):
    dq, gains = _leaf(path, 8)
    table = build_gaussian_codebook(P)
    states, scale, gain = recover_codes(dq, table, P, gains=gains)
    init, codes = codes_from_states(states, P)

    assert np.array_equal(states_from_codes(init, codes, P), states)
    back = decode(init, codes, scale, table, P, gain=gain)
    assert np.array_equal(back, dq), "decode is not bit-exact"
    assert np.abs(back - dq).max() == 0.0


@leaf_param
def test_rate_matches_the_qtip_formula(path):
    """Rate is k plus the per-row header amortized over the row.

    Header is L - kV + 16 = 26 bits: the free initial state plus an fp16 scale.
    A 5120-column leaf lands at 3.0051, a 17408-column one at 3.0015.
    """
    dq, _ = _leaf(path, 1)
    cols = dq.shape[1]
    steps = cols // P.V
    bpw = (P.L + (steps - 1) * P.kV + 16) / cols
    assert abs(bpw - (P.k + (P.L - P.kV + 16) / cols)) < 1e-9
    assert 3.0 < bpw < 3.01


@leaf_param
def test_base_scale_is_not_stored_and_must_be_solved(path):
    """Regression guard on a packaging gap.

    The artifact stores the Hessian gain as `s_row` (~1.0) but NOT the per-row
    fp16 std the trellis quantized against, and the pre-feedback weights it
    came from are not saved either. recover_codes solves for it; a real packed
    artifact must store it instead.
    """
    dq, gains = _leaf(path, 4)
    _, scale, gain = recover_codes(dq, build_gaussian_codebook(P), P, gains=gains)
    assert np.allclose(gain, gains)
    assert (scale < 0.1).all(), "base scale should be ~1e-2, not the ~1.0 gain"


def test_both_checkpoints_share_one_decoder():
    """The reason this PR exists: one kernel has to cover both checkpoints.

    Same L, k, V, seed and the same 512 KiB table decode Qwen3.6 k3 and
    Qwen3.8 t0data-k3. Only the per-row scales differ.
    """
    if len(present) < 2:
        pytest.skip("need at least two checkpoints present")
    table = build_gaussian_codebook(P)
    for _, path in present:
        dq, gains = _leaf(path, 4)
        states, scale, gain = recover_codes(dq, table, P, gains=gains)
        init, codes = codes_from_states(states, P)
        assert np.array_equal(decode(init, codes, scale, table, P, gain=gain), dq)
