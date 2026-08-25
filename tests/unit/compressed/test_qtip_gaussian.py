"""Bit-exactness against a real k3 leaf.

Not approximate. Every fitted value is exactly `(table[state]*scale)*gain`, so
a wrong codebook, scale convention, gain order or transition fails outright
rather than degrading. That is the property a Metal port must preserve, so it
is the property asserted here.

Fixture: sbt-air-qtip-k3-fit-v1/l10_delta_in_qtipsym.pt (box-4), 16480x5120.
Verified: 24 rows x 2560 groups round-trip with max abs error 0.0.
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

FIT = os.environ.get(
    "QTIP_K3_FIT",
    "/data/ry2009/naq_research_20260726/sbt-air-qtip-k3-fit-v1/l10_delta_in_qtipsym.pt",
)
P = QTIPGaussianParams(L=16, k=3, V=2, seed=1234)
pytestmark = pytest.mark.skipif(not os.path.exists(FIT), reason="k3 fixture absent")


def _leaf(n_rows: int):
    import torch

    d = torch.load(FIT, map_location="cpu", weights_only=False)
    return (d["dequant_rot"][:n_rows].float().numpy(),
            d["s_row"][:n_rows].float().numpy())      # s_row is the GAIN


def test_codebook_is_512_kib():
    t = build_gaussian_codebook(P)
    assert t.shape == (1 << 16, 2) and t.dtype == np.float32
    assert P.codebook_bytes == 512 * 1024          # the number @ccy cares about
    assert P.bits_per_weight == 3.0


def test_transition_is_a_sliding_bit_window():
    """The claim the whole kernel design rests on.

    Measured on the shipped fit: holds for 100% of transitions, including
    across the fitter's 64-column block boundaries. If it were false, decode
    would be a serial scan.
    """
    dq, gains = _leaf(4)
    st, _, _ = recover_codes(dq, build_gaussian_codebook(P), P, gains=gains)
    base = 1 << (P.L - P.kV)
    for r in range(st.shape[0]):
        s = st[r]
        assert all((s[i + 1] >> P.kV) == (s[i] & (base - 1))
                   for i in range(len(s) - 1)), f"row {r}"


def test_roundtrip_is_bit_exact():
    dq, gains = _leaf(24)
    table = build_gaussian_codebook(P)
    states, scale, gain = recover_codes(dq, table, P, gains=gains)
    init, codes = codes_from_states(states, P)

    assert np.array_equal(states_from_codes(init, codes, P), states)
    back = decode(init, codes, scale, table, P, gain=gain)
    assert np.array_equal(back, dq), "decode is not bit-exact"
    assert np.abs(back - dq).max() == 0.0


def test_rate_matches_the_qtip_formula():
    dq, _ = _leaf(1)
    cols = dq.shape[1]
    steps = cols // P.V
    bpw = (P.L + (steps - 1) * P.kV + 16) / cols     # init state + codes + fp16 scale
    assert abs(bpw - (P.k + (P.L - P.kV + 16) / cols)) < 1e-9
    assert 3.005 < bpw < 3.006


def test_base_scale_is_not_stored_and_must_be_solved():
    """Regression guard on a packaging gap.

    The artifact stores the Hessian gain as `s_row` (~1.0) but NOT the per-row
    fp16 std the trellis quantized against, and the pre-feedback weights it
    came from are not saved either. recover_codes solves for it; a real packed
    artifact must store it instead.
    """
    dq, gains = _leaf(4)
    _, scale, gain = recover_codes(dq, build_gaussian_codebook(P), P, gains=gains)
    assert np.allclose(gain, gains)
    assert (scale < 0.1).all(), "base scale should be ~1e-2, not the ~1.0 gain"
    assert len(set(scale.tolist())) > 1, "scale is per-row, not global"
