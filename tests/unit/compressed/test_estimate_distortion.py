from lalamo.compressed.data.distortion import DistortionKey, _csv_distortions
from lalamo.compressed.data.estimate_distortion import _default_configs, _estimate_distortion, _spec_from_key


def test_estimate_distortion_default_trellis_keys_rebuild_specs_that_read_their_csv_rows() -> None:
    trellis_keys = [key for key in _default_configs() if key.format_name == "trellis"]

    assert trellis_keys
    for key in trellis_keys:
        assert _spec_from_key(key).distortion == _csv_distortions()[key]


def test_estimate_distortion_samples_at_least_one_trellis_row() -> None:
    key = DistortionKey(format_name="trellis", bits=2, group_size=64, window_bits=16)

    distortion = _estimate_distortion(key, sample_groups=1)

    assert 0 < distortion < 1
