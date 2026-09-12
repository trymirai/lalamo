from lalamo.compressed.data.distortion import DistortionKey, _csv_distortions
from lalamo.compressed.data.estimate_distortion import _default_configs, _estimate_distortion, _spec_from_key
from lalamo.compressed.trellis import TrellisSpec


def test_estimate_distortion_default_trellis_keys_are_in_the_csv_and_rebuild_their_specs() -> None:
    trellis_keys = [key for key in _default_configs() if key.format_name == "trellis"]

    assert len(trellis_keys) == 16
    assert set(trellis_keys) <= set(_csv_distortions())
    for key in trellis_keys:
        spec = _spec_from_key(key)
        assert isinstance(spec, TrellisSpec)
        assert (spec.bits, spec.window_bits, spec.restart_columns) == (key.bits, 16, key.group_size)


def test_estimate_distortion_samples_at_least_one_trellis_row() -> None:
    key = DistortionKey(format_name="trellis", bits=2, group_size=64, window_bits=16)

    distortion = _estimate_distortion(key, sample_groups=1)

    assert 0 < distortion < 1
