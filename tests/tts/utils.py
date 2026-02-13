"""Utilities for generating pseudo-voice signals for audio codec testing."""

import numpy as np
import torch
from jax import numpy as jnp
from scipy.interpolate import CubicSpline

from lalamo.modules.torch_interop import torch_to_jax


def prepare_state_dict_for_lalamo_loaders(
    state_dict: dict[str, torch.Tensor],
    prefix: str = "",
) -> dict[str, jnp.ndarray]:
    """Convert PyTorch state_dict to JAX arrays with optional key prefix for Lalamo loaders.

    Args:
        state_dict: PyTorch state_dict mapping string keys to tensors.
        prefix: Optional prefix to prepend to all keys.

    Returns:
        Dictionary mapping string keys to JAX arrays, compatible with ParameterPath lookups.
    """
    result = {}
    for key, tensor in state_dict.items():
        full_key = f"{prefix}.{key}" if prefix else key
        result[full_key] = torch_to_jax(tensor.detach())
    return result

_rng = np.random.default_rng()


def pink_noise_via_fft(n: int) -> np.ndarray:
    """Generate pink noise via FFT with 1/f spectrum."""
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1]  # avoid DC
    magnitude = 1 / np.sqrt(freqs)
    phase = _rng.uniform(0, 2 * np.pi, len(freqs))
    spectrum = magnitude * np.exp(1j * phase)
    return np.fft.irfft(spectrum, n)


def generate_envelope_with_formants(fir_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a spectral envelope filter with formant-like characteristics.

    Returns:
        Tuple of (fir_coefficients, envelope)
    """
    envelope = np.sin(2 * np.pi * 2 * np.arange(fir_size) / fir_size)
    envelope = np.abs(envelope) * np.linspace(1.0, 0.05, fir_size)
    envelope = np.convolve(envelope, np.ones(int(fir_size * 0.15)), mode="same")
    envelope = envelope / max(envelope)

    h = np.fft.ifft(envelope).real
    h = np.fft.fftshift(h)
    fir_coefs = h * np.hamming(fir_size)
    return fir_coefs, envelope


def generate_spline_curve(
    points: np.ndarray,
    num_output: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a cubic spline curve passing through given points.

    Args:
        points: Array of shape (n, 2) with x,y coordinates the curve must cross
        num_output: Number of points in output curve

    Returns:
        Tuple of (x_values, y_values) arrays of the interpolated curve
    """
    points = np.asarray(points)
    x_pts = points[:, 0]
    y_pts = points[:, 1]

    cs = CubicSpline(x_pts, y_pts)
    x_out = np.linspace(x_pts.min(), x_pts.max(), num_output)
    y_out = cs(x_out)
    assert isinstance(y_out, np.ndarray)

    return x_out, y_out


def generate_harmonic_row(
    fs: float, n_points: int, f0: float, num_harmonics: int = 50, flat_frequency: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a harmonic signal with varying f0 and formant filtering.

    Args:
        fs: Sample rate in Hz
        n_points: Number of samples to generate
        f0: Fundamental frequency in Hz
        num_harmonics: Maximum number of harmonics

    Returns:
        Tuple of (filtered_signal, raw_harmonics)
    """
    f0_ref_points = [0.8, 0.95, 1.05, 1.15, 1.25]
    np.random.shuffle(f0_ref_points)
    f0_for_interpolation = np.stack(([0.0, 0.25, 0.5, 0.75, 1.0], f0_ref_points), axis=1)

    if not flat_frequency:
        _, f0_track = generate_spline_curve(f0_for_interpolation, n_points)
    else:
        f0_track = np.ones(n_points)

    if f0 * num_harmonics > fs / 2:
        num_harmonics = int((fs / 2) / f0) - 1

    timesteps = np.arange(n_points, dtype=float) / fs
    harmonics = np.zeros((num_harmonics, n_points))
    for k in range(num_harmonics):
        cur_freq = f0 * (k + 1)
        harmonics[k] = np.sin(2 * np.pi * cur_freq * f0_track * timesteps)

    harmonic_row = np.sum(harmonics, axis=0) / num_harmonics
    spec_envelope_fir, _ = generate_envelope_with_formants(256)
    harmonic_row = np.convolve(harmonic_row, spec_envelope_fir, "same")

    return harmonic_row, harmonics


def generate_pseudo_voice_signal(
    fs: int = 44100,
    duration_sec: float = 2.0,
    f0: float = 200.0,
    voiced_segments: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Generate a pseudo-voice signal with voiced and unvoiced segments.

    Creates a signal with pink noise background and harmonic voiced segments,
    useful for smoke testing neural audio codecs.

    Args:
        fs: Sample rate in Hz
        duration_sec: Signal duration in seconds
        f0: Fundamental frequency for voiced segments
        voiced_segments: List of (start, end) tuples as fractions of duration.
                        Defaults to [(0.1, 0.3), (0.6, 0.9)]

    Returns:
        Generated pseudo-voice signal
    """
    # cross-fading margins when transition betwenn voiced-unvoiced segments
    margin_n_samples = 200
    half_margin = margin_n_samples // 2

    if voiced_segments and int(np.array(voiced_segments).min() * fs) <= half_margin:
        raise ValueError("Specified signal duration is two short for given combination of fs and voice segment.")

    if voiced_segments is None:
        voiced_segments = [(0.1, 0.3), (0.6, 0.9)]

    siglen = int(fs * duration_sec)
    signal = pink_noise_via_fft(siglen) * 3

    voice_seg_borders = (np.array(voiced_segments) * siglen).astype(np.int32)

    for seg_brd in voice_seg_borders:
        # Attenuate noise in voiced region
        signal[seg_brd[0] : seg_brd[1]] *= 0.01

        # Generate voiced segment with margins for crossfade
        voiced_length = seg_brd[1] - seg_brd[0] + margin_n_samples
        voiced_seg, _ = generate_harmonic_row(fs, voiced_length, f0, 50)

        # Apply fade in/out
        voiced_seg[:half_margin] *= np.linspace(0.0, 1.0, half_margin)
        voiced_seg[-half_margin:] *= np.linspace(1.0, 0.0, half_margin)

        signal[seg_brd[0] - half_margin : seg_brd[1] + half_margin] = voiced_seg

    return signal
