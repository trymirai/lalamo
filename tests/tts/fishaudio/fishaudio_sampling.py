from dataclasses import dataclass

import jax
import numpy as np
from jaxtyping import Array

from lalamo.sampling import SamplingPolicy


@dataclass(frozen=True)
class FishAudioSamplingParams:
    argmax_decoding: bool
    top_p: float
    temperature: float
    repetition_penalty: float


def sampling_params_from_policy(
    policy: SamplingPolicy | None,
    repetition_penalty: float = 1.0,
) -> FishAudioSamplingParams:
    """Convert a SamplingPolicy to FishAudioSamplingParams for PyTorch wrapper compatibility."""
    if policy is None:
        return FishAudioSamplingParams(
            argmax_decoding=True,
            top_p=0.0,
            temperature=0.0,
            repetition_penalty=0.0,
        )
    temperature = _scalar_float(policy.temperature)
    top_p = _scalar_float(policy.top_p)
    argmax_decoding = temperature == 0.0

    return FishAudioSamplingParams(
        argmax_decoding=argmax_decoding,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
    )


def _scalar_float(value: Array) -> float:
    if value.shape != ():
        raise ValueError("FishAudio sampling params expect a scalar sampling policy.")
    return float(np.asarray(jax.device_get(value)).item())
