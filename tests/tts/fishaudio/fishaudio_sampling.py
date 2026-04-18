from dataclasses import dataclass

from lalamo.sampling import (
    LogitTransform,
    SamplingPipeline,
    TemperaturePolicy,
    TopPPolicy,
)


@dataclass(frozen=True)
class FishAudioSamplingParams:
    argmax_decoding: bool
    top_p: float
    temperature: float
    repetition_penalty: float


def sampling_params_from_policy(
    policy: SamplingPipeline | None,
    repetition_penalty: float = 1.0,
) -> FishAudioSamplingParams:
    """Convert a SamplingPipeline to FishAudioSamplingParams for PyTorch wrapper compatibility."""
    if policy is None:
        return FishAudioSamplingParams(
            argmax_decoding=True,
            top_p=0.0,
            temperature=0.0,
            repetition_penalty=0.0,
        )
    temperature = 1.0
    top_p = 1.0
    argmax_decoding = False

    policies_to_check: list[LogitTransform] = list(policy.stages)

    for p in policies_to_check:
        if isinstance(p, TemperaturePolicy):
            if p.temperature == 0.0:
                argmax_decoding = True
            else:
                temperature = p.temperature
        elif isinstance(p, TopPPolicy):
            top_p = p.p

    return FishAudioSamplingParams(
        argmax_decoding=argmax_decoding,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
    )
