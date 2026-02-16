from dataclasses import dataclass

from lalamo.sampling import CompositePolicy, GreedyPolicy, NoTieGreedyPolicy, SamplingPolicy, TemperaturePolicy, TopPPolicy


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
    temperature = 1.0
    top_p = 1.0
    argmax_decoding = False

    policies_to_check: list[SamplingPolicy] = []

    if isinstance(policy, CompositePolicy):
        policies_to_check.extend(policy.policies)
    else:
        policies_to_check.append(policy)

    for p in policies_to_check:
        if isinstance(p, (GreedyPolicy, NoTieGreedyPolicy)):
            argmax_decoding = True
        elif isinstance(p, TemperaturePolicy):
            temperature = p.temperature
        elif isinstance(p, TopPPolicy):
            top_p = p.p

    return FishAudioSamplingParams(
        argmax_decoding=argmax_decoding,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
    )
