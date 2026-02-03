from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from .common import InferenceConfig

if TYPE_CHECKING:
    from jax._src.stages import Compiled

    from .language_model import ForwardPassConfig, GenerationConfig, LanguageModel

_compile_cache: dict[tuple[GenerationConfig | None, InferenceConfig, ForwardPassConfig | None], Compiled] = {}


def compile_generate_tokens(
    model: LanguageModel,
    generation_config: GenerationConfig | None = None,
    inference_config: InferenceConfig = InferenceConfig(),  # noqa: B008
    *,
    forward_pass_config: ForwardPassConfig | None = None,
) -> Compiled:
    key = (generation_config, inference_config, forward_pass_config)
    if key not in _compile_cache:
        generate_tokens_fn = functools.partial(
            model.generate_tokens,
            generation_config=generation_config,
            max_output_length=inference_config.max_output_length,
            num_top_logits_to_return=inference_config.num_top_logits_to_return,
            forward_pass_config=forward_pass_config,
        )
        _compile_cache[key] = (
            jax.jit(generate_tokens_fn)
            .lower(
                prompt_token_ids=jax.ShapeDtypeStruct(
                    (inference_config.batch_size, inference_config.padded_length),
                    jnp.int32,
                ),
                prompt_lengths_without_padding=jax.ShapeDtypeStruct((inference_config.batch_size,), jnp.int32),
                keys=jax.ShapeDtypeStruct((inference_config.batch_size,), jax.random.key(0).dtype),
            )
            .compile(compiler_options={"xla_gpu_autotune_level": "0"})
        )
    return _compile_cache[key]
