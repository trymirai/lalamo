from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import jax
from jax.sharding import Sharding

from .common import InferenceConfig

if TYPE_CHECKING:
    from jax._src.stages import Compiled
    from jaxtyping import Array, Int, Key

    from .language_model import ForwardPassConfig, GenerationConfig, LanguageModel

_compile_cache: dict[
    tuple[int, GenerationConfig | None, InferenceConfig | None, ForwardPassConfig | None, Sharding | None],
    Compiled,
] = {}


def compile_generate_tokens(
    model: LanguageModel,
    generation_config: GenerationConfig | None = None,
    inference_config: InferenceConfig = InferenceConfig(),  # noqa: B008
    *,
    forward_pass_config: ForwardPassConfig | None = None,
    prompt_token_ids: Int[Array, "batch length"],
    prompt_lengths_without_padding: Int[Array, " batch"],
    keys: Key[Array, " batch"],
) -> Compiled:
    from .language_model import LanguageModel

    key = (id(model), generation_config, inference_config, forward_pass_config, prompt_token_ids.sharding)
    if key not in _compile_cache:
        generate_tokens_fn = functools.partial(
            LanguageModel.generate_tokens,
            generation_config=generation_config,
            max_output_length=inference_config.max_output_length,
            num_top_logits_to_return=inference_config.num_top_logits_to_return,
            forward_pass_config=forward_pass_config,
        )
        _compile_cache[key] = (
            jax.jit(generate_tokens_fn)
            .lower(
                model,
                prompt_token_ids=prompt_token_ids,
                prompt_lengths_without_padding=prompt_lengths_without_padding,
                keys=keys,
            )
            # the autotune levels are (according to https://guides.lw1.at/all-xla-options/#--xla_gpu_autotune_level)
            # 0 - no autotune, gpu shouldn't be touched
            # 1 - basic level, gpu should be touched veeery little
            # 2,3 - gpu touched more and more
            # 4 (default) - gpu might allocate more memory than the run would require!
            .compile(compiler_options={"xla_gpu_autotune_level": "0"})
        )
    return _compile_cache[key]
