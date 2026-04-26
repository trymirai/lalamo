import functools
import weakref

import jax
from jax._src.stages import Compiled
from jaxtyping import Array, Int

from lalamo.module import Keychain
from .common import InferenceConfig

_compile_cache: dict[int, dict[tuple, Compiled]] = {}


def _make_weak_finalizer(model_id: int) -> None:
    _compile_cache.pop(model_id, None)


def compile_generate_tokens(
    model: "LanguageModel",  # noqa: F821  # type: ignore[name-defined]
    generation_config: "GenerationConfig | None" = None,  # noqa: F821  # type: ignore[name-defined]
    inference_config: InferenceConfig = InferenceConfig(),
    *,
    forward_pass_config: "ForwardPassConfig",  # noqa: F821  # type: ignore[name-defined]
    generation_trace_config: "GenerationTraceConfig | None" = None,  # noqa: F821  # type: ignore[name-defined]
    prompt_token_ids: Int[Array, "batch length"],
    prompt_lengths_without_padding: Int[Array, " batch"],
    keychain: Keychain,
) -> Compiled:
    model_id = id(model)
    key = (
        generation_config,
        inference_config,
        forward_pass_config,
        generation_trace_config,
        prompt_token_ids.sharding,
    )
    if model_id not in _compile_cache:
        _compile_cache[model_id] = {}
        weakref.finalize(model, _make_weak_finalizer, model_id)
    if key not in _compile_cache[model_id]:
        generate_tokens_fn = functools.partial(
            type(model).generate_tokens,
            generation_config=generation_config,
            max_output_length=inference_config.max_output_length,
            num_top_logits_to_return=inference_config.num_top_logits_to_return,
            forward_pass_config=forward_pass_config,
            generation_trace_config=generation_trace_config,
        )
        _compile_cache[model_id][key] = (
            jax.jit(generate_tokens_fn)
            .lower(
                model,
                prompt_token_ids=prompt_token_ids,
                prompt_lengths_without_padding=prompt_lengths_without_padding,
                keychain=keychain,
            )
            # the autotune levels are (according to https://guides.lw1.at/all-xla-options/#--xla_gpu_autotune_level)
            # 0 - no autotune, gpu shouldn't be touched
            # 1 - basic level, gpu should be touched veeery little
            # 2,3 - gpu touched more and more
            # 4 (default) - gpu might allocate more memory than the run would require!
            .compile(compiler_options={"xla_gpu_autotune_level": "0"})
        )
    return _compile_cache[model_id][key]
