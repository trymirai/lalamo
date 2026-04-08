from __future__ import annotations

import functools
import threading
import weakref
from typing import TYPE_CHECKING

import jax

jax.config.update("jax_compiler_enable_remat_pass", False)

from .common import InferenceConfig

if TYPE_CHECKING:
    from jax._src.stages import Compiled
    from jax.sharding import Sharding
    from jaxtyping import Array, Int, Key

    from .language_model import ForwardPassConfig, GenerationConfig, GenerationTraceConfig, LanguageModel

_compile_cache: dict[
    int,
    dict[
        tuple[
            GenerationConfig | None,
            InferenceConfig | None,
            ForwardPassConfig | None,
            GenerationTraceConfig | None,
            Sharding | None,
        ],
        Compiled,
    ],
] = {}
_compile_lock = threading.Lock()


def _make_weak_finalizer(model_id: int) -> None:
    _compile_cache.pop(model_id, None)


def compile_generate_tokens(
    model: LanguageModel,
    generation_config: GenerationConfig | None = None,
    inference_config: InferenceConfig = InferenceConfig(),  # noqa: B008
    *,
    forward_pass_config: ForwardPassConfig | None = None,
    generation_trace_config: GenerationTraceConfig | None = None,
    prompt_token_ids: Int[Array, "batch length"],
    prompt_lengths_without_padding: Int[Array, " batch"],
    keys: Key[Array, " batch"],
) -> Compiled:
    from .language_model import LanguageModel

    model_id = id(model)
    key = (
        generation_config,
        inference_config,
        forward_pass_config,
        generation_trace_config,
        prompt_token_ids.sharding,
    )

    with _compile_lock:
        if model_id not in _compile_cache:
            _compile_cache[model_id] = {}
            weakref.finalize(model, _make_weak_finalizer, model_id)
        if key in _compile_cache[model_id]:
            return _compile_cache[model_id][key]

    # Compile outside the lock — .compile() releases the GIL so other threads can compile concurrently
    generate_tokens_fn = functools.partial(
        LanguageModel.generate_tokens,
        generation_config=generation_config,
        max_output_length=inference_config.max_output_length,
        num_top_logits_to_return=inference_config.num_top_logits_to_return,
        forward_pass_config=forward_pass_config,
        generation_trace_config=generation_trace_config,
    )
    compiled = (
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
        .compile(
            compiler_options={
                "xla_gpu_autotune_level": "0",
                # "xla_disable_hlo_passes": "hlo-rematerialization",
            }
        )
    )

    with _compile_lock:
        _compile_cache[model_id].setdefault(key, compiled)
    return _compile_cache[model_id][key]
