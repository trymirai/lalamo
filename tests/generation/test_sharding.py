import subprocess
import sys
import textwrap

import pytest

from lalamo.model_import.model_specs.common import ModelType
from tests.common import skip_on_gpu
from tests.conftest import filter_specs
from tests.model_test_tiers import ModelTier

MODELS = [spec.repo for spec in filter_specs(model_type=ModelType.LANGUAGE_MODEL, max_tier=ModelTier.CANONICAL)]


@pytest.mark.parametrize("model", MODELS)
def test_sharded_model_satisfies_sharding_invariants(model: str) -> None:
    skip_on_gpu("Sharding test forces CPU; incompatible with GPU mesh")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(f"""
            import os
            os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
            os.environ["JAX_PLATFORMS"] = "cpu"

            import jax
            jax.config.update("jax_default_matmul_precision", "F32_F32_F32")

            import jax.numpy as jnp

            assert jax.device_count() == 8, f"expected 8 devices, got {{jax.device_count()}}"

            import equinox as eqx
            import jax.sharding as shd

            from lalamo import ShardingConfig, import_model
            from lalamo.modules import pad_and_apply_data_sharding

            SHARDING_CONFIGS = [
                ShardingConfig.build(data_parallelism=8, tensor_parallelism=1),
                ShardingConfig.build(data_parallelism=4, tensor_parallelism=2),
                ShardingConfig.build(data_parallelism=2, tensor_parallelism=4),
                ShardingConfig.build(data_parallelism=1, tensor_parallelism=8),
                ShardingConfig.build(data_parallelism=4, tensor_parallelism=2, fsdp=True),
                ShardingConfig.build(data_parallelism=2, tensor_parallelism=4, fsdp=True),
            ]

            MODEL = "{model}"
            BATCH_SIZE = 8
            SEQ_LEN = 16

            token_ids = jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32)
            token_positions = jnp.broadcast_to(jnp.arange(SEQ_LEN), (BATCH_SIZE, SEQ_LEN))

            def spec_mentions_axis(spec: tuple[object, ...], axis_name: str) -> bool:
                return any(part == axis_name or (isinstance(part, tuple) and axis_name in part) for part in spec)

            def require_named_sharding(array: jax.Array) -> tuple[object, ...]:
                assert isinstance(array.sharding, shd.NamedSharding), array.sharding
                return tuple(array.sharding.spec)

            def named_sharding_spec(array: jax.Array) -> tuple[object, ...] | None:
                if not isinstance(array.sharding, shd.NamedSharding):
                    return None
                return tuple(array.sharding.spec)

            for sharding_config in SHARDING_CONFIGS:
                print(f"testing {{sharding_config}}...")
                decoder = import_model(MODEL, sharding_config=sharding_config, precision=jnp.float32).model.model
                sharded_token_ids, sharded_token_positions = pad_and_apply_data_sharding(
                    (token_ids, token_positions), sharding_config=sharding_config, batch_axis=0,
                )

                expected_input_spec = (sharding_config.data_axis_name, None)
                assert require_named_sharding(sharded_token_ids) == expected_input_spec
                assert require_named_sharding(sharded_token_positions) == expected_input_spec

                parameter_specs = [
                    spec
                    for leaf in jax.tree.leaves(decoder)
                    if eqx.is_array(leaf)
                    for spec in [named_sharding_spec(leaf)]
                    if spec is not None
                ]
                if sharding_config.tensor_axis_size > 1:
                    assert any(
                        spec_mentions_axis(spec, sharding_config.tensor_axis_name)
                        for spec in parameter_specs
                    )
                if sharding_config.fsdp:
                    assert any(
                        spec_mentions_axis(spec, sharding_config.data_axis_name)
                        for spec in parameter_specs
                    )
                print(f"  OK")

            print("all sharding configs passed")
        """),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"


def test_sharded_canonical_model_generates_simple_math_answer() -> None:
    skip_on_gpu("Sharding test forces CPU; incompatible with GPU mesh")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(f"""
            import os
            os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
            os.environ["JAX_PLATFORMS"] = "cpu"

            import jax
            jax.config.update("jax_default_matmul_precision", "F32_F32_F32")
            import jax.numpy as jnp

            from lalamo import ShardingConfig, UserMessage, import_model
            from lalamo.modules import pad_and_apply_data_sharding

            assert jax.device_count() == 8, f"expected 8 devices, got {{jax.device_count()}}"

            model_repo = "{MODELS[0]}"
            prompt = "What is 2+2? Answer with a single number, no thinking."
            sharding_config = ShardingConfig.build(data_parallelism=8, tensor_parallelism=1)
            model = import_model(model_repo, sharding_config=sharding_config, precision=jnp.float32).model

            request = model.message_processor.request_to_dict([UserMessage(prompt)], enable_thinking=False)
            rendered_prompt = model.message_processor.prompt_template.render(
                dict(request, strftime_now=lambda format_string: ""),
            )
            prompt_token_ids = model.message_processor.tokenize_text(rendered_prompt)
            token_ids = jnp.array(prompt_token_ids, dtype=jnp.int32)[None, :]
            lengths = jnp.array([len(prompt_token_ids)], dtype=jnp.int32)
            token_ids, lengths = pad_and_apply_data_sharding(
                (token_ids, lengths),
                sharding_config=sharding_config,
                batch_axis=0,
            )

            prefill = model._prefill(
                token_ids,
                state_capacity=token_ids.shape[1] + 1,
                lengths_without_padding=lengths,
            )
            next_token_id = jnp.argmax(prefill.last_token_logits[0]).item()
            response = model.message_processor.detokenize([next_token_id]).strip()
            assert response.startswith("4"), f"Expected a response starting with '4', got: {{response!r}}"
            print(response)
        """),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
