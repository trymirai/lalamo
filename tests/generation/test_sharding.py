import subprocess
import sys
import textwrap

import pytest

from tests.common import skip_on_gpu

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
]


@pytest.mark.parametrize("model", MODELS)
def test_sharded_forward_passes_match(model: str) -> None:
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
            import jax.numpy as jnp

            assert jax.device_count() == 8, f"expected 8 devices, got {{jax.device_count()}}"

            from lalamo import ShardingConfig, import_model
            from lalamo.modules import pad_and_apply_data_sharding
            from tests.common import assert_close

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

            reference_decoder = import_model(MODEL, precision=jnp.float32).model.model
            reference_logits = reference_decoder(
                token_ids, token_positions, return_updated_state=True,
            ).logits

            for sharding_config in SHARDING_CONFIGS:
                print(f"testing {{sharding_config}}...")
                decoder = import_model(MODEL, sharding_config=sharding_config, precision=jnp.float32).model.model
                sharded_token_ids, sharded_token_positions = pad_and_apply_data_sharding(
                    (token_ids, token_positions), sharding_config=sharding_config, batch_axis=0,
                )
                decoder_result = decoder(sharded_token_ids, sharded_token_positions, return_updated_state=True)
                assert_close(
                    result=decoder_result.logits,
                    reference=reference_logits,
                    atol=5e-4,
                    operation_name=f"sharding {{sharding_config}}",
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
