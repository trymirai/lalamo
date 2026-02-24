import subprocess
import sys
import textwrap

import pytest

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "LiquidAI/LFM2-350M",
    "cartesia-ai/Llamba-1B",
]


@pytest.mark.parametrize("model", MODELS)
def test_sharded_forward_passes_match(model: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(f"""
            import os
            os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
            os.environ["JAX_PLATFORMS"] = "cpu"

            import jax
            import jax.numpy as jnp

            assert jax.device_count() == 8, f"expected 8 devices, got {{jax.device_count()}}"

            from lalamo import Sharding, import_model
            from tests.common import assert_close

            SHARDING_CONFIGS = [
                Sharding.build(data_parallelism=8, tensor_parallelism=1),
                Sharding.build(data_parallelism=4, tensor_parallelism=2),
                Sharding.build(data_parallelism=2, tensor_parallelism=4),
                Sharding.build(data_parallelism=1, tensor_parallelism=8),
                Sharding.build(data_parallelism=4, tensor_parallelism=2, fsdp=True),
                Sharding.build(data_parallelism=2, tensor_parallelism=4, fsdp=True),
            ]

            MODEL = "{model}"
            BATCH_SIZE = 8
            SEQ_LEN = 16

            token_ids = jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32)
            token_positions = jnp.broadcast_to(jnp.arange(SEQ_LEN), (BATCH_SIZE, SEQ_LEN))

            reference_decoder = import_model(MODEL, precision=jnp.float32).model.model
            reference_logits = reference_decoder(token_ids, token_positions).logits

            for sharding in SHARDING_CONFIGS:
                print(f"testing {{sharding}}...")
                decoder = import_model(MODEL, sharding=sharding, precision=jnp.float32).model.model
                logits = decoder(token_ids, token_positions).logits
                assert_close(
                    result=logits,
                    reference=reference_logits,
                    operation_name=f"sharding {{sharding}}",
                )
                print(f"  OK")

            print("all sharding configs passed")
        """)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
