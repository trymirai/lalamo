from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from lalamo.compressed.s_trellis import STrellisMatrix, STrellisSpec
from tests.helpers import make_test_sharding_config


def load_saved_trellis(name: str, spec: STrellisSpec) -> STrellisMatrix:
    # Four synthetic rows fitted and packed by the independent Torch producers.
    filename = "s_trellis_muse.npz" if name.endswith("_connected") else "s_trellis_hyb036.npz"
    with np.load(Path(__file__).parent / "data" / filename) as data:
        return STrellisMatrix(
            spec=spec,
            sharding_config=make_test_sharding_config(),
            is_sharded=True,
            codes=jnp.asarray(data[f"{name}_codes"]),
            scales=jnp.asarray(data[f"{name}_scales"]),
            gains=jax.lax.bitcast_convert_type(jnp.asarray(data[f"{name}_gains_bits"]), jnp.bfloat16),
            table=jnp.asarray(
                data[f"{name}_table"] if name.endswith("_connected") else data[f"table_v{spec.vector_width}"]
            ),
            signs=jnp.asarray(data[f"{name}_signs"]),
            small_q=jnp.asarray(data[f"{name}_small_q"]),
        ).switch_sharding_config(make_test_sharding_config())
