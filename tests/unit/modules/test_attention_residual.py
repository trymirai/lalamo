import jax
import pytest
import torch
from jaxtyping import Array, Float
from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig
from transformers.models.inkling2_loop.modeling_inkling2_loop import (
    Inkling2LoopBlockAttentionResidual,
    Inkling2LoopMessageAttentionResidual,
    Inkling2LoopMessageGate,
)

from lalamo.model_import.loaders.inkling import load_message_gates
from lalamo.module import LogicalAxis
from lalamo.modules import (
    LinearConfig,
    MessageGateConfig,
)
from lalamo.modules.attention_residual import attention_residual_read
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.torch_interop import torch_to_jax
from lalamo.weight_matrix import MatmulConfig
from tests.common import assert_close
from tests.helpers import make_test_sharding_config
from tests.unit.modules.qwen4_exp import make_initializer, make_keychain, randomized

HIDDEN_SIZE = 32
GATE_RANK = 8
EPSILON = 1e-6


def make_hf_config() -> Inkling2LoopTextConfig:
    return Inkling2LoopTextConfig(
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=6,
        n_layers_per_attn_res_block=2,
        loop_layer_index_start=2,
        loop_layer_index_end=4,
        loop_gate_rank=GATE_RANK,
        rms_norm_eps=EPSILON,
    )


def random_states(*shape: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(*shape)


def as_sharded(tensor: torch.Tensor) -> Float[Array, "batch ..."]:
    return jax.device_put(
        torch_to_jax(tensor),
        make_test_sharding_config().resolve_sharding((LogicalAxis.BATCH, *(None,) * (tensor.ndim - 1))),
    )


@pytest.mark.parametrize("num_queries", [1, 3])
def test_read_matches_hf(num_queries: int) -> None:
    hf_read = randomized(Inkling2LoopBlockAttentionResidual(make_hf_config(), num_queries=num_queries))
    block_states = random_states(2, 5, 3, HIDDEN_SIZE, seed=1)
    partial = random_states(2, 5, HIDDEN_SIZE, seed=2)

    with torch.no_grad():
        reference = torch.stack(hf_read(block_states, partial), dim=2)
    result = attention_residual_read(
        torch_to_jax(hf_read.query.reshape(num_queries, HIDDEN_SIZE)) * HIDDEN_SIZE**-0.5,
        torch_to_jax(torch.cat([block_states, partial.unsqueeze(2)], dim=2)),
        EPSILON,
    )

    assert_close(result=result, reference=torch_to_jax(reference))


@pytest.mark.usefixtures("fake_mesh")
@pytest.mark.parametrize("role", range(4))
def test_message_gate_matches_hf(role: int) -> None:
    hf_config = make_hf_config()
    hf_messages = randomized(Inkling2LoopMessageAttentionResidual(hf_config))
    hf_gate = randomized(Inkling2LoopMessageGate(hf_config))
    loop_layer_index = 1
    previous_blocks = random_states(2, 5, 2, HIDDEN_SIZE, seed=1)
    previous_partial = random_states(2, 5, HIDDEN_SIZE, seed=2)
    hidden_states = random_states(2, 5, HIDDEN_SIZE, seed=3)

    with torch.no_grad():
        message = hf_messages(previous_blocks, previous_partial, loop_layer_index)[:, :, role]
        reference = hf_gate(hidden_states, message)

    controller_path = ParameterPath("controller")
    gates = load_message_gates(
        MessageGateConfig(linear_config=LinearConfig(), rank=GATE_RANK).init_gates(
            make_initializer(),
            HIDDEN_SIZE,
            1,
            shares_hidden_projection=True,
        ),
        {
            controller_path / "down_weight": torch_to_jax(hf_gate.down_proj.weight.detach()),
            controller_path / "forget_weight": torch_to_jax(hf_gate.forget_gate_proj.weight.detach()),
            controller_path / "gain_weight": torch_to_jax(hf_gate.message_gate_proj.weight.detach()),
        },
        (controller_path,),
        (torch_to_jax(hf_messages.query[loop_layer_index, role].detach()),),
    )
    (result,) = gates(
        (as_sharded(hidden_states),),
        as_sharded(torch.cat([previous_blocks, previous_partial.unsqueeze(2)], dim=2)),
        EPSILON,
        MatmulConfig.for_tracer_tests(),
        keychain=make_keychain(),
    )

    assert_close(result=result, reference=torch_to_jax(reference))
