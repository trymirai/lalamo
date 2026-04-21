import pytest
import torch

from lalamo.qwen_moe_offload_eval import (
    OffloadRuntimeController,
    OffloadRuntimeState,
    ResidentExpertCache,
    install_expert_offload,
)

pytestmark = pytest.mark.fast


class FakeExperts(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_experts = 2
        self.gate_up_proj = torch.nn.Parameter(
            torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[0.0, 1.0], [1.0, 0.0]],
                ]
            )
        )
        self.down_proj = torch.nn.Parameter(
            torch.tensor(
                [
                    [[1.0], [2.0]],
                    [[-1.0], [1.0]],
                ]
            )
        )
        self.act_fn = torch.nn.Identity()


class FakeLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = torch.nn.Module()
        self.mlp.experts = FakeExperts()


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([FakeLayer()])
        self.config = type("Config", (), {"text_config": type("Text", (), {"num_experts_per_tok": 1})()})()


def test_resident_expert_cache_tracks_hits_and_evictions() -> None:
    state = OffloadRuntimeState(expert_bytes=64)
    cache = ResidentExpertCache(
        cache_size=1,
        gate_up_cpu=torch.arange(8, dtype=torch.float32).reshape(2, 2, 2),
        down_cpu=torch.arange(4, dtype=torch.float32).reshape(2, 2, 1),
        device=torch.device("cpu"),
        state=state,
    )

    cache.weights(0)
    cache.weights(0)
    cache.weights(1)

    assert state.prompt.misses == 2
    assert state.prompt.hits == 1
    assert state.prompt.bytes_loaded == 128
    assert cache.slot_by_expert == {1: 0}


def test_install_expert_offload_patches_forward_and_uses_runtime_cache() -> None:
    model = FakeModel()
    controller = install_expert_offload(model, cache_size=1, expert_bytes=32)
    experts = model.model.layers[0].mlp.experts
    hidden_states = torch.tensor([[2.0, 3.0], [5.0, 7.0]])
    top_k_index = torch.tensor([[0], [0]])
    top_k_weights = torch.ones((2, 1))

    first_output = experts(hidden_states, top_k_index, top_k_weights)
    second_output = experts(hidden_states[:1], top_k_index[:1], top_k_weights[:1])
    third_output = experts(hidden_states[:1], torch.tensor([[1]]), torch.ones((1, 1)))

    assert experts.gate_up_proj.numel() == 0
    assert experts.down_proj.numel() == 0
    assert torch.allclose(first_output, torch.tensor([[6.0, 12.0], [35.0, 70.0]]))
    assert torch.allclose(second_output, torch.tensor([[6.0, 12.0]]))
    assert torch.allclose(third_output, torch.tensor([[-6.0, 6.0]]))
    assert controller.state.prompt.misses == 2
    assert controller.state.prompt.hits == 1
    assert controller.state.prompt.bytes_loaded == 64


def test_offload_runtime_controller_finalize_reports_transfer_metrics() -> None:
    state = OffloadRuntimeState(expert_bytes=8)
    state.prompt.bytes_loaded = 16
    state.continuation.bytes_loaded = 32
    state.continuation.hits = 6
    state.continuation.misses = 2
    state.prompt.elapsed_seconds = 0.5
    state.continuation.elapsed_seconds = 2.0
    controller = OffloadRuntimeController(caches=(object(), object()), state=state, expert_bytes=8, cache_size=4)

    stats = controller.finalize(continuation_token_count=8)

    assert stats.cache_size == 4
    assert stats.prompt_transfer_bytes == 16
    assert stats.continuation_transfer_bytes == 32
    assert stats.continuation_expert_loads_per_token == 0.25
    assert stats.continuation_hit_rate == 0.75
    assert stats.continuation_tokens_per_second == 4.0
    assert stats.resident_gib_total == pytest.approx((4 * 2 * 8) / (1024**3))
