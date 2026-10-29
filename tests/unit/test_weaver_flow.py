from collections.abc import Mapping
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from torch.nn import functional

from lalamo.initializer import EmptyInitializer
from lalamo.model_import.loaders.weaver_loader import load_weaver
from lalamo.model_import.model_configs.huggingface.weaver import HFWeaverConfig
from lalamo.module import Keychain, ShardingConfig
from lalamo.modules.speculators.weaver import WeaverConfig


def _weights(config: HFWeaverConfig) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(3)
    weights: dict[str, torch.Tensor] = {}

    def linear(path: str, source: int, target: int, *, bias: bool) -> None:
        weights[f"{path}.weight"] = torch.randn(target, source, generator=generator) / source**0.5
        if bias:
            weights[f"{path}.bias"] = torch.randn(target, generator=generator) * 0.1

    def norm(path: str, channels: int) -> None:
        weights[f"{path}.weight"] = 1 + torch.randn(channels, generator=generator) * 0.1
        weights[f"{path}.bias"] = torch.randn(channels, generator=generator) * 0.1

    norm("embed_norm", config.d_embed)
    norm("output_norm", config.d_model)
    norm("out_norm", config.d_rank)
    linear("token_in", config.d_embed, config.d_rank, bias=True)
    linear("proposal_in", config.d_model, config.d_rank, bias=True)
    linear("lm_head_query_in", config.d_rank, config.d_model, bias=False)
    for index in range(config.num_layers):
        path = f"blocks.{index}"
        norm(f"{path}.norm_attn", config.d_rank)
        norm(f"{path}.norm_mlp", config.d_rank)
        linear(f"{path}.qkv_proj", config.d_rank, 3 * config.d_rank, bias=False)
        linear(f"{path}.o_proj", config.d_rank, config.d_rank, bias=False)
        linear(f"{path}.gate_up_proj", config.d_rank, 2 * config.mlp_channels, bias=True)
        linear(f"{path}.down_proj", config.mlp_channels, config.d_rank, bias=True)
    return weights


def _reference(
    config: HFWeaverConfig,
    weights: Mapping[str, torch.Tensor],
    last: torch.Tensor,
    proposals: torch.Tensor,
    embedding: torch.Tensor,
    lm_head: torch.Tensor,
    token_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
    candidate_scores: torch.Tensor,
    positions: torch.Tensor,
    ancestor_keys: torch.Tensor,
    ancestor_values: torch.Tensor,
    ancestor_mask: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    def linear(inputs: torch.Tensor, path: str) -> torch.Tensor:
        return functional.linear(inputs, weights[f"{path}.weight"], weights.get(f"{path}.bias"))

    def norm(inputs: torch.Tensor, path: str) -> torch.Tensor:
        return (
            functional.rms_norm(inputs, (inputs.shape[-1],), weights[f"{path}.weight"], 1e-6) + weights[f"{path}.bias"]
        )

    def body(
        inputs: torch.Tensor,
        positions: torch.Tensor,
        prefix_keys: torch.Tensor | None = None,
        prefix_values: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, _ = inputs.shape
        head_dim = config.d_rank // config.num_heads
        frequencies = config.rope_base ** (-torch.arange(0, head_dim, 2).float() / head_dim)
        angles = positions[..., None] * frequencies
        angles = torch.cat((angles, angles), dim=-1)[:, :, None]

        def rotate(heads: torch.Tensor) -> torch.Tensor:
            left, right = heads.chunk(2, dim=-1)
            return heads * angles.cos() + torch.cat((-right, left), dim=-1) * angles.sin()

        saved_keys, saved_values = [], []
        for index in range(config.num_layers):
            path = f"blocks.{index}"
            query, key, value = (
                part.reshape(batch, length, config.num_heads, head_dim)
                for part in linear(norm(inputs, f"{path}.norm_attn"), f"{path}.qkv_proj").chunk(3, dim=-1)
            )
            query, key = rotate(query), rotate(key)
            saved_keys.append(key)
            saved_values.append(value)
            if prefix_keys is None:
                mask = torch.ones(length, length, dtype=torch.bool).tril()[None]
            else:
                assert prefix_values is not None
                prefix_length = prefix_keys.shape[2]
                key = torch.cat((prefix_keys[index], ancestor_keys[index], key), dim=1)
                value = torch.cat((prefix_values[index], ancestor_values[index], value), dim=1)
                mask = functional.pad(ancestor_mask, (prefix_length, 1), value=True)[:, None]
            scores = torch.einsum("bshd,bthd->bhst", query, key) * head_dim**-0.5
            scores = scores.masked_fill(~mask[:, None], -torch.inf)
            attended = torch.einsum("bhst,bthd->bshd", scores.softmax(-1), value).reshape(batch, length, -1)
            inputs = inputs + linear(attended, f"{path}.o_proj")
            gate, up = linear(norm(inputs, f"{path}.norm_mlp"), f"{path}.gate_up_proj").chunk(2, dim=-1)
            inputs = inputs + linear(functional.silu(gate) * up, f"{path}.down_proj")
        return norm(inputs, "out_norm"), torch.stack(saved_keys), torch.stack(saved_values)

    features = torch.cat((last[:, None], proposals), dim=1)
    projected = linear(norm(features, "output_norm"), "proposal_in")
    _, prefix_keys, prefix_values = body(projected, torch.arange(projected.shape[1])[None].expand(projected.shape[:2]))
    tokens = linear(norm(embedding[token_ids], "embed_norm"), "token_in")[:, None]
    position_ids = positions.clamp(0, ancestor_keys.shape[2] - 1)[:, None] + 1
    hidden, own_keys, own_values = body(tokens, position_ids, prefix_keys, prefix_values)
    query = linear(hidden[:, 0], "lm_head_query_in").to(torch.bfloat16)
    candidate_weights = lm_head[candidate_ids.clamp_min(0)].to(torch.bfloat16)
    residual = torch.einsum("rh,rch->rc", query, candidate_weights).float()
    logits = (candidate_scores + residual).masked_fill((candidate_ids < 0) | ~candidate_scores.isfinite(), -torch.inf)
    return prefix_keys, prefix_values, logits, own_keys[:, :, 0], own_values[:, :, 0]


@pytest.mark.parametrize("backend", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
def test_weaver_checkpoint_prefix_tree_and_native(backend: str, tmp_path: Path) -> None:
    raw: dict[str, object] = {
        "d_model": 10,
        "d_embed": 6,
        "d_rank": 8,
        "num_layers": 2,
        "num_heads": 2,
        "mlp_dim": 12,
        "K": 4,
        "candidate_pool_size": 4,
        "rope_base": 10000.0,
    }
    foreign = HFWeaverConfig.from_dict(raw)
    weights = _weights(foreign)
    path = tmp_path / "weaver.pt"
    torch.save({"config": raw, "state_dict": weights}, path)
    generator = torch.Generator().manual_seed(7)

    def random(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator)

    last, proposals = random(2, 10), random(2, 2, 10)
    embedding, lm_head = random(16, 6), random(16, 10)
    tokens, candidates = torch.tensor([1, 3]), torch.tensor([[5, 6, 7, -1], [2, 3, 4, 5]])
    scores, positions = random(2, 4), torch.tensor([1, 3])
    ancestor_keys, ancestor_values = random(2, 2, 4, 2, 4), random(2, 2, 4, 2, 4)
    mask = torch.tensor([[True, False, True, False], [False, True, True, False]])
    expected = _reference(
        foreign,
        weights,
        last,
        proposals,
        embedding,
        lm_head,
        tokens,
        candidates,
        scores,
        positions,
        ancestor_keys,
        ancestor_values,
        mask,
    )
    device = jax.devices(backend)[0]
    sharding = ShardingConfig.replicated([device])
    with jax.default_device(device), jax.set_mesh(sharding.mesh), jax.default_matmul_precision("highest"):
        model = load_weaver(path, sharding, jnp.float32)
        keychain = Keychain.init(0, sharding_config=sharding)
        (
            last,
            proposals,
            embedding,
            lm_head,
            tokens,
            candidates,
            scores,
            positions,
            ancestor_keys,
            ancestor_values,
            mask,
        ) = (
            jnp.asarray(tensor.numpy())
            for tensor in (
                last,
                proposals,
                embedding,
                lm_head,
                tokens,
                candidates,
                scores,
                positions,
                ancestor_keys,
                ancestor_values,
                mask,
            )
        )
        prefix = model.prompt_prefix(last, proposals, keychain=keychain)
        actual = model.step(
            lm_head,
            embedding,
            tokens,
            candidates,
            scores,
            prefix.keys,
            prefix.values,
            positions,
            ancestor_keys,
            ancestor_values,
            mask,
            keychain=keychain,
        )
        for observed, reference in zip((prefix.keys, prefix.values, *actual), expected, strict=True):
            np.testing.assert_allclose(observed, reference.numpy(), atol=3e-5, rtol=3e-5)
        restored = (
            WeaverConfig.from_json(model.config.to_json())
            .init(EmptyInitializer(jnp.float32, sharding))
            .load_exported(model.export())
        )
        restored_prefix = restored.prompt_prefix(last, proposals, keychain=keychain)
        np.testing.assert_array_equal(restored_prefix.keys, prefix.keys)
        np.testing.assert_array_equal(restored_prefix.values, prefix.values)
