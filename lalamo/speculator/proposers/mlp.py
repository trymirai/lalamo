from __future__ import annotations

import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any, Self

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PRNGKeyArray  # noqa: TC002
from typer import Option

from lalamo.data.completion_features import FeatureRequest, LalamoCompletionFeatures
from lalamo.data.lalamo_completions import LalamoCompletion  # noqa: TC001
from lalamo.modules.decoder import Decoder  # noqa: TC001
from lalamo.speculator.common import (
    Speculator,
    SpeculatorBackend,
    write_speculator_artifact,
)
from lalamo.speculator.proposal import TrieProposal  # noqa: TC001
from lalamo.speculator.state import LMState, StateRequest
from lalamo.speculator.training import (
    SpeculatorBatchResult,
    SpeculatorTrainer,
    SpeculatorTrainingConfig,
    SpeculatorTrainingContext,
    SpeculatorTrainingEvent,
)

__all__ = [
    "MLPBackend",
    "MLPConfig",
    "MLPModel",
    "MLPSpeculator",
    "MLPTrainer",
    "MLPTrainingState",
]


class MLPModel(eqx.Module):
    input_weights: Float[Array, "hidden input"]
    input_bias: Float[Array, " hidden"]
    output_weights: Float[Array, "depth vocab hidden"]
    output_bias: Float[Array, "depth vocab"]
    input_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    vocab_size: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        input_dim: int,
        hidden_dim: int,
        vocab_size: int,
        depth: int,
        *,
        key: PRNGKeyArray,
    ) -> Self:
        input_key, output_key = jax.random.split(key)
        input_scale = 1 / math.sqrt(input_dim)
        output_scale = 1 / math.sqrt(hidden_dim)
        return cls(
            input_weights=jax.random.uniform(
                input_key,
                (hidden_dim, input_dim),
                minval=-input_scale,
                maxval=input_scale,
                dtype=jnp.float32,
            ),
            input_bias=jnp.zeros((hidden_dim,), dtype=jnp.float32),
            output_weights=jax.random.uniform(
                output_key,
                (depth, vocab_size, hidden_dim),
                minval=-output_scale,
                maxval=output_scale,
                dtype=jnp.float32,
            ),
            output_bias=jnp.zeros((depth, vocab_size), dtype=jnp.float32),
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            depth=depth,
        )

    def __call__(
        self,
        features: Float[Array, "batch input"],
    ) -> Float[Array, "batch depth vocab"]:
        hidden = jax.nn.gelu(features.astype(jnp.float32) @ self.input_weights.T + self.input_bias)
        return jnp.einsum("bh,dvh->bdv", hidden, self.output_weights) + self.output_bias[None, :, :]


@dataclass(frozen=True, kw_only=True)
class MLPSpeculator(Speculator):
    model: MLPModel
    width: int = 4

    @classmethod
    def create(
        cls,
        model: MLPModel,
        width: int = 4,
    ) -> Self:
        return cls(
            model=model,
            width=width,
        )

    @property
    def state_request(self) -> StateRequest:
        return StateRequest(output_norm_capacity=1)

    def draft(self, state: LMState) -> TrieProposal:
        output_norm, _ = state.recent_output_norm(1)
        logits = self.model(output_norm[:, -1])
        width = min(self.width, logits.shape[-1])
        token_ids = jax.lax.top_k(logits, width)[1]
        proposal = state.create_root_proposal(budget=self.model.depth * width + 1)
        batch_indices = jnp.arange(state.root_bonus_id.shape[0], dtype=jnp.int32)
        parent_indices = jnp.zeros((state.root_bonus_id.shape[0],), dtype=jnp.int32)

        for depth_index in range(self.model.depth):
            for rank in range(width):
                proposal, node_index = proposal.add_nodes(
                    batch_indices=batch_indices,
                    parent_indices=parent_indices,
                    token_ids=token_ids[:, depth_index, rank],
                )
                if rank == 0:
                    next_parent_indices = jnp.full(parent_indices.shape, node_index, dtype=jnp.int32)
            parent_indices = next_parent_indices

        return proposal


@dataclass(frozen=True)
class MLPConfig:
    hidden_dim: Annotated[int, Option("--hidden_dim", help="Hidden dimension of the MLP predictor.")] = 512
    depth: Annotated[int, Option("--depth", help="Number of draft positions predicted by the MLP.")] = 4
    width: Annotated[int, Option("--width", help="Number of token candidates retained per draft position.")] = 4
    learning_rate: Annotated[float, Option("--learning_rate", help="AdamW learning rate.")] = 3e-4
    weight_decay: Annotated[float, Option("--weight_decay", help="AdamW weight decay.")] = 0.0
    seed: Annotated[int, Option("--seed", help="Random seed used to initialize the MLP.")] = 0


@dataclass(frozen=True)
class MLPTrainingState:
    model: MLPModel | None
    optimizer_state: optax.OptState | None
    key: PRNGKeyArray


@dataclass(frozen=True, kw_only=True)
class MLPTrainer(SpeculatorTrainer[MLPSpeculator, MLPTrainingState]):
    artifact_path: Path | str
    target_model: Decoder
    config: MLPConfig

    def make_feature_request(
        self,
        completions: tuple[LalamoCompletion, ...],
        config: SpeculatorTrainingConfig,
    ) -> FeatureRequest:
        return FeatureRequest(
            completions=completions,
            top_k_logits=config.top_k_logits,
            output_features=True,
        )

    def init_state(self) -> MLPTrainingState:
        return MLPTrainingState(
            model=None,
            optimizer_state=None,
            key=jax.random.key(self.config.seed),
        )

    def train(
        self,
        state: MLPTrainingState,
        features: LalamoCompletionFeatures,
        context: SpeculatorTrainingContext,
    ) -> tuple[MLPTrainingState, SpeculatorBatchResult]:
        del context
        model, optimizer_state = self.init_model_and_optimizer(state, features)
        (loss, loss_weight), grads = eqx.filter_value_and_grad(mlp_loss, has_aux=True)(model, features)
        updates, optimizer_state = self.optimizer.update(grads, optimizer_state, model)
        model = eqx.apply_updates(model, updates)
        return (
            MLPTrainingState(
                model=model,
                optimizer_state=optimizer_state,
                key=state.key,
            ),
            SpeculatorBatchResult(
                loss=float(loss),
                loss_weight=float(loss_weight),
            ),
        )

    def evaluate(
        self,
        state: MLPTrainingState,
        features: LalamoCompletionFeatures,
        context: SpeculatorTrainingContext,
    ) -> SpeculatorBatchResult:
        del context
        assert state.model is not None
        loss, loss_weight = mlp_loss(state.model, features)
        return SpeculatorBatchResult(
            loss=float(loss),
            loss_weight=float(loss_weight),
        )

    def finish(
        self,
        state: MLPTrainingState,
    ) -> MLPSpeculator:
        assert state.model is not None
        return MLPSpeculator.create(
            model=state.model,
            width=self.config.width,
        )

    def save(
        self,
        state: MLPTrainingState,
        event: SpeculatorTrainingEvent,
    ) -> None:
        del event
        assert state.model is not None
        buffer = BytesIO()
        eqx.tree_serialise_leaves(buffer, state.model)
        write_speculator_artifact(
            self.artifact_path,
            MLPBackend,
            self.config.width,
            state.model.depth,
            state.model.input_dim,
            state.model.hidden_dim,
            state.model.vocab_size,
            buffer.getvalue(),
        )

    @property
    def optimizer(self) -> optax.GradientTransformation:
        return optax.adamw(
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def init_model_and_optimizer(
        self,
        state: MLPTrainingState,
        features: LalamoCompletionFeatures,
    ) -> tuple[MLPModel, optax.OptState]:
        if state.model is not None and state.optimizer_state is not None:
            return state.model, state.optimizer_state

        output_features = features.output_features
        assert output_features is not None
        model = MLPModel.init(
            input_dim=output_features.shape[-1],
            hidden_dim=self.config.hidden_dim,
            vocab_size=self.target_model.vocab_size,
            depth=self.config.depth,
            key=state.key,
        )
        return model, self.optimizer.init(eqx.filter(model, eqx.is_array))


def mlp_loss(
    model: MLPModel,
    features: LalamoCompletionFeatures,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    output_features = features.output_features
    assert output_features is not None

    logits = model(output_features.reshape((-1, output_features.shape[-1]))).reshape(
        (*output_features.shape[:2], model.depth, model.vocab_size),
    )
    target_ids = features.completion_batch.target_token_ids
    target_mask = features.completion_batch.target_mask
    loss_sum = jnp.array(0.0, dtype=jnp.float32)
    loss_weight = jnp.array(0.0, dtype=jnp.float32)

    for depth_index in range(model.depth):
        mask = target_mask[:, depth_index:]
        token_losses = optax.softmax_cross_entropy_with_integer_labels(
            logits[:, : mask.shape[1], depth_index],
            target_ids[:, depth_index:],
        )
        loss_sum = loss_sum + jnp.sum(jnp.where(mask, token_losses, 0.0))
        loss_weight = loss_weight + mask.sum().astype(jnp.float32)

    return loss_sum / jnp.maximum(loss_weight, 1.0), loss_weight


class MLPBackend(SpeculatorBackend[MLPConfig]):
    name = "mlp"
    config_type = MLPConfig

    @classmethod
    def create_trainer(
        cls,
        config: MLPConfig,
        artifact_path: Path,
        target_model: Decoder,
    ) -> MLPTrainer:
        return MLPTrainer(
            artifact_path=artifact_path,
            target_model=target_model,
            config=config,
        )

    @classmethod
    def deserialize(cls, fields: tuple[Any, ...], target_model: Decoder) -> MLPSpeculator:
        del target_model
        if len(fields) != 6:
            raise ValueError(
                "mlp speculator artifact must contain width, depth, input_dim, hidden_dim, "
                "vocab_size, and model bytes."
            )
        width, depth, input_dim, hidden_dim, vocab_size, data = fields
        if not all(isinstance(value, int) for value in (width, depth, input_dim, hidden_dim, vocab_size)):
            raise TypeError("mlp speculator metadata fields must be integers.")
        if not isinstance(data, bytes):
            raise TypeError("mlp speculator parameter payload must be bytes.")
        template = MLPModel.init(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            depth=depth,
            key=jax.random.key(0),
        )
        model = eqx.tree_deserialise_leaves(BytesIO(data), template)
        return MLPSpeculator.create(
            model=model,
            width=width,
        )
