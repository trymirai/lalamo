"""Tests for the forward pass of the DFlare block drafter and its recurrent DSpark head.

The expectations are written in the checkpoint's own terms, never taken from the code under test: a tiny random
checkpoint with the tensor names of the original (`fuse_weight`, `layers.N.self_attn.q_proj.weight`,
`markov_head.rnn_proj.weight`, ...) goes through the real loader, and a plain reference (numpy float64, Python
loops) computes the same quantities from the raw tensors the way the serving implementation of the checkpoint
defines them:

  context   alpha = softmax(fuse_weight row) kept in bfloat16; a committed row gives every layer
            c = rms_norm(sum_t alpha_t * tap_t) * dflare_norm, k = rope(rms_norm(k_proj_ctx c per head) * k_norm_ctx),
            v = v_proj_ctx c
  block     [embedding of the anchor, mask_embed x (block - 1)] at positions p .. p + block - 1; a pre-norm layer:
            x += o_proj(attention(q, k, v)), x += down(silu(gate(n)) * up(n)); queries and keys are normalized per
            head and rotated; every block row sees the whole block and the committed rows before p, a sliding layer
            only the newest `window - 1` of them (the trained window counts the anchor); hidden = rms_norm(x) * norm
  head      f = rms_norm(sum_t alpha_t * tap_t of row p - 1) * markov dflare_norm, s = 0, previous = anchor;
            [g, c, l] = rnn_proj [s, markov_w1[previous], hidden_d, f]; s = sigmoid(g) s + (1 - sigmoid(g)) tanh(c);
            logits_d = soft_cap(lm_head hidden_d + lm_head (hidden_out tanh(l)))

The loader keeps the scales of the target's output normalization out of the draft (lalamo folds them into the LM
head): the test hands the drafter the final tap without those scales and an LM head that carries them, as the
target's own loader builds it, while the reference works with the original tensors. Agreement therefore covers the
loader's folds together with the forward pass. The remaining tests are properties that hold for any correct
implementation.
"""

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from lalamo.initializer import EmptyInitializer
from lalamo.model_import.loaders.dflare_loader import load_dflare_speculator
from lalamo.module import Keychain
from lalamo.modules.activations import SiLU
from lalamo.modules.linear import LinearConfig
from lalamo.modules.mlp import DenseMLPConfig
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.rope import UnscaledRoPEConfig
from lalamo.modules.speculators import (
    DFlareContext,
    DFlareDraftConfig,
    DFlareSpeculator,
    DFlareSpeculatorConfig,
    DSparkHeadConfig,
    MarkovHeadType,
    mix_taps,
)
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.modules.transformer_layer import SublayerConfig, TransformerLayerConfig
from lalamo.utils.sharding import ShardingConfig

CHANNELS, HEADS, GROUPS, HEAD_DIM, HIDDEN = 16, 4, 2, 4, 24
BLOCK, TAPS, RANK, LAYERS = 4, 3, 6, 3
VOCAB, UNPADDED_VOCAB, CHECKPOINT_VOCAB = 24, 21, 26
TRAINED_WINDOW = 3  # of the sliding layers 0 and 2; layer 1 sees the whole context
ROPE_BASE, EPSILON, SOFT_CAP = 100.0, 1e-6, 3.0
ROWS = 9
# float32 against a float64 reference through three layers: rounding of 1.2e-7 per operation, amplified by the
# cancellations of the residual sums and the normalizations; quantities are of order one.
RTOL, ATOL = 2e-4, 2e-5

Checkpoint = dict[str, np.ndarray]


def speculator_config(
    windows: tuple[int | None, ...] = (TRAINED_WINDOW, None, TRAINED_WINDOW),
) -> DFlareSpeculatorConfig:
    linear_config = LinearConfig()
    norm_config = NormalizationConfig(
        epsilon=EPSILON,
        scale_offset=None,
        upcast_mode=UpcastMode.FULL_LAYER,
        subtract_mean=False,
    )
    rope_config = UnscaledRoPEConfig(base=ROPE_BASE, max_sequence_length=64, head_dim=HEAD_DIM)
    layer_configs = tuple(
        TransformerLayerConfig(
            mixer=SublayerConfig.pre_norm(
                norm_config,
                AttentionConfig(
                    qkvg_projection_config=linear_config,
                    out_projection_config=linear_config,
                    query_norm_config=norm_config,
                    key_norm_config=norm_config,
                    num_heads=HEADS,
                    num_groups=GROUPS,
                    head_dim=HEAD_DIM,
                    is_causal=False,
                    scale=HEAD_DIM**-0.5,
                    # The import config stores twice the trained window of a sliding layer.
                    sliding_window_size=None if window is None else 2 * window,
                    logit_soft_cap=None,
                    has_sinks=False,
                    has_qkvg_biases=False,
                    has_out_biases=False,
                ),
            ),
            mlp=SublayerConfig.pre_norm(
                norm_config,
                DenseMLPConfig(
                    linear_config=linear_config,
                    activation=SiLU(),
                    has_up_biases=False,
                    has_down_biases=False,
                    gate_clipping=None,
                    up_clipping=None,
                ),
            ),
            rope_config=rope_config,
        )
        for window in windows
    )
    return DFlareSpeculatorConfig(
        draft_config=DFlareDraftConfig(
            model_dim=CHANNELS,
            hidden_dim=HIDDEN,
            block_size=BLOCK,
            target_layer_ids=(1, 3, 5),
            num_target_layers=6,
            vocab_size=VOCAB,
            context_projection_config=linear_config,
            context_norm_config=norm_config,
            rope_config=rope_config,
            layer_configs=layer_configs,
            output_norm_config=norm_config,
        ),
        dspark_config=DSparkHeadConfig(
            markov_rank=RANK,
            markov_head_type=MarkovHeadType.RNN_HIDDEN,
            projection_config=linear_config,
            norm_config=norm_config,
            logit_soft_cap=SOFT_CAP,
        ),
    )


def random_checkpoint(seed: int) -> Checkpoint:
    """Tensors under the names of the original checkpoint, float32 values of order one after every operation."""
    generator = np.random.default_rng(seed)

    def matrix(rows: int, columns: int) -> np.ndarray:
        return (generator.normal(size=(rows, columns)) / np.sqrt(columns)).astype(np.float32)

    def scales(size: int) -> np.ndarray:
        return generator.uniform(0.5, 1.5, size=size).astype(np.float32)

    checkpoint: Checkpoint = {
        "fuse_weight": generator.normal(size=(LAYERS + 1, TAPS)).astype(np.float32),
        "mask_embed": generator.normal(size=CHANNELS).astype(np.float32),
        "norm.weight": scales(CHANNELS),
        "markov_head.markov_w1.weight": generator.normal(size=(CHECKPOINT_VOCAB, RANK)).astype(np.float32),
        "markov_head.rnn_proj.weight": matrix(3 * RANK, 2 * RANK + 2 * CHANNELS),
        "markov_head.rnn_proj.bias": (0.3 * generator.normal(size=3 * RANK)).astype(np.float32),
        "markov_head.hidden_out.weight": matrix(CHANNELS, RANK),
        "markov_head.hidden_out.bias": (0.3 * generator.normal(size=CHANNELS)).astype(np.float32),
        "markov_head.dflare_norm.weight": scales(CHANNELS),
    }
    for layer in range(LAYERS):
        prefix = f"layers.{layer}"
        checkpoint |= {
            f"{prefix}.dflare_norm.weight": scales(CHANNELS),
            f"{prefix}.input_layernorm.weight": scales(CHANNELS),
            f"{prefix}.post_attention_layernorm.weight": scales(CHANNELS),
            f"{prefix}.self_attn.q_proj.weight": matrix(HEADS * HEAD_DIM, CHANNELS),
            f"{prefix}.self_attn.k_proj.weight": matrix(GROUPS * HEAD_DIM, CHANNELS),
            f"{prefix}.self_attn.v_proj.weight": matrix(GROUPS * HEAD_DIM, CHANNELS),
            f"{prefix}.self_attn.o_proj.weight": matrix(CHANNELS, HEADS * HEAD_DIM),
            f"{prefix}.self_attn.q_norm.weight": scales(HEAD_DIM),
            f"{prefix}.self_attn.k_norm.weight": scales(HEAD_DIM),
            f"{prefix}.self_attn.k_proj_ctx.weight": matrix(GROUPS * HEAD_DIM, CHANNELS),
            f"{prefix}.self_attn.v_proj_ctx.weight": matrix(GROUPS * HEAD_DIM, CHANNELS),
            f"{prefix}.self_attn.k_norm_ctx.weight": scales(HEAD_DIM),
            f"{prefix}.mlp.gate_proj.weight": matrix(HIDDEN, CHANNELS),
            f"{prefix}.mlp.up_proj.weight": matrix(HIDDEN, CHANNELS),
            f"{prefix}.mlp.down_proj.weight": matrix(CHANNELS, HIDDEN),
        }
    return checkpoint


class Problem:
    """One random checkpoint with a target (its output-norm scales and LM head) and committed rows of taps."""

    def __init__(self, seed: int, windows: tuple[int | None, ...] = (TRAINED_WINDOW, None, TRAINED_WINDOW)) -> None:
        generator = np.random.default_rng(1000 + seed)
        self.windows = windows
        self.checkpoint = random_checkpoint(seed)
        self.output_norm_scales = generator.uniform(0.5, 1.5, size=CHANNELS).astype(np.float32)
        self.lm_head = (generator.normal(size=(UNPADDED_VOCAB, CHANNELS)) / np.sqrt(CHANNELS)).astype(np.float32)
        # What lalamo hands over: hidden states after layer visits, the final one normalized without scales.
        self.taps = generator.normal(size=(ROWS, TAPS, CHANNELS)).astype(np.float32)
        self.positions = np.arange(ROWS, dtype=np.int32)
        self.anchor_embedding = generator.normal(size=CHANNELS).astype(np.float32)
        self.sharding_config = ShardingConfig.replicated(jax.devices("cpu")[:1])
        self.keychain = Keychain.init(0, sharding_config=self.sharding_config)
        self.speculator: DFlareSpeculator = load_dflare_speculator(
            speculator_config(windows).init(EmptyInitializer(jnp.float32, self.sharding_config)),
            {name: jnp.asarray(tensor) for name, tensor in self.checkpoint.items()},
            jnp.asarray(self.output_norm_scales),
            UNPADDED_VOCAB,
        )

    @property
    def original_taps(self) -> np.ndarray:
        """The taps as the original model produces them: the final one carries the output-norm scales."""
        taps = self.taps.astype(np.float64).copy()
        taps[:, -1] *= self.output_norm_scales
        return taps

    @property
    def folded_lm_head(self) -> np.ndarray:
        """The LM head as the target's loader stores it: the output-norm scales multiplied into its columns."""
        return self.lm_head * self.output_norm_scales[None, :]

    def context(self, taps: np.ndarray | None = None, positions: np.ndarray | None = None) -> DFlareContext:
        return self.speculator.draft_model.project_context(
            jnp.asarray(self.taps if taps is None else taps),
            jnp.asarray(self.positions if positions is None else positions),
            keychain=self.keychain,
        )

    def hidden(
        self,
        position: int,
        taps: np.ndarray | None = None,
        positions: np.ndarray | None = None,
        valid: np.ndarray | None = None,
    ) -> np.ndarray:
        positions = self.positions if positions is None else positions
        valid = np.ones(positions.shape[0], dtype=bool) if valid is None else valid
        return np.asarray(
            self.speculator.draft_model(
                jnp.asarray(self.anchor_embedding),
                self.context(taps, positions),
                jnp.asarray(positions),
                jnp.asarray(valid),
                position,
                keychain=self.keychain,
            ),
        )


# --- the reference, in the terms of the original checkpoint --------------------------------------------------------


def rms_norm(values: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return values / np.sqrt(np.mean(values**2) + EPSILON) * scales


def rotate(head: np.ndarray, position: int) -> np.ndarray:
    half = HEAD_DIM // 2
    rotated = np.zeros(HEAD_DIM)
    for pair in range(half):
        angle = position * ROPE_BASE ** (-2.0 * pair / HEAD_DIM)
        first, second = head[pair], head[pair + half]
        rotated[pair] = first * np.cos(angle) - second * np.sin(angle)
        rotated[pair + half] = second * np.cos(angle) + first * np.sin(angle)
    return rotated


def mix_weights(checkpoint: Checkpoint, row: int) -> np.ndarray:
    logits = checkpoint["fuse_weight"][row].astype(np.float64)
    alpha = np.exp(logits - logits.max())
    return (alpha / alpha.sum()).astype(np.float32).astype(ml_dtypes.bfloat16).astype(np.float64)


def reference_mix(checkpoint: Checkpoint, row: int, taps: np.ndarray, norm_name: str) -> np.ndarray:
    alpha = mix_weights(checkpoint, row)
    mixed = np.zeros(CHANNELS)
    for tap in range(TAPS):
        mixed += alpha[tap] * taps[tap]
    return rms_norm(mixed, checkpoint[norm_name].astype(np.float64))


def reference_context(problem: Problem, layer: int, row: int) -> tuple[np.ndarray, np.ndarray]:
    """(keys, values) [groups, head_dim] of one committed row for one layer."""
    checkpoint = problem.checkpoint
    prefix = f"layers.{layer}"
    mixed = reference_mix(checkpoint, layer, problem.original_taps[row], f"{prefix}.dflare_norm.weight")
    keys = checkpoint[f"{prefix}.self_attn.k_proj_ctx.weight"].astype(np.float64) @ mixed
    values = checkpoint[f"{prefix}.self_attn.v_proj_ctx.weight"].astype(np.float64) @ mixed
    key_scales = checkpoint[f"{prefix}.self_attn.k_norm_ctx.weight"].astype(np.float64)
    position = int(problem.positions[row])
    return (
        np.stack(
            [
                rotate(rms_norm(keys[group * HEAD_DIM : (group + 1) * HEAD_DIM], key_scales), position)
                for group in range(GROUPS)
            ],
        ),
        values.reshape(GROUPS, HEAD_DIM),
    )


def reference_hidden(problem: Problem, position: int, valid: np.ndarray | None = None) -> np.ndarray:
    """Hidden states [block - 1, channels] of the proposal positions in the ORIGINAL space (with `norm.weight`)."""
    checkpoint = problem.checkpoint
    valid = np.ones(ROWS, dtype=bool) if valid is None else valid
    stream = [problem.anchor_embedding.astype(np.float64)] + [
        checkpoint["mask_embed"].astype(np.float64) for _ in range(BLOCK - 1)
    ]
    for layer, window in enumerate(problem.windows):
        prefix = f"layers.{layer}"

        def weight(name: str, prefix: str = prefix) -> np.ndarray:
            return checkpoint[f"{prefix}.{name}"].astype(np.float64)

        visible = [
            row
            for row in range(ROWS)
            if valid[row]
            and problem.positions[row] < position
            and (window is None or problem.positions[row] >= position - (window - 1))
        ]
        context = [reference_context(problem, layer, row) for row in visible]
        queries, keys, values = [], [], []
        for index, row in enumerate(stream):
            normalized = rms_norm(row, weight("input_layernorm.weight"))
            query = weight("self_attn.q_proj.weight") @ normalized
            key = weight("self_attn.k_proj.weight") @ normalized
            value = weight("self_attn.v_proj.weight") @ normalized
            queries.append(
                [
                    rotate(
                        rms_norm(query[head * HEAD_DIM : (head + 1) * HEAD_DIM], weight("self_attn.q_norm.weight")),
                        position + index,
                    )
                    for head in range(HEADS)
                ],
            )
            keys.append(
                [
                    rotate(
                        rms_norm(key[group * HEAD_DIM : (group + 1) * HEAD_DIM], weight("self_attn.k_norm.weight")),
                        position + index,
                    )
                    for group in range(GROUPS)
                ],
            )
            values.append(value.reshape(GROUPS, HEAD_DIM))
        all_keys = [key for key, _ in context] + keys
        all_values = [value for _, value in context] + values
        updated = []
        for index, row in enumerate(stream):
            attended = np.zeros(HEADS * HEAD_DIM)
            for head in range(HEADS):
                group = head // (HEADS // GROUPS)
                logits = np.asarray([HEAD_DIM**-0.5 * (queries[index][head] @ key[group]) for key in all_keys])
                weights = np.exp(logits - logits.max())
                weights /= weights.sum()
                for source, source_weight in enumerate(weights):
                    attended[head * HEAD_DIM : (head + 1) * HEAD_DIM] += source_weight * all_values[source][group]
            mixed = row + weight("self_attn.o_proj.weight") @ attended
            normalized = rms_norm(mixed, weight("post_attention_layernorm.weight"))
            gate = weight("mlp.gate_proj.weight") @ normalized
            up = weight("mlp.up_proj.weight") @ normalized
            updated.append(mixed + weight("mlp.down_proj.weight") @ (gate / (1.0 + np.exp(-gate)) * up))
        stream = updated
    return np.stack([rms_norm(row, checkpoint["norm.weight"].astype(np.float64)) for row in stream[1:]])


def reference_chain(
    problem: Problem,
    position: int,
    anchor: int,
    path: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """(logits [len(path), unpadded vocabulary], states after every step) of the tokens after `anchor`, `path[d]`
    being the token chosen at proposal position d; the last token of the path conditions nothing."""
    checkpoint = problem.checkpoint
    hidden = reference_hidden(problem, position)
    context = reference_mix(
        checkpoint,
        LAYERS,
        problem.original_taps[position - 1],
        "markov_head.dflare_norm.weight",
    )
    lm_head = problem.lm_head.astype(np.float64)
    state = np.zeros(RANK)
    previous = anchor
    logits, states = [], []
    for depth in range(len(path)):
        embedding = (
            checkpoint["markov_head.markov_w1.weight"][previous].astype(np.float64)
            if previous < UNPADDED_VOCAB
            else np.zeros(RANK)
        )
        inputs = np.concatenate([state, embedding, hidden[depth], context])
        projected = (
            checkpoint["markov_head.rnn_proj.weight"].astype(np.float64) @ inputs
            + checkpoint["markov_head.rnn_proj.bias"]
        )
        gate = 1.0 / (1.0 + np.exp(-projected[:RANK]))
        state = gate * state + (1.0 - gate) * np.tanh(projected[RANK : 2 * RANK])
        bias_hidden = (
            checkpoint["markov_head.hidden_out.weight"].astype(np.float64) @ np.tanh(projected[2 * RANK :])
            + checkpoint["markov_head.hidden_out.bias"]
        )
        raw = lm_head @ hidden[depth] + lm_head @ bias_hidden
        logits.append(np.tanh(raw / SOFT_CAP) * SOFT_CAP)
        states.append(state)
        previous = path[depth]
    return np.stack(logits), np.stack(states)


def chain(problem: Problem, position: int, anchor: int, path: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """The same chain through the modules: one head step per node of the path."""
    head = problem.speculator.dspark_head
    hidden = jnp.asarray(problem.hidden(position))
    context = head.mix_context(jnp.asarray(problem.taps[position - 1]))
    unembedding = jnp.asarray(problem.folded_lm_head)
    state = head.initial_state()
    previous = anchor
    logits, states = [], []
    for depth in range(len(path)):
        correction, state = head.step(state, previous, hidden[depth], context, keychain=problem.keychain)
        logits.append(np.asarray(head.logits(hidden[depth], correction, unembedding)))
        states.append(np.asarray(state))
        previous = path[depth]
    return np.stack(logits), np.stack(states)


# --- agreement with the reference ----------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1])
def test_context_keys_and_values_match_the_reference(seed: int) -> None:
    # Reference: `reference_context`, from the raw `fuse_weight`, `dflare_norm`, `k_proj_ctx`, `v_proj_ctx` and
    # `k_norm_ctx`. Catches a wrong tap mix, a missing bfloat16 rounding of the mix weights, the fold of the
    # output-norm scales into the wrong tap, swapped keys and values, a missing key norm or rotation.
    problem = Problem(seed)
    context = problem.context()
    assert context.keys.shape == context.values.shape == (LAYERS, ROWS, GROUPS, HEAD_DIM)
    for layer in range(LAYERS):
        for row in range(ROWS):
            keys, values = reference_context(problem, layer, row)
            np.testing.assert_allclose(np.asarray(context.keys[layer, row]), keys, rtol=RTOL, atol=ATOL)
            np.testing.assert_allclose(np.asarray(context.values[layer, row]), values, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("position", [0, 1, 2, 5, ROWS])
def test_base_logits_of_the_block_match_the_reference(seed: int, position: int) -> None:
    # Reference: `reference_hidden` in the original space, turned into logits by the original LM head. The
    # drafter's hidden states live in lalamo's space (without the output-norm scales) and meet the LM head that
    # carries them, so equal logits check the fold. Positions cover no context, less than the window, exactly
    # the window, more than the window, and all rows.
    problem = Problem(seed)
    hidden = problem.hidden(position)
    assert hidden.shape == (BLOCK - 1, CHANNELS)
    assert hidden.dtype == np.float32
    expected = reference_hidden(problem, position) @ problem.lm_head.astype(np.float64).T
    np.testing.assert_allclose(hidden @ problem.folded_lm_head.T, expected, rtol=RTOL, atol=ATOL)


def test_rows_marked_invalid_are_left_out_of_the_block(seed: int = 2) -> None:
    # Reference with the same rows removed: an invalid row is padding, not context.
    problem = Problem(seed)
    valid = np.asarray([True, False, True, True, False, True, True, True, True])
    expected = reference_hidden(problem, 7, valid) @ problem.lm_head.astype(np.float64).T
    np.testing.assert_allclose(
        problem.hidden(7, valid=valid) @ problem.folded_lm_head.T,
        expected,
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.parametrize("seed", [0, 1])
def test_head_logits_and_states_along_a_path_match_the_reference(seed: int) -> None:
    # Reference: `reference_chain`, the left-to-right loop of the serving implementation with the anchor as the
    # first previous token. Token 22 lies past the unpadded vocabulary: its embedding row is dropped by the loader.
    problem = Problem(seed)
    for anchor, path in ((3, (7, 0, 19)), (20, (22, 5, 5))):
        expected_logits, expected_states = reference_chain(problem, 6, anchor, path)
        logits, states = chain(problem, 6, anchor, path)
        np.testing.assert_allclose(logits, expected_logits, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(states, expected_states, rtol=RTOL, atol=ATOL)


def test_tap_mix_weights_are_the_bfloat16_softmax_with_the_final_norm_folded_in() -> None:
    # Spec of the loader: weight[tap, channel] = bfloat16(softmax(fuse_weight row))[tap], times the scales of the
    # target's output normalization for the final tap only.
    problem = Problem(3)
    draft, head = problem.speculator.draft_model, problem.speculator.dspark_head
    loaded = [np.asarray(context.tap_weights) for context in draft.contexts] + [np.asarray(head.context_tap_weights)]
    for row, weights in enumerate(loaded):
        alpha = mix_weights(problem.checkpoint, row)
        expected = np.repeat(alpha[:, None], CHANNELS, axis=1)
        expected[-1] *= problem.output_norm_scales
        np.testing.assert_allclose(weights, expected, rtol=1e-6, atol=0.0)


# --- properties ----------------------------------------------------------------------------------------------------


def test_context_rows_are_projected_independently() -> None:
    # MR: a row's keys and values depend on its own taps and position only, so a subset of the rows gives the
    # same rows of the result. Catches any mixing along the row axis.
    problem = Problem(4)
    rows = np.asarray([6, 1, 4])
    whole = problem.context()
    part = problem.context(problem.taps[rows], problem.positions[rows])
    np.testing.assert_array_equal(np.asarray(part.keys), np.asarray(whole.keys)[:, rows])
    np.testing.assert_array_equal(np.asarray(part.values), np.asarray(whole.values)[:, rows])


def test_the_block_reads_only_committed_rows_before_the_anchor() -> None:
    # Spec: context rows count when they are valid and lie before the anchor. Changing the taps of the other
    # rows must leave the result bit for bit; changing a counted row must not.
    problem = Problem(5)
    position = 5
    valid = np.asarray([True, True, True, False, True, True, True, True, True])
    base = problem.hidden(position, valid=valid)
    ignored = problem.taps.copy()
    ignored[[3, 5, 6, 7, 8]] += 1.0
    np.testing.assert_array_equal(problem.hidden(position, taps=ignored, valid=valid), base)
    counted = problem.taps.copy()
    counted[4] += 1.0
    assert np.abs(problem.hidden(position, taps=counted, valid=valid) - base).max() > 1e-3


def test_a_sliding_layer_sees_one_row_fewer_than_its_trained_window() -> None:
    # Spec: the trained window counts the anchor, so `window - 1` context rows are visible, the same ones for
    # every row of the block. With every layer sliding, older rows cannot matter at all; the newest row outside
    # the window is the edge case.
    problem = Problem(6, windows=(TRAINED_WINDOW,) * LAYERS)
    assert [problem.speculator.draft_model.context_window(layer) for layer in range(LAYERS)] == [
        TRAINED_WINDOW - 1,
    ] * LAYERS
    position = 7
    base = problem.hidden(position)
    outside = problem.taps.copy()
    outside[: position - (TRAINED_WINDOW - 1)] += 1.0
    np.testing.assert_array_equal(problem.hidden(position, taps=outside), base)
    edge = problem.taps.copy()
    edge[position - (TRAINED_WINDOW - 1)] += 1.0
    assert np.abs(problem.hidden(position, taps=edge) - base).max() > 1e-3


def test_a_layer_without_a_window_sees_the_whole_context() -> None:
    # Spec: `sliding_window_size=None` is full attention. One such layer is enough for old rows to matter.
    problem = Problem(6)
    assert problem.speculator.draft_model.context_window(1) is None
    position = 7
    old = problem.taps.copy()
    old[0] += 1.0
    assert np.abs(problem.hidden(position, taps=old) - problem.hidden(position)).max() > 1e-3


def test_the_order_of_context_rows_does_not_matter() -> None:
    # MR: attention is a sum over keys and every row carries its own position, so permuting the rows together
    # with their positions and validity changes only the order of float32 additions.
    problem = Problem(7)
    order = np.random.default_rng(0).permutation(ROWS)
    valid = np.asarray([True, True, False, True, True, True, True, True, True])
    np.testing.assert_allclose(
        problem.hidden(6, problem.taps[order], problem.positions[order], valid[order]),
        problem.hidden(6, valid=valid),
        rtol=1e-5,
        atol=1e-6,
    )


def test_an_empty_context_gives_finite_hidden_states() -> None:
    # Implicit oracle: the first block of a sequence has nothing committed; no row, or only invalid rows.
    problem = Problem(8)
    none = problem.hidden(0, problem.taps[:0], problem.positions[:0])
    masked = problem.hidden(4, valid=np.zeros(ROWS, dtype=bool))
    assert none.shape == masked.shape == (BLOCK - 1, CHANNELS)
    assert np.isfinite(none).all() and np.isfinite(masked).all()


def test_anchors_mapped_together_agree_with_anchors_one_at_a_time() -> None:
    # MR: blocks of different anchors do not interact. Catches state shared between calls and a position that
    # leaks from one anchor into another.
    problem = Problem(9)
    draft, context = problem.speculator.draft_model, problem.context()
    embeddings = np.random.default_rng(1).normal(size=(3, CHANNELS)).astype(np.float32)
    positions = np.asarray([2, 6, 9], dtype=np.int32)

    def one(embedding: jax.Array, position: jax.Array) -> jax.Array:
        return draft(
            embedding,
            context,
            jnp.asarray(problem.positions),
            jnp.ones(ROWS, dtype=jnp.bool_),
            position,
            keychain=problem.keychain,
        )

    together = np.asarray(jax.vmap(one)(jnp.asarray(embeddings), jnp.asarray(positions)))
    for index in range(3):
        alone = np.asarray(one(jnp.asarray(embeddings[index]), jnp.asarray(positions[index])))
        np.testing.assert_allclose(together[index], alone, rtol=1e-5, atol=1e-6)


def test_the_tap_mix_ignores_the_overall_scale_of_the_taps() -> None:
    # MR: rms_norm(c x) = rms_norm(x) for c > 0 up to the epsilon inside the root, relatively 1e-6 / mean(x^2).
    problem = Problem(10)
    context = problem.speculator.draft_model.contexts[0]
    taps = jnp.asarray(problem.taps[2])
    np.testing.assert_allclose(
        np.asarray(mix_taps(context.tap_weights, context.norm, 7.0 * taps)),
        np.asarray(mix_taps(context.tap_weights, context.norm, taps)),
        rtol=1e-5,
        atol=1e-6,
    )


def test_a_head_step_depends_only_on_the_path_to_its_node() -> None:
    # MR of a tree: two paths that share a prefix give the same logits and states along the prefix, and differ
    # right after the first different token (the logits at depth d are conditioned on the tokens before d).
    problem = Problem(11)
    logits_a, states_a = chain(problem, 6, 4, (9, 2, 13))
    logits_b, states_b = chain(problem, 6, 4, (9, 15, 13))
    np.testing.assert_array_equal(logits_a[:2], logits_b[:2])
    np.testing.assert_array_equal(states_a[:2], states_b[:2])
    assert np.abs(logits_a[2] - logits_b[2]).max() > 1e-3
    assert np.abs(states_a[2] - states_b[2]).max() > 1e-4


def test_sibling_nodes_stepped_together_agree_with_siblings_stepped_alone() -> None:
    # MR: the step is a function of one node, so mapping it over the children of a parent (one state, one
    # hidden row, different tokens) gives what separate calls give.
    problem = Problem(12)
    head = problem.speculator.dspark_head
    hidden = jnp.asarray(problem.hidden(5))
    context = head.mix_context(jnp.asarray(problem.taps[4]))
    _, parent_state = head.step(head.initial_state(), 3, hidden[0], context, keychain=problem.keychain)
    tokens = jnp.asarray([1, 8, 17, 20], dtype=jnp.int32)

    def child(token: jax.Array) -> tuple[jax.Array, jax.Array]:
        return head.step(parent_state, token, hidden[1], context, keychain=problem.keychain)

    corrections, states = jax.vmap(child)(tokens)
    assert corrections.shape == (4, CHANNELS) and states.shape == (4, RANK)
    for index in range(4):
        correction, state = child(tokens[index])
        np.testing.assert_allclose(np.asarray(corrections[index]), np.asarray(correction), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np.asarray(states[index]), np.asarray(state), rtol=1e-5, atol=1e-6)


def test_logits_of_a_pool_are_the_full_logits_at_the_pool() -> None:
    # MR: the LM head acts row by row, so restricting it to a pool of tokens restricts the logits; the soft cap
    # is elementwise and bounds them.
    problem = Problem(13)
    head = problem.speculator.dspark_head
    hidden = jnp.asarray(problem.hidden(4))[0]
    correction, _ = head.step(
        head.initial_state(),
        2,
        hidden,
        head.mix_context(jnp.asarray(problem.taps[3])),
        keychain=problem.keychain,
    )
    unembedding = jnp.asarray(problem.folded_lm_head)
    pool = jnp.asarray([14, 0, 7])
    full = np.asarray(head.logits(hidden, correction, unembedding))
    np.testing.assert_allclose(
        np.asarray(head.logits(hidden, correction, unembedding[pool])),
        full[np.asarray(pool)],
        rtol=1e-6,
        atol=1e-6,
    )
    assert np.abs(full).max() <= SOFT_CAP


def test_the_initial_state_is_zero_and_the_step_returns_float32() -> None:
    # Spec: the recurrence starts from a zero state at the anchor; results are float32 whatever comes in.
    problem = Problem(14)
    head = problem.speculator.dspark_head
    state = head.initial_state()
    assert state.shape == (RANK,) and not np.asarray(state).any()
    hidden = jnp.asarray(problem.hidden(3))[0].astype(jnp.bfloat16)
    context = head.mix_context(jnp.asarray(problem.taps[2]).astype(jnp.bfloat16))
    correction, next_state = head.step(state.astype(jnp.bfloat16), 5, hidden, context, keychain=problem.keychain)
    assert correction.dtype == next_state.dtype == jnp.float32
    assert correction.shape == (CHANNELS,) and next_state.shape == (RANK,)
    assert np.isfinite(np.asarray(correction)).all() and np.isfinite(np.asarray(next_state)).all()


def test_repeated_calls_return_identical_results() -> None:
    # Implicit oracle: no state survives a call.
    problem = Problem(15)
    first = problem.hidden(6)
    np.testing.assert_array_equal(problem.hidden(2), problem.hidden(2))
    np.testing.assert_array_equal(problem.hidden(6), first)
