import math
import warnings
from collections.abc import Callable, Sequence
from enum import Enum, IntEnum
from functools import cache, reduce
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax._src.pallas.mosaic_gpu.core import ParameterizedLayout
from jax.experimental import pallas as pl
from jax.experimental.mosaic import gpu as mgpu
from jax.experimental.mosaic.gpu import utils as mgpu_utils
from jax.experimental.pallas import mosaic_gpu as plgpu
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith, gpu, llvm, nvvm, vector
from jaxtyping import Array, Float, Int

from lalamo.kernels.mosaic import supports_mosaic_gpu
from lalamo.utils.sharding import sharding_of

from .xla import xla_top_k_blocks

__all__ = ["pallas_top_k_blocks"]

NUM_BINS = 2048
NUM_WARPGROUPS = 4
NUM_THREADS = 128 * NUM_WARPGROUPS
EMPTY_SLOT = 2**31 - 1


class StateSlot(IntEnum):
    THRESHOLD = 0
    NEED = 1
    BIN_COUNT = 2
    CUT = 3
    COUNTER = 4
    WARP_TOTALS = 8


class ScanDirection(Enum):
    DESCENDING = "descending"
    ASCENDING = "ascending"


class SelectionStage(Enum):
    FULL_ROWS = "full_rows"
    SEGMENT_CANDIDATES = "segment_candidates"
    MERGE_CANDIDATES = "merge_candidates"


def i32_constant(value: int) -> ir.Value:
    return mgpu_utils.c(value, ir.IntegerType.get_signless(32))


def register_elements(array: mgpu.FragmentedArray) -> list[ir.Value]:
    elements = []
    for register in array.registers.flat:
        [width] = ir.VectorType(register.type).shape
        elements.extend(
            vector.extract(register, dynamic_position=[], static_position=ir.DenseI64ArrayAttr.get([lane]))
            for lane in range(width)
        )
    return elements


def shared_pointer(ref: ir.Value, offset: ir.Value | int) -> ir.Value:
    return mgpu_utils.getelementptr(mgpu_utils.memref_ptr(ref), [offset], ir.IntegerType.get_signless(32))


def load_shared(ref: ir.Value, offset: ir.Value | int) -> ir.Value:
    return llvm.load(ir.IntegerType.get_signless(32), shared_pointer(ref, offset))


def store_shared(value: ir.Value, ref: ir.Value, offset: ir.Value | int) -> None:
    llvm.store(value, shared_pointer(ref, offset))


def warp_scan_step(value: ir.Value, lane: ir.Value, distance: int, direction: ScanDirection) -> ir.Value:
    match direction:
        case ScanDirection.DESCENDING:
            shuffle_kind, clamp = nvvm.ShflKind.down, i32_constant(31)
            has_neighbor = arith.cmpi(
                arith.CmpIPredicate.ult, arith.addi(lane, i32_constant(distance)), i32_constant(32)
            )
        case ScanDirection.ASCENDING:
            shuffle_kind, clamp = nvvm.ShflKind.up, i32_constant(0)
            has_neighbor = arith.cmpi(arith.CmpIPredicate.uge, lane, i32_constant(distance))
    neighbor = mgpu_utils.nvvm_shfl_sync(
        value.type,
        i32_constant(-1),
        value,
        i32_constant(distance),
        clamp,
        shuffle_kind,
    )
    return arith.addi(value, arith.select(has_neighbor, neighbor, i32_constant(0)))


@plgpu.inline_mgpu()
def block_barrier(_ctx: object) -> None:
    gpu.barrier()


def make_set_state(entries: Sequence[tuple[StateSlot, int]]) -> Callable[[jax.Ref], None]:
    @plgpu.inline_mgpu(arg_types=(plgpu.RefType(),))
    def set_state(_ctx: object, state_ref: ir.Value) -> None:
        with mgpu_utils.when(arith.cmpi(arith.CmpIPredicate.eq, mgpu_utils.thread_idx(), i32_constant(0))):
            for slot, value in entries:
                store_shared(i32_constant(value), state_ref, int(slot))
        gpu.barrier()

    return set_state


def make_histogram_add(layout: ParameterizedLayout) -> Callable[[Array, jax.Ref], None]:
    @plgpu.inline_mgpu(arg_types=(layout, plgpu.RefType()))
    def histogram_add(_ctx: object, bins: mgpu.FragmentedArray, histogram_ref: ir.Value) -> None:
        base = mgpu_utils.memref_ptr(histogram_ref)
        for bin_index in register_elements(bins):
            llvm.inline_asm(
                ir.Type.parse("!llvm.void"),
                [base, bin_index],
                "{ .reg .pred p; .reg .u32 a; setp.ge.s32 p, $1, 0; shl.b32 a, $1, 2; add.u32 a, a, $0;"
                " @p red.shared::cta.add.u32 [a], 1; }",
                "r,r",
                has_side_effects=True,
            )

    return histogram_add


def make_compact(
    layout: ParameterizedLayout,
    capacity: int,
) -> Callable[[Array, Array, jax.Ref, jax.Ref, jax.Ref], None]:
    @plgpu.inline_mgpu(arg_types=(layout, layout, plgpu.RefType(), plgpu.RefType(), plgpu.RefType()))
    def compact(
        _ctx: object,
        positions: mgpu.FragmentedArray,
        value_bits: mgpu.FragmentedArray,
        index_buffer_ref: ir.Value,
        value_buffer_ref: ir.Value,
        state_ref: ir.Value,
    ) -> None:
        counter = shared_pointer(state_ref, int(StateSlot.COUNTER))
        index_base = mgpu_utils.memref_ptr(index_buffer_ref)
        value_base = mgpu_utils.memref_ptr(value_buffer_ref)
        for position, bits in zip(register_elements(positions), register_elements(value_bits), strict=True):
            llvm.inline_asm(
                ir.Type.parse("!llvm.void"),
                [counter, index_base, value_base, position, bits],
                "{ .reg .pred p, q; .reg .u32 s, a; setp.ge.s32 p, $3, 0; @p atom.shared::cta.add.u32 s, [$0], 1;"
                f" setp.lt.and.u32 q, s, {capacity}, p; @q shl.b32 s, s, 2; @q add.u32 a, s, $1;"
                " @q st.shared.u32 [a], $3; @q add.u32 a, s, $2; @q st.shared.u32 [a], $4; }",
                "r,r,r,r,r",
                has_side_effects=True,
            )

    return compact


def make_find_bin(direction: ScanDirection, shift: int, target: StateSlot) -> Callable[[jax.Ref, jax.Ref], None]:
    bins_per_thread = NUM_BINS // NUM_THREADS

    @plgpu.inline_mgpu(arg_types=(plgpu.RefType(), plgpu.RefType()))
    def find_bin(_ctx: object, histogram_ref: ir.Value, state_ref: ir.Value) -> None:
        thread = mgpu_utils.thread_idx()
        lane = arith.andi(thread, i32_constant(31))
        warp = arith.shrui(thread, i32_constant(5))
        need = load_shared(state_ref, int(StateSlot.NEED))
        first_bin = arith.muli(thread, i32_constant(bins_per_thread))
        counts = [
            load_shared(histogram_ref, arith.addi(first_bin, i32_constant(offset)))
            for offset in range(bins_per_thread)
        ]
        total = reduce(arith.addi, counts)
        inclusive = total
        for distance in (1, 2, 4, 8, 16):
            inclusive = warp_scan_step(inclusive, lane, distance, direction)
        warp_leader = 0 if direction == ScanDirection.DESCENDING else 31
        with mgpu_utils.when(arith.cmpi(arith.CmpIPredicate.eq, lane, i32_constant(warp_leader))):
            store_shared(inclusive, state_ref, arith.addi(i32_constant(int(StateSlot.WARP_TOTALS)), warp))
        gpu.barrier()
        preceding_predicate = (
            arith.CmpIPredicate.ugt if direction == ScanDirection.DESCENDING else arith.CmpIPredicate.ult
        )
        running = arith.subi(inclusive, total)
        for other_warp in range(NUM_THREADS // 32):
            other_total = load_shared(state_ref, int(StateSlot.WARP_TOTALS) + other_warp)
            precedes = arith.cmpi(preceding_predicate, i32_constant(other_warp), warp)
            running = arith.addi(running, arith.select(precedes, other_total, i32_constant(0)))
        offsets = range(bins_per_thread)
        for offset in reversed(offsets) if direction == ScanDirection.DESCENDING else offsets:
            count = counts[offset]
            after = arith.addi(running, count)
            is_boundary = arith.andi(
                arith.cmpi(arith.CmpIPredicate.slt, running, need),
                arith.cmpi(arith.CmpIPredicate.sle, need, after),
            )
            with mgpu_utils.when(is_boundary):
                chosen = arith.shli(arith.addi(first_bin, i32_constant(offset)), i32_constant(shift))
                previous = load_shared(state_ref, int(target))
                store_shared(arith.ori(previous, chosen), state_ref, int(target))
                store_shared(arith.subi(need, running), state_ref, int(StateSlot.NEED))
                store_shared(count, state_ref, int(StateSlot.BIN_COUNT))
            running = after
        gpu.barrier()

    return find_bin


def make_rank_scatter(
    layout: ParameterizedLayout,
    padded_k: int,
) -> Callable[[Array, jax.Ref, jax.Ref], None]:
    @plgpu.inline_mgpu(arg_types=(layout, plgpu.RefType(), plgpu.RefType()))
    def rank_scatter(
        _ctx: object,
        indices: mgpu.FragmentedArray,
        index_buffer_ref: ir.Value,
        sorted_ref: ir.Value,
    ) -> None:
        own_indices = register_elements(indices)

        @mgpu_utils.fori(i32_constant(padded_k // 4), [i32_constant(0)] * len(own_indices))
        def ranks(step: ir.Value, current_ranks: list[ir.Value]) -> list[ir.Value]:
            first = arith.muli(step, i32_constant(4))
            others = [load_shared(index_buffer_ref, arith.addi(first, i32_constant(lane))) for lane in range(4)]
            updated_ranks = []
            for rank, index in zip(current_ranks, own_indices, strict=True):
                smaller_counts = [
                    arith.extui(ir.IntegerType.get_signless(32), arith.cmpi(arith.CmpIPredicate.slt, other, index))
                    for other in others
                ]
                updated_ranks.append(reduce(arith.addi, smaller_counts, rank))
            return updated_ranks

        for index, rank in zip(own_indices, ranks.results, strict=True):
            store_shared(index, sorted_ref, rank)
        gpu.barrier()

    return rank_scatter


def make_value_bins(shift: int, width: int, prefix: Array) -> Callable[[Array, Array, Array], Array]:
    high = shift + width

    def value_bins(keys: Array, _positions: Array, is_valid: Array) -> Array:
        digits = (keys >> shift) & ((1 << width) - 1)
        if high == 32:
            return jnp.where(is_valid, digits ^ (1 << (width - 1)), -1)
        is_candidate = is_valid & ((keys >> high) == (prefix >> high))
        return jnp.where(is_candidate, digits, -1)

    return value_bins


def make_position_bins(
    shift: int,
    width: int,
    tied_key: Array,
    cut_prefix: Array,
) -> Callable[[Array, Array, Array], Array]:
    def position_bins(keys: Array, positions: Array, is_valid: Array) -> Array:
        high = shift + width
        is_tied = is_valid & (keys == tied_key) & ((positions >> high) == (cut_prefix >> high))
        return jnp.where(is_tied, (positions >> shift) & ((1 << width) - 1), -1)

    return position_bins


@cache
def make_selection(
    num_programs: int,
    num_values: int,
    k: int,
    tile: int,
    num_segments: int,
    stage: SelectionStage,
) -> Callable[..., Any]:
    padded_k = math.ceil(k / 512) * 512
    share = padded_k // NUM_WARPGROUPS
    tile_layout = plgpu.Layout.WG_STRIDED((tile,), vec_size=4)
    share_layout = plgpu.Layout.WG_STRIDED((share,), vec_size=share // 128)
    histogram_add = make_histogram_add(tile_layout)
    compact = make_compact(tile_layout, padded_k)
    rank_scatter = make_rank_scatter(share_layout, padded_k)
    num_tiles = math.ceil(num_values / tile)
    tiles_per_segment = math.ceil(num_tiles / num_segments)
    steps_per_segment = math.ceil(tiles_per_segment / NUM_WARPGROUPS)

    reads_positions = stage == SelectionStage.MERGE_CANDIDATES
    sorts_output = stage != SelectionStage.SEGMENT_CANDIDATES

    def kernel(
        values_ref: jax.Ref,
        visible_or_positions_ref: jax.Ref,
        out_indices_ref: jax.Ref,
        out_values_ref: jax.Ref,
        histogram_ref: jax.Ref,
        state_ref: jax.Ref,
        index_buffer_ref: jax.Ref,
        value_buffer_ref: jax.Ref,
        sorted_ref: jax.Ref,
    ) -> None:
        program = pl.program_id(0)
        warpgroup = lax.axis_index("warpgroup")
        row = program // num_segments
        first_tile = (program % num_segments) * tiles_per_segment
        end_tile = jnp.minimum(first_tile + tiles_per_segment, num_tiles)
        if reads_positions:
            visible = num_values
        else:
            visible = jnp.minimum(visible_or_positions_ref[row], num_values)
        num_valid = jnp.clip(visible - first_tile * tile, 0, (end_tile - first_tile) * tile)

        def for_each_tile(body: Callable[[Array, Array, Array, Array], None]) -> None:
            def step(step_index: Array, carry: Array) -> Array:
                tile_index = first_tile + step_index * NUM_WARPGROUPS + warpgroup

                @pl.when(tile_index < end_tile)
                def _() -> None:
                    tile_start = tile_index * tile
                    load_start = jnp.minimum(tile_start, num_values - tile)
                    tile_slice = pl.ds(load_start, tile)
                    values = plgpu.load(values_ref.at[row, tile_slice], layout=tile_layout, optimized=False)
                    signed_bits = lax.bitcast_convert_type(values, jnp.int32)
                    keys = signed_bits ^ ((signed_bits >> 31) & 0x7FFFFFFF)
                    offsets = load_start + plgpu.broadcasted_iota(jnp.int32, (tile,), 0, layout=tile_layout)
                    if reads_positions:
                        positions = plgpu.load(
                            visible_or_positions_ref.at[row, tile_slice],
                            layout=tile_layout,
                            optimized=False,
                        )
                        is_valid = (offsets >= tile_start) & (positions >= 0)
                    else:
                        positions = offsets
                        is_valid = (offsets >= tile_start) & (offsets < visible)
                    body(signed_bits, keys, positions, is_valid)

                return carry

            lax.fori_loop(0, steps_per_segment, step, 0)

        def signed_threshold() -> Array:
            return state_ref[StateSlot.THRESHOLD] ^ -(2**31)

        def histogram_pass(bins_of: Callable[[Array, Array, Array], Array]) -> None:
            @pl.when(warpgroup == 0)
            def _() -> None:
                histogram_ref[...] = jnp.zeros((NUM_BINS,), jnp.int32)

            block_barrier()

            def accumulate(_bits: Array, keys: Array, positions: Array, is_valid: Array) -> None:
                histogram_add(bins_of(keys, positions, is_valid), histogram_ref)

            for_each_tile(accumulate)
            block_barrier()

        make_set_state(
            (
                (StateSlot.THRESHOLD, 0),
                (StateSlot.NEED, k),
                (StateSlot.BIN_COUNT, 0),
                (StateSlot.CUT, EMPTY_SLOT),
                (StateSlot.COUNTER, 0),
            ),
        )(state_ref)

        @pl.when(num_valid > k)
        def _() -> None:
            for shift, width in ((21, 11), (10, 11), (0, 10)):
                histogram_pass(make_value_bins(shift, width, signed_threshold()))
                make_find_bin(ScanDirection.DESCENDING, shift, StateSlot.THRESHOLD)(histogram_ref, state_ref)

            @pl.when(state_ref[StateSlot.BIN_COUNT] > state_ref[StateSlot.NEED])
            def _() -> None:
                make_set_state(((StateSlot.CUT, 0),))(state_ref)
                tied_key = signed_threshold()
                for shift, width in ((11, 11), (0, 11)):
                    histogram_pass(make_position_bins(shift, width, tied_key, state_ref[StateSlot.CUT]))
                    make_find_bin(ScanDirection.ASCENDING, shift, StateSlot.CUT)(histogram_ref, state_ref)

        threshold = signed_threshold()
        cut = state_ref[StateSlot.CUT]
        share_slice = pl.ds(warpgroup * share, share)
        index_buffer_ref.at[share_slice][...] = jnp.full((share,), EMPTY_SLOT if sorts_output else -1, jnp.int32)
        value_buffer_ref.at[share_slice][...] = jnp.zeros((share,), jnp.int32)
        sorted_ref.at[share_slice][...] = jnp.full((share,), EMPTY_SLOT, jnp.int32)
        block_barrier()

        def emit(bits: Array, keys: Array, positions: Array, is_valid: Array) -> None:
            is_selected = is_valid & ((keys > threshold) | ((keys == threshold) & (positions <= cut)))
            compact(jnp.where(is_selected, positions, -1), bits, index_buffer_ref, value_buffer_ref, state_ref)

        for_each_tile(emit)
        block_barrier()
        if sorts_output:
            own_indices = plgpu.load(index_buffer_ref.at[share_slice], layout=share_layout)
            rank_scatter(own_indices, index_buffer_ref, sorted_ref)
            selected = plgpu.load(sorted_ref.at[share_slice], layout=share_layout)
            out_indices_ref.at[program, share_slice][...] = jnp.where(selected == EMPTY_SLOT, -1, selected)
        else:
            out_indices_ref.at[program, share_slice][...] = plgpu.load(
                index_buffer_ref.at[share_slice],
                layout=share_layout,
            )
            out_values_ref.at[program, share_slice][...] = lax.bitcast_convert_type(
                plgpu.load(value_buffer_ref.at[share_slice], layout=share_layout),
                jnp.float32,
            )

    return plgpu.kernel(
        kernel,
        out_type=(
            jax.ShapeDtypeStruct((num_programs, padded_k), jnp.int32),
            jax.ShapeDtypeStruct((num_programs, padded_k), jnp.float32),
        ),
        grid=(num_programs,),
        grid_names=("programs",),
        num_threads=NUM_WARPGROUPS,
        thread_name="warpgroup",
        scratch_types=(
            plgpu.SMEM((NUM_BINS,), jnp.int32),
            plgpu.SMEM((int(StateSlot.WARP_TOTALS) + NUM_THREADS // 32,), jnp.int32),
            plgpu.SMEM((padded_k,), jnp.int32),
            plgpu.SMEM((padded_k,), jnp.int32),
            plgpu.SMEM((padded_k,), jnp.int32),
        ),
        compiler_params=plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Warpgroup),
        kernel_name=f"top_k_{stage.value}",
    )


def tile_size(num_values: int, num_segments: int) -> int:
    return next(
        tile
        for tile in (4096, 2048, 1024, 512)
        if tile == 512 or num_values // (tile * num_segments) >= NUM_WARPGROUPS
    )


def folded_pallas_top_k_blocks(
    scores: Float[Array, "queries blocks"],
    num_visible_blocks: Int[Array, " queries"],
    k: int,
) -> Int[Array, "queries k"]:
    num_queries, num_blocks = scores.shape
    padded_k = math.ceil(k / 512) * 512
    balanced_segments = math.isqrt(max(num_blocks // k, 1))
    parallel_segments = max(288 // num_queries, 1)
    num_segments = 1 << (min(balanced_segments, parallel_segments, num_blocks // 512).bit_length() - 1)
    tile = tile_size(num_blocks, num_segments)
    if num_segments == 1:
        selection = make_selection(num_queries, num_blocks, k, tile, 1, SelectionStage.FULL_ROWS)
        selected, _ = selection(scores, num_visible_blocks)
        return selected[:, :k]
    segment_selection = make_selection(
        num_queries * num_segments,
        num_blocks,
        k,
        tile,
        num_segments,
        SelectionStage.SEGMENT_CANDIDATES,
    )
    candidate_indices, candidate_values = segment_selection(scores, num_visible_blocks)
    num_candidates = num_segments * padded_k
    merge_selection = make_selection(
        num_queries,
        num_candidates,
        k,
        tile_size(num_candidates, 1),
        1,
        SelectionStage.MERGE_CANDIDATES,
    )
    merged, _ = merge_selection(
        candidate_values.reshape(num_queries, num_candidates),
        candidate_indices.reshape(num_queries, num_candidates),
    )
    return merged[:, :k]


@cache
def make_pallas_top_k_blocks(
    k: int,
) -> Callable[[Float[Array, "queries blocks"], Int[Array, " queries"]], Int[Array, "queries k"]]:
    def xla_top_k(
        scores: Float[Array, "queries blocks"],
        num_visible_blocks: Int[Array, " queries"],
    ) -> Int[Array, "queries k"]:
        return xla_top_k_blocks(scores, num_visible_blocks, k)

    top_k_blocks = jax.custom_batching.custom_vmap(xla_top_k)

    @top_k_blocks.def_vmap
    def top_k_blocks_vmap(
        axis_size: int,
        in_batched: list[bool | None],
        scores: Float[Array, "batch queries blocks"],
        num_visible_blocks: Int[Array, "batch queries"] | Int[Array, " queries"],
    ) -> tuple[Int[Array, "batch queries k"], bool]:
        scores_batched, visible_batched = in_batched
        if not scores_batched:
            return jax.vmap(xla_top_k, in_axes=(None, 0))(scores, num_visible_blocks), True
        _, num_queries, num_blocks = scores.shape
        if num_blocks < 512:
            return jax.vmap(xla_top_k, in_axes=(0, 0 if visible_batched else None))(scores, num_visible_blocks), True
        if not (
            scores.dtype == jnp.float32 and num_blocks < 2**22 and supports_mosaic_gpu(sharding_of(scores).mesh, 10)
        ):
            warnings.warn(
                "Pallas top-k does not support this configuration; falling back to XLA top-k.",
                RuntimeWarning,
                stacklevel=2,
            )
            return jax.vmap(xla_top_k, in_axes=(0, 0 if visible_batched else None))(scores, num_visible_blocks), True
        if not visible_batched:
            num_visible_blocks = jnp.broadcast_to(num_visible_blocks, (axis_size, num_queries))
        aligned_blocks = math.ceil(num_blocks / 4) * 4
        folded_scores = jnp.pad(
            scores.reshape(axis_size * num_queries, num_blocks),
            ((0, 0), (0, aligned_blocks - num_blocks)),
        )
        selected = folded_pallas_top_k_blocks(
            folded_scores,
            num_visible_blocks.reshape(axis_size * num_queries).astype(jnp.int32),
            k,
        )
        return selected.reshape(axis_size, num_queries, k), True

    return top_k_blocks


def pallas_top_k_blocks(
    scores: Float[Array, "queries blocks"],
    num_visible_blocks: Int[Array, " queries"],
    k: int,
) -> Int[Array, "queries k"]:
    return make_pallas_top_k_blocks(k)(scores, num_visible_blocks)
