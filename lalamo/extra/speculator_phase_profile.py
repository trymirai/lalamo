from __future__ import annotations

import csv
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import jax
import jax.numpy as jnp
from jax import Array
from rich import box
from rich.console import Console
from rich.table import Table
from typer import Argument, Option, Typer

from lalamo.data.utils import pad_sequences
from lalamo.models.chat_codec import UserMessage
from lalamo.models.language_model import GenerationConfig, LanguageModel
from lalamo.module import Keychain, KeychainBroadcastMode
from lalamo.modules.decoder import DecoderForwardPassConfig, DecoderResult
from lalamo.sampling import SamplingPolicy
from lalamo.speculator.common import NoSpeculator, Speculator, load_speculator
from lalamo.speculator.proposal import AcceptedProposal, ProposalInputs, TrieProposal
from lalamo.speculator.state import LMState


class PhaseName(StrEnum):
    PREFILL = "prefill"
    INIT_STATE = "init_state"
    DRAFT = "draft"
    FORWARD = "forward"
    TARGET_SAMPLE = "target_sample"
    VERIFY = "verify"
    COMMIT = "commit"


@dataclass(frozen=True)
class PhaseRecord:
    step_index: int
    phase: PhaseName
    seconds: float


@dataclass(frozen=True)
class StepRecord:
    step_index: int
    proposal_nodes: int
    accepted_tokens: tuple[int, ...]

    @property
    def min_accepted_tokens(self) -> int:
        return min(self.accepted_tokens)

    @property
    def max_accepted_tokens(self) -> int:
        return max(self.accepted_tokens)

    @property
    def mean_accepted_tokens(self) -> float:
        return sum(self.accepted_tokens) / len(self.accepted_tokens)


@dataclass(frozen=True)
class PhaseTotal:
    phase: PhaseName
    seconds: float
    calls: int

    @property
    def seconds_per_call(self) -> float:
        return self.seconds / max(self.calls, 1)


@dataclass(frozen=True)
class ProfileConfig:
    reasoning: bool
    temperature: float
    top_p: float | None
    top_k: int | None
    min_p: float | None


@dataclass(frozen=True)
class ProfileResult:
    phase_records: tuple[PhaseRecord, ...]
    step_records: tuple[StepRecord, ...]
    output_token_ids: Array
    output_token_mask: Array
    batch_size: int
    max_output_length: int
    config: ProfileConfig

    @property
    def decode_seconds(self) -> float:
        setup_phases = (PhaseName.PREFILL, PhaseName.INIT_STATE)
        return sum(record.seconds for record in self.phase_records if record.phase not in setup_phases)

    @property
    def total_seconds(self) -> float:
        return sum(record.seconds for record in self.phase_records)

    @property
    def generated_tokens(self) -> int:
        return sum(sum(record.accepted_tokens) for record in self.step_records)

    @property
    def decode_steps(self) -> int:
        return len(self.step_records)

    @property
    def tokens_per_step(self) -> float:
        return self.generated_tokens / max(self.decode_steps * self.batch_size, 1)

    @property
    def mean_accepted_length(self) -> float:
        return max(self.tokens_per_step - 1.0, 0.0)

    @property
    def tokens_per_second(self) -> float:
        return self.generated_tokens / max(self.decode_seconds, 1e-9)

    def phase_totals(self) -> tuple[PhaseTotal, ...]:
        totals = []
        for phase in PhaseName:
            records = tuple(record for record in self.phase_records if record.phase == phase)
            if records:
                totals.append(PhaseTotal(phase, sum(record.seconds for record in records), len(records)))
        return tuple(totals)


def block_until_ready(value: object) -> None:
    for leaf in jax.tree.leaves(value):
        if isinstance(leaf, Array):
            leaf.block_until_ready()


def measure_phase[ResultT](
    phase_records: list[PhaseRecord],
    step_index: int,
    phase: PhaseName,
    thunk: Callable[[], ResultT],
) -> ResultT:
    started_at = time.perf_counter()
    result = thunk()
    block_until_ready(result)
    phase_records.append(PhaseRecord(step_index, phase, time.perf_counter() - started_at))
    return result


def prompt_token_batch(
    model: LanguageModel,
    prompt: str,
    batch_size: int,
    reasoning: bool,
) -> tuple[Array, Array]:
    tokenized_prompt = model.token_codec.encode_request([UserMessage(prompt)], enable_thinking=reasoning)
    tokenized_batch = [tokenized_prompt for _ in range(batch_size)]
    prompt_lengths = jnp.asarray([len(tokens) for tokens in tokenized_batch], dtype=jnp.int32)
    prompt_token_ids, _ = pad_sequences(tokenized_batch, pad_token_id=0, padded_length=len(tokenized_prompt))
    return prompt_token_ids, prompt_lengths


def profile_generation(
    model: LanguageModel,
    prompt_token_ids: Array,
    prompt_lengths_without_padding: Array,
    max_output_length: int,
    speculator: Speculator | None,
    generation_config: GenerationConfig | None,
    profile_config: ProfileConfig,
    seed: int,
    prefill_forward_pass_config: DecoderForwardPassConfig | None = None,
    decode_forward_pass_config: DecoderForwardPassConfig | None = None,
) -> ProfileResult:
    if max_output_length < 1:
        raise ValueError("max_output_length must be at least 1.")

    active_speculator: Speculator
    if speculator is None:
        active_speculator = NoSpeculator()
    else:
        active_speculator = speculator

    batch_size, prompt_length = prompt_token_ids.shape
    eos_token_ids = model.resolve_eos_token_ids(generation_config, None)
    sampling_policy = model.default_sampling_policy()
    if generation_config is not None:
        sampling_policy = generation_config.default_policy()
    sampling_policy = sampling_policy.broadcast(batch_size)
    feature_request = active_speculator.feature_request
    keychain = Keychain.init(seed, shape=(batch_size,))
    prefill_keychain, sampling_keychain, decoding_keychain = keychain.split(3)
    phase_records: list[PhaseRecord] = []
    step_records: list[StepRecord] = []
    output_token_ids = jnp.zeros((batch_size, max_output_length), dtype=jnp.int32)
    output_token_mask = jnp.zeros((batch_size, max_output_length), dtype=jnp.bool)

    prefill_results = measure_phase(
        phase_records,
        -1,
        PhaseName.PREFILL,
        lambda: model.prefill_tokens(
            prompt_token_ids,
            prompt_length + max_output_length + 1,
            prompt_lengths_without_padding,
            prefill_forward_pass_config,
            feature_request=feature_request,
            keychain=prefill_keychain,
        ),
    )
    sampling_keys = sampling_keychain.rolling_broadcast(
        (batch_size,),
        mode=KeychainBroadcastMode.SUFFIX,
    ).vmapped_keys
    state = measure_phase(
        phase_records,
        -1,
        PhaseName.INIT_STATE,
        lambda: active_speculator.init_state(
            prefill_results,
            prompt_lengths_without_padding,
            sampling_policy,
            sampling_keys,
        ),
    )
    decoding_keys = decoding_keychain.rolling_broadcast(
        (max_output_length, *decoding_keychain.vmapped_keys.shape),
        mode=KeychainBroadcastMode.PREFIX,
    ).vmapped_keys

    def output_lengths(lm_state: LMState) -> Array:
        return lm_state.output_lengths

    def is_done(lm_state: LMState) -> Array:
        return lm_state.stop_flags | (output_lengths(lm_state) >= max_output_length)

    def stop_flags_after_step(
        lm_state: LMState,
        token_ids: Array,
        write_mask: Array,
    ) -> Array:
        if eos_token_ids.shape[0] == 0:
            return lm_state.stop_flags
        eos_hits = jnp.any(token_ids[:, :, None] == eos_token_ids[None, None, :], axis=-1)
        return lm_state.stop_flags | jnp.any(write_mask & eos_hits, axis=1)

    for step_index, decoding_key in enumerate(decoding_keys):
        if bool(jax.device_get(jnp.all(is_done(state)))):
            break

        current_output_lengths = output_lengths(state)
        done = is_done(state)

        def draft_phase(current_state: LMState = state) -> TrieProposal:
            return active_speculator.draft(current_state)

        proposal = measure_phase(
            phase_records,
            step_index,
            PhaseName.DRAFT,
            draft_phase,
        )
        proposal_inputs = proposal.forward_inputs(state.next_token_position)
        forward_pass_config = decode_forward_pass_config
        if forward_pass_config is None:
            forward_pass_config = DecoderForwardPassConfig.for_inference(proposal_inputs.forward_pass_mode)

        def forward_phase(
            current_state: LMState = state,
            current_proposal_inputs: ProposalInputs = proposal_inputs,
            current_forward_pass_config: DecoderForwardPassConfig = forward_pass_config,
            current_decoding_key: Array = decoding_key,
        ) -> DecoderResult:
            return model.decoder(
                token_ids=current_proposal_inputs.token_ids,
                token_positions=current_proposal_inputs.token_positions,
                state=current_state.kv_cache,
                return_updated_state=True,
                return_activation_trace=feature_request.output_features or bool(feature_request.layer_indices),
                lengths_without_padding=current_proposal_inputs.lengths_without_padding,
                forward_pass_config=current_forward_pass_config,
                attention_parent_indices=current_proposal_inputs.attention_parent_indices,
                keychain=Keychain(vmapped_keys=current_decoding_key, batch_key=decoding_keychain.batch_key),
            )

        decoder_result = measure_phase(
            phase_records,
            step_index,
            PhaseName.FORWARD,
            forward_phase,
        )

        def target_sample_phase(
            current_proposal: TrieProposal = proposal,
            current_decoder_result: DecoderResult = decoder_result,
        ) -> tuple[Array, Array, SamplingPolicy]:
            return current_proposal.sample(current_decoder_result.logits)

        processed_tree_logits, sampled_token_ids, next_sampling_policies = measure_phase(
            phase_records,
            step_index,
            PhaseName.TARGET_SAMPLE,
            target_sample_phase,
        )

        def verify_phase(
            current_proposal: TrieProposal = proposal,
            current_sampled_token_ids: Array = sampled_token_ids,
            current_next_sampling_policies: SamplingPolicy = next_sampling_policies,
            current_output_lengths_snapshot: Array = current_output_lengths,
            current_done: Array = done,
        ) -> tuple[AcceptedProposal, Array]:
            verified = current_proposal.verify(current_sampled_token_ids, current_next_sampling_policies)
            truncated, write_mask = verified.truncate(
                current_output_lengths_snapshot,
                max_output_length,
                current_done,
                eos_token_ids,
            )
            return truncated, write_mask

        accepted, write_mask = measure_phase(
            phase_records,
            step_index,
            PhaseName.VERIFY,
            verify_phase,
        )
        next_stop_flags = stop_flags_after_step(state, accepted.accepted_token_ids, write_mask)

        def commit_phase(
            current_state: LMState = state,
            current_decoder_result: DecoderResult = decoder_result,
            current_processed_tree_logits: Array = processed_tree_logits,
            current_accepted: AcceptedProposal = accepted,
            current_stop_flags: Array = next_stop_flags,
        ) -> LMState:
            return current_state.commit(
                current_decoder_result,
                current_processed_tree_logits,
                current_accepted,
                current_stop_flags,
                feature_request,
            )

        slots = jnp.arange(accepted.accepted_token_ids.shape[1], dtype=jnp.int32)[None, :]
        positions = current_output_lengths[:, None] + slots
        valid = (slots < accepted.num_compact_indices[:, None]) & (positions < max_output_length)
        batch_indices = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
        output_token_ids = output_token_ids.at[batch_indices, positions].add(
            jnp.where(valid, accepted.accepted_token_ids, 0),
            mode="drop",
        )
        output_token_mask = output_token_mask.at[batch_indices, positions].set(valid, mode="drop")
        state = measure_phase(
            phase_records,
            step_index,
            PhaseName.COMMIT,
            commit_phase,
        )
        accepted_tokens = tuple(int(value) for value in jax.device_get(accepted.num_compact_indices).tolist())
        step_records.append(
            StepRecord(
                step_index=step_index,
                proposal_nodes=proposal.num_nodes,
                accepted_tokens=accepted_tokens,
            ),
        )

    output_token_ids = jnp.where(output_token_mask, output_token_ids, 0)
    block_until_ready((output_token_ids, output_token_mask))
    return ProfileResult(
        phase_records=tuple(phase_records),
        step_records=tuple(step_records),
        output_token_ids=output_token_ids,
        output_token_mask=output_token_mask,
        batch_size=batch_size,
        max_output_length=max_output_length,
        config=profile_config,
    )


def write_csv_profile(profile: ProfileResult, path: Path | str) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "step",
                "phase",
                "seconds",
                "proposal_nodes",
                "accepted_min",
                "accepted_mean",
                "accepted_max",
            ],
        )
        for phase_record in profile.phase_records:
            matching_steps = tuple(
                step_record
                for step_record in profile.step_records
                if step_record.step_index == phase_record.step_index
            )
            if matching_steps:
                step_record = matching_steps[0]
                writer.writerow(
                    [
                        phase_record.step_index,
                        phase_record.phase.value,
                        f"{phase_record.seconds:.9f}",
                        step_record.proposal_nodes,
                        step_record.min_accepted_tokens,
                        f"{step_record.mean_accepted_tokens:.6f}",
                        step_record.max_accepted_tokens,
                    ],
                )
            else:
                writer.writerow(
                    [
                        phase_record.step_index,
                        phase_record.phase.value,
                        f"{phase_record.seconds:.9f}",
                        "",
                        "",
                        "",
                        "",
                    ],
                )


def print_profile(profile: ProfileResult, console: Console) -> None:
    summary = Table(
        title="Speculator phase profile",
        show_header=True,
        header_style="bold",
        box=box.ROUNDED,
    )
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("batch_size", str(profile.batch_size))
    summary.add_row("max_output_length", str(profile.max_output_length))
    summary.add_row("reasoning", str(profile.config.reasoning).lower())
    summary.add_row("temperature", f"{profile.config.temperature:g}")
    summary.add_row("top_p", "none" if profile.config.top_p is None else f"{profile.config.top_p:g}")
    summary.add_row("top_k", "none" if profile.config.top_k is None else str(profile.config.top_k))
    summary.add_row("min_p", "none" if profile.config.min_p is None else f"{profile.config.min_p:g}")
    summary.add_row("decode_steps", str(profile.decode_steps))
    summary.add_row("generated_tokens", str(profile.generated_tokens))
    summary.add_row("tok/step", f"{profile.tokens_per_step:.3f}")
    summary.add_row("mal", f"{profile.mean_accepted_length:.3f}")
    summary.add_row("tok/sec(decode)", f"{profile.tokens_per_second:.3f}")
    summary.add_row("decode_seconds", f"{profile.decode_seconds:.6f}")
    summary.add_row("total_seconds", f"{profile.total_seconds:.6f}")
    console.print(summary)

    phase_table = Table(
        title="Phase totals",
        show_header=True,
        header_style="bold",
        box=box.ROUNDED,
    )
    phase_table.add_column("Phase")
    phase_table.add_column("Seconds", justify="right")
    phase_table.add_column("Calls", justify="right")
    phase_table.add_column("Sec/call", justify="right")
    phase_table.add_column("Decode share", justify="right")
    for total in profile.phase_totals():
        if total.phase in (PhaseName.PREFILL, PhaseName.INIT_STATE):
            share = "setup"
        else:
            share = f"{total.seconds / max(profile.decode_seconds, 1e-9):.2%}"
        phase_table.add_row(
            total.phase.value,
            f"{total.seconds:.6f}",
            str(total.calls),
            f"{total.seconds_per_call:.6f}",
            share,
        )
    console.print(phase_table)

    if not profile.step_records:
        return
    step_table = Table(
        title="Step acceptance",
        show_header=True,
        header_style="bold",
        box=box.ROUNDED,
    )
    step_table.add_column("Step", justify="right")
    step_table.add_column("Nodes", justify="right")
    step_table.add_column("Accepted min", justify="right")
    step_table.add_column("Accepted mean", justify="right")
    step_table.add_column("Accepted max", justify="right")
    for step_record in profile.step_records:
        step_table.add_row(
            str(step_record.step_index),
            str(step_record.proposal_nodes),
            str(step_record.min_accepted_tokens),
            f"{step_record.mean_accepted_tokens:.3f}",
            str(step_record.max_accepted_tokens),
        )
    console.print(step_table)


def resolve_prompt(prompt: str, prompt_file: Path | None) -> str:
    if prompt_file is None:
        return prompt
    return prompt_file.read_text()


app = Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def main(
    model_path: Annotated[
        Path,
        Argument(help="Path to a converted lalamo model directory."),
    ],
    speculator_path: Annotated[
        Path | None,
        Option("--speculator", help="Path to a speculator artifact."),
    ] = None,
    prompt: Annotated[
        str,
        Option("--prompt", help="Prompt used for profiling."),
    ] = "Explain speculative decoding in one concise paragraph.",
    prompt_file: Annotated[
        Path | None,
        Option("--prompt-file", help="Read the profiling prompt from a file."),
    ] = None,
    batch_size: Annotated[
        int,
        Option("--batch-size", "--batch_size", help="Batch size. The prompt is repeated across the batch."),
    ] = 1,
    max_output_length: Annotated[
        int,
        Option("--max-output-length", "--max_output_length", help="Maximum generated tokens per row."),
    ] = 4096,
    reasoning: Annotated[
        bool,
        Option("--reasoning/--no-reasoning", help="Render profiling prompt with reasoning/thinking enabled."),
    ] = True,
    temperature: Annotated[
        float,
        Option("--temperature", help="Sampling temperature. Use 0 for greedy decoding."),
    ] = 1.0,
    top_p: Annotated[
        float | None,
        Option("--top_p", "--top-p", help="Nucleus sampling threshold.", show_default="none"),
    ] = None,
    top_k: Annotated[
        int | None,
        Option("--top_k", "--top-k", help="Top-k sampling cutoff.", show_default="none"),
    ] = None,
    min_p: Annotated[
        float | None,
        Option("--min_p", "--min-p", help="Min-p sampling cutoff.", show_default="none"),
    ] = None,
    seed: Annotated[
        int,
        Option("--seed", help="Sampling seed."),
    ] = 0,
    warmup: Annotated[
        bool,
        Option("--warmup/--no-warmup", help="Run one unrecorded phase-profile warmup before measuring."),
    ] = True,
    csv_path: Annotated[
        Path | None,
        Option("--csv", help="Write per-phase timing rows to CSV."),
    ] = None,
) -> None:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    console = Console()
    prompt_text = resolve_prompt(prompt, prompt_file)
    model = LanguageModel.load(model_path)
    speculator = None
    if speculator_path is not None:
        speculator = load_speculator(speculator_path, model.decoder)
    profile_config = ProfileConfig(
        reasoning=reasoning,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
    )
    generation_config = GenerationConfig(
        stop_token_ids=model.config.generation_config.stop_token_ids,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
    )
    prompt_token_ids, prompt_lengths = prompt_token_batch(model, prompt_text, batch_size, reasoning)
    if warmup:
        profile_generation(
            model,
            prompt_token_ids,
            prompt_lengths,
            max_output_length,
            speculator,
            generation_config=generation_config,
            profile_config=profile_config,
            seed=seed + 1,
        )
    profile = profile_generation(
        model,
        prompt_token_ids,
        prompt_lengths,
        max_output_length,
        speculator,
        generation_config=generation_config,
        profile_config=profile_config,
        seed=seed,
    )
    print_profile(profile, console)
    if csv_path is not None:
        write_csv_profile(profile, csv_path)
        console.print(f"Wrote CSV: {csv_path}")


if __name__ == "__main__":
    app()
