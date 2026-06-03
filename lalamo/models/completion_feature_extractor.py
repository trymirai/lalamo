from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from queue import Full, Queue
from threading import Event, Thread
from typing import Annotated

import jax
import jax.numpy as jnp
from annotated_types import Ge
from jaxtyping import Array, Float, Int

from lalamo.data.completion_features import (
    FeatureRequest,
    LalamoCompletionBatch,
    LalamoCompletionFeatures,
)
from lalamo.models.language_model import LanguageModel
from lalamo.module import Keychain
from lalamo.modules import DecoderForwardPassConfig, ForwardPassMode


@dataclass(frozen=True)
class OnlineCompletionFeatureExtractor:
    model: LanguageModel
    device_id: int
    pad_token_id: int = 0
    prompt_padding_multiple: int = 512
    generation_padding_multiple: int = 1024
    forward_pass_config: DecoderForwardPassConfig | None = None

    @property
    def device(self) -> jax.Device:
        return jax_device(self.device_id)

    def iter_features(
        self,
        requests: Iterable[FeatureRequest],
    ) -> Iterator[LalamoCompletionFeatures]:
        with jax.default_device(self.device):
            for request in requests:
                yield self.extract_features(request)

    def extract_features(self, request: FeatureRequest) -> LalamoCompletionFeatures:
        batch = LalamoCompletionBatch.from_completions(
            request.completions,
            pad_token_id=self.pad_token_id,
            prompt_padding_multiple=self.prompt_padding_multiple,
            generation_padding_multiple=self.generation_padding_multiple,
        )
        token_positions = jnp.broadcast_to(
            jnp.arange(batch.input_token_ids.shape[1], dtype=jnp.int32),
            batch.input_token_ids.shape,
        )
        forward_pass_config = self.forward_pass_config
        if forward_pass_config is None:
            forward_pass_config = DecoderForwardPassConfig.for_inference(ForwardPassMode.MULTI_TOKEN)
        decoder_result = self.model.decoder(
            batch.input_token_ids,
            token_positions,
            return_activation_trace=request.output_features or bool(request.layer_indices),
            lengths_without_padding=batch.input_lengths,
            forward_pass_config=forward_pass_config,
            keychain=Keychain.init(0, sharding_config=self.model.sharding_config),
        )
        target_logits = gather_target_positions(decoder_result.logits, batch.target_positions)

        target_logsumexp = jax.nn.logsumexp(target_logits, axis=-1)
        top_k = min(request.top_k_logits, target_logits.shape[-1])
        target_top_k_logits, target_top_k_ids = jax.lax.top_k(target_logits, top_k)

        activation_trace = decoder_result.activation_trace
        output_features = None
        if request.output_features:
            assert activation_trace is not None
            output_features = gather_target_positions(
                activation_trace.output_norm,
                batch.target_positions,
            )

        layer_features = None
        if request.layer_indices:
            assert activation_trace is not None
            layer_features = jnp.stack(
                [
                    gather_target_positions(
                        activation_trace.layer_results[layer_index].outputs,
                        batch.target_positions,
                    )
                    for layer_index in self.resolve_layer_indices(request)
                ],
                axis=1,
            )

        return LalamoCompletionFeatures(
            completion_batch=batch,
            target_logsumexp=target_logsumexp,
            target_top_k_ids=target_top_k_ids,
            target_top_k_logits=target_top_k_logits,
            output_features=output_features,
            layer_features=layer_features,
        )

    def resolve_layer_indices(self, request: FeatureRequest) -> tuple[int, ...]:
        num_layers = len(self.model.decoder.transformer.layers)
        resolved = tuple(
            layer_index if layer_index >= 0 else num_layers + layer_index for layer_index in request.layer_indices
        )
        if any(layer_index < 0 or layer_index >= num_layers for layer_index in resolved):
            raise ValueError("layer_indices index out of range.")
        return tuple(sorted(set(resolved)))


@dataclass(frozen=True)
class FeatureQueue:
    extractor: OnlineCompletionFeatureExtractor
    max_prefetch: Annotated[int, Ge(1)]
    target_device_id: int

    def iter_features(
        self,
        requests: Iterable[FeatureRequest],
    ) -> Iterator[LalamoCompletionFeatures]:
        queue: Queue[LalamoCompletionFeatures | None] = Queue(maxsize=self.max_prefetch)
        stop_event = Event()
        error_holder: list[BaseException] = []
        worker = Thread(
            target=self.run_worker,
            args=(requests, queue, stop_event, error_holder),
            daemon=True,
        )
        worker.start()

        try:
            while True:
                item = queue.get()
                if item is None:
                    if error_holder:
                        raise error_holder[0]
                    return
                yield item
        finally:
            stop_event.set()

    def run_worker(
        self,
        requests: Iterable[FeatureRequest],
        queue: Queue[LalamoCompletionFeatures | None],
        stop_event: Event,
        error_holder: list[BaseException],
    ) -> None:
        target_device = jax_device(self.target_device_id)
        try:
            for features in self.extractor.iter_features(requests):
                if stop_event.is_set():
                    return
                put_feature_queue_item(
                    queue,
                    jax.device_put(features, target_device),
                    stop_event,
                )
        except BaseException as error:  # noqa: BLE001
            error_holder.append(error)
        finally:
            put_feature_queue_item(queue, None, stop_event)


def jax_device(device_id: int) -> jax.Device:
    devices = jax.devices()
    if device_id < 0 or device_id >= len(devices):
        raise ValueError(f"device_id {device_id} is out of range for {len(devices)} JAX devices.")
    return devices[device_id]


def put_feature_queue_item(
    queue: Queue[LalamoCompletionFeatures | None],
    item: LalamoCompletionFeatures | None,
    stop_event: Event,
) -> None:
    while not stop_event.is_set():
        try:
            queue.put(item, timeout=0.1)
        except Full:
            pass
        else:
            return


def gather_target_positions(
    values: Float[Array, "batch tokens channels"],
    positions: Int[Array, "batch completion_tokens"],
) -> Float[Array, "batch completion_tokens channels"]:
    batch_indices = jnp.arange(values.shape[0])[:, None]
    return values[batch_indices, positions]
