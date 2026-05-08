from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from queue import Full, Queue
from threading import Event, Thread

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from lalamo.data.completion_features import (
    FeatureRequest,
    LalamoCompletionBatch,
    LalamoCompletionFeatures,
)
from lalamo.models.language_model import ForwardPassConfig, LanguageModel
from lalamo.modules import ForwardPassMode


@dataclass(frozen=True)
class OnlineCompletionFeatureExtractor:
    model: LanguageModel
    device_id: int
    pad_token_id: int = 0
    prompt_padding_multiple: int = 128
    generation_padding_multiple: int = 512
    forward_pass_config: ForwardPassConfig | None = None

    @property
    def device(self) -> jax.Device:
        devices = jax.devices()
        if self.device_id < 0 or self.device_id >= len(devices):
            raise ValueError(f"device_id {self.device_id} is out of range for {len(devices)} JAX devices.")
        return devices[self.device_id]

    def iter_features(
        self,
        requests: Iterable[FeatureRequest],
    ) -> Iterator[LalamoCompletionFeatures]:
        with jax.default_device(self.device):
            for request in requests:
                yield self.extract_features(request)

    def extract_features(self, request: FeatureRequest) -> LalamoCompletionFeatures:
        with jax.default_device(self.device):
            completion_batch = LalamoCompletionBatch.from_completions(
                request.completions,
                pad_token_id=self.pad_token_id,
                prompt_padding_multiple=self.prompt_padding_multiple,
                generation_padding_multiple=self.generation_padding_multiple,
            )
            batch = self.place_batch_on_device(completion_batch)
            token_positions = jnp.broadcast_to(
                jnp.arange(batch.input_token_ids.shape[1], dtype=jnp.int32),
                batch.input_token_ids.shape,
            )
            decoder_result = self.model.model(
                batch.input_token_ids,
                token_positions,
                return_activation_trace=request.activation_trace,
                lengths_without_padding=batch.input_lengths,
                forward_pass_mode=ForwardPassMode.MULTI_TOKEN,
                forward_pass_config=self.forward_pass_config,
            )
            target_logits = gather_target_positions(decoder_result.logits, batch.target_positions)

            target_logsumexp = jax.nn.logsumexp(target_logits, axis=-1) if request.logits else None
            target_top_k_logits = None
            target_top_k_ids = None
            if request.top_k_logits is not None:
                top_k = min(request.top_k_logits, target_logits.shape[-1])
                target_top_k_logits, target_top_k_ids = jax.lax.top_k(target_logits, top_k)

            output_features = None
            layer_features = None
            if request.activation_trace:
                activation_trace = decoder_result.activation_trace
                assert activation_trace is not None
                if request.output_features:
                    output_features = gather_target_positions(activation_trace.output_norm, batch.target_positions)
                if request.layer_indices:
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
                target_logits=target_logits if request.full_logits else None,
                target_logsumexp=target_logsumexp,
                target_top_k_ids=target_top_k_ids,
                target_top_k_logits=target_top_k_logits,
                output_features=output_features,
                layer_features=layer_features,
            )

    def place_batch_on_device(self, completion_batch: LalamoCompletionBatch) -> LalamoCompletionBatch:
        device = self.device
        return LalamoCompletionBatch(
            prefix_token_ids=jax.device_put(completion_batch.prefix_token_ids, device),
            prefix_mask=jax.device_put(completion_batch.prefix_mask, device),
            completion_token_ids=jax.device_put(completion_batch.completion_token_ids, device),
            completion_mask=jax.device_put(completion_batch.completion_mask, device),
            input_token_ids=jax.device_put(completion_batch.input_token_ids, device),
            input_lengths=jax.device_put(completion_batch.input_lengths, device),
            target_token_ids=jax.device_put(completion_batch.target_token_ids, device),
            target_mask=jax.device_put(completion_batch.target_mask, device),
            target_positions=jax.device_put(completion_batch.target_positions, device),
        )

    def resolve_layer_indices(self, request: FeatureRequest) -> tuple[int, ...]:
        num_layers = len(self.model.model.transformer.layers)
        resolved = tuple(
            layer_index if layer_index >= 0 else num_layers + layer_index for layer_index in request.layer_indices
        )
        if any(layer_index < 0 or layer_index >= num_layers for layer_index in resolved):
            raise ValueError("layer_indices index out of range.")
        return tuple(sorted(set(resolved)))


@dataclass(frozen=True)
class FeatureQueueEnd:
    pass


@dataclass(frozen=True)
class FeatureQueueError:
    error: Exception


type FeatureQueueItem = LalamoCompletionFeatures | FeatureQueueEnd | FeatureQueueError


@dataclass(frozen=True)
class InMemoryCompletionFeatureQueue:
    extractor: OnlineCompletionFeatureExtractor
    max_batches: int

    def iter_features(
        self,
        requests: Iterable[FeatureRequest],
    ) -> Iterator[LalamoCompletionFeatures]:
        assert self.max_batches > 0
        queue: Queue[FeatureQueueItem] = Queue(maxsize=self.max_batches)
        stop_event = Event()
        worker = Thread(
            target=self.run_worker,
            args=(requests, queue, stop_event),
            daemon=True,
        )
        worker.start()

        try:
            while True:
                item = queue.get()
                match item:
                    case LalamoCompletionFeatures():
                        yield item
                    case FeatureQueueEnd():
                        break
                    case FeatureQueueError(error):
                        raise error
        finally:
            stop_event.set()

    def run_worker(
        self,
        requests: Iterable[FeatureRequest],
        queue: Queue[FeatureQueueItem],
        stop_event: Event,
    ) -> None:
        try:
            for features in self.extractor.iter_features(requests):
                if stop_event.is_set():
                    return
                put_feature_queue_item(queue, features, stop_event)
            put_feature_queue_item(queue, FeatureQueueEnd(), stop_event)
        except Exception as error:  # noqa: BLE001
            put_feature_queue_item(queue, FeatureQueueError(error), stop_event)


def put_feature_queue_item(
    queue: Queue[FeatureQueueItem],
    item: FeatureQueueItem,
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
