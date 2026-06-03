import numpy as np

from lalamo.data.completion_features import LalamoCompletionBatch
from lalamo.data.lalamo_completions import LalamoCompletion


def test_completion_batch_aligns_inputs_and_targets() -> None:
    batch = LalamoCompletionBatch.from_completions(
        [
            LalamoCompletion(prefix_token_ids=[10, 11], completion_token_ids=[12, 13]),
            LalamoCompletion(prefix_token_ids=[20], completion_token_ids=[21]),
        ],
        prompt_padding_multiple=1,
        generation_padding_multiple=1,
    )

    np.testing.assert_array_equal(batch.prefix_token_ids, np.asarray([[10, 11], [20, 0]], dtype=np.int32))
    np.testing.assert_array_equal(batch.prefix_mask, np.asarray([[True, True], [True, False]], dtype=np.bool_))
    np.testing.assert_array_equal(batch.completion_token_ids, np.asarray([[12, 13], [21, 0]], dtype=np.int32))
    np.testing.assert_array_equal(batch.completion_mask, np.asarray([[True, True], [True, False]], dtype=np.bool_))
    np.testing.assert_array_equal(batch.input_token_ids, np.asarray([[10, 11, 12], [20, 0, 0]], dtype=np.int32))
    np.testing.assert_array_equal(batch.input_lengths, np.asarray([3, 1], dtype=np.int32))
    np.testing.assert_array_equal(batch.target_token_ids, batch.completion_token_ids)
    np.testing.assert_array_equal(batch.target_mask, batch.completion_mask)
    np.testing.assert_array_equal(batch.target_positions, np.asarray([[1, 2], [0, 0]], dtype=np.int32))
