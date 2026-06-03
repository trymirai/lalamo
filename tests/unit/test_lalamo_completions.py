from pathlib import Path

import msgpack

from lalamo.data.lalamo_completions import LalamoCompletion, load_completions, save_completions


def test_lalamo_completion_roundtrip_msgpack_token_only(tmp_path: Path) -> None:
    path = tmp_path / "traces.msgpack"
    completions = [
        LalamoCompletion(prefix_token_ids=[1, 2], completion_token_ids=[3, 4]),
        LalamoCompletion(prefix_token_ids=[5], completion_token_ids=[6]),
    ]

    save_completions(path, completions)

    assert load_completions(path) == completions
    with path.open("rb") as fd:
        records = list(msgpack.Unpacker(file_like=fd, strict_map_key=False))
    assert records == [
        {"prefix_token_ids": [1, 2], "completion_token_ids": [3, 4]},
        {"prefix_token_ids": [5], "completion_token_ids": [6]},
    ]
