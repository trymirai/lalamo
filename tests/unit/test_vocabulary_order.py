import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from safetensors.numpy import save_file
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace

from lalamo.model_import.origins import FileSpec, LocalOrigin
from lalamo.model_import.vocabulary_order import VocabularyOrderedOrigin, frequency_order


def test_frequency_ordered_origin_renames_every_token_id_and_vocabulary_row(tmp_path: Path) -> None:
    pieces = ["a", "b", "c", "ab", "abc", "id"]
    tokenizer = Tokenizer(BPE({piece: i for i, piece in enumerate(pieces)}, [("a", "b"), ("ab", "c")]))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.add_special_tokens(["<eos>"])
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    vocab = 8  # one padded row past the 7 tokens
    eos = tokenizer.token_to_id("<eos>")
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"vocab_size": vocab, "eos_token_id": [eos]}}))
    embedding = np.arange(vocab * 2, dtype=np.float32).reshape(vocab, 2)
    norm = np.ones(2, np.float32)
    save_file({"embed_tokens.weight": embedding, "norm.weight": norm}, str(tmp_path / "w.safetensors"))
    (tmp_path / "counts.json").write_text(json.dumps([[eos, 9], [5, 7], [4, 7], [1, 3]]))

    order = frequency_order(tmp_path / "counts.json", vocab)
    assert order.tolist() == [6, 4, 5, 1, 0, 2, 3, 7]
    new_ids = np.argsort(order)
    origin = VocabularyOrderedOrigin(LocalOrigin(str(tmp_path), ("w.safetensors",)), order)

    renamed = Tokenizer.from_file(str(origin.resolve_file(FileSpec("tokenizer.json"))))
    text = "abc id ab<eos>"
    assert renamed.encode(text).ids == [int(new_ids[i]) for i in tokenizer.encode(text).ids]
    assert renamed.decode(renamed.encode(text).ids, skip_special_tokens=False) == tokenizer.decode(
        tokenizer.encode(text).ids,
        skip_special_tokens=False,
    )
    renamed_config = json.loads(origin.resolve_file(FileSpec("config.json")).read_text())
    assert renamed_config["text_config"]["eos_token_id"] == [0]

    (shard,) = origin.get_weights()
    with shard as (weights, _):
        assert jnp.array_equal(weights["embed_tokens.weight"], embedding[order])
        assert jnp.array_equal(weights["norm.weight"], norm)
