from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from lalamo.model_import.common import token_ids_to_text


def test_token_ids_resolve_to_correct_text() -> None:
    tokenizer = Tokenizer(
        WordLevel(
            vocab={
                "BOS": 0,
                "EOS": 1,
                "hello": 2,
                "[UNK]": 3,
            },
            unk_token="[UNK]",
        ),
    )

    assert token_ids_to_text(tokenizer, 1) == "EOS"
    assert token_ids_to_text(tokenizer, [1]) == "EOS"
    assert token_ids_to_text(tokenizer, [1, 0]) == "EOS"
    assert token_ids_to_text(tokenizer, None) is None
    assert token_ids_to_text(tokenizer, [1, "banana"]) is None  # type:ignore
