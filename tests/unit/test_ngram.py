from lalamo.speculator.drafter import Drafter
from lalamo.speculator.drafters.ngram import NGramDrafter, NGramModel


def test_ngram_token_id_zero_not_corrupted() -> None:
    """Regression: token id 0 must not be overwritten by zero-filled empty slots."""
    speculator = NGramModel.init(256, 4, max_order=2)

    speculator.train([1, 0], [{1: 1.0}, {0: 1.0}])
    speculator.compress()

    probs = speculator.probs([1])
    assert 0 in probs, f"Token 0 missing from probs: {probs}"
    assert probs[0] > 0.0, f"Token 0 has zero probability: {probs}"

    total = sum(probs.values())
    assert abs(total - 1.0) < 0.01, f"Probs sum to {total}, expected ~1.0"


def test_ngram_serialize_roundtrip() -> None:
    speculator = NGramModel.init(512, 8, max_order=3, discount=0.01)

    token_ids = list(range(200))
    token_logits = [{k: 1.0} for k in token_ids]
    speculator.train(token_ids, token_logits)
    speculator.compress()

    drafter = NGramDrafter(model=speculator)
    blob = drafter.serialize()
    restored = Drafter.deserialize("ngram", blob)

    assert blob == restored.serialize()
