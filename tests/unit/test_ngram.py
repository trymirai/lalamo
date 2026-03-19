from lalamo.speculator.ngram import NGramSpeculator


def test_ngram_train_compress_probs() -> None:
    speculator = NGramSpeculator.new(1024, 16, max_order=2)

    speculator.train(range(100), ({k: 1.0} for k in range(100)))
    speculator.compress()

    for seq in [[0], [1], [0, 1]]:
        probs = speculator.probs(seq)
        assert any(v != 0.0 for v in probs.values()), probs


def test_ngram_backoff() -> None:
    speculator = NGramSpeculator.new(1024, 8, max_order=3)

    # Train with a simple repeating pattern
    token_ids = [0, 1, 2] * 50
    token_logits = [{tid: 1.0} for tid in token_ids]
    speculator.train(token_ids, token_logits)
    speculator.compress()

    # Should get non-empty predictions via backoff
    probs = speculator.probs([0, 1])
    assert probs, "Expected non-empty probs from backoff"


def test_ngram_serialize_roundtrip() -> None:
    speculator = NGramSpeculator.new(512, 8, max_order=3, discount=0.01)

    token_ids = list(range(200))
    token_logits = [{k: 1.0} for k in token_ids]
    speculator.train(token_ids, token_logits)
    speculator.compress()

    blob = speculator.serialize()
    restored = NGramSpeculator.deserialize(blob)

    # Verify same predictions
    for ctx in [[0], [1, 2], [10, 20, 30]]:
        assert speculator.probs(ctx) == restored.probs(ctx)


def test_ngram_empty_context() -> None:
    speculator = NGramSpeculator.new(256, 4, max_order=2)

    speculator.train([42, 43, 44], [{42: 1.0}, {43: 1.0}, {44: 1.0}])
    speculator.compress()

    # Empty context should hit unigram table
    probs = speculator.probs([])
    assert isinstance(probs, dict)


def test_ngram_token_id_zero_not_corrupted() -> None:
    """Regression: token id 0 must not be overwritten by zero-filled empty slots."""
    speculator = NGramSpeculator.new(256, 4, max_order=2)

    # Train with token 0 as a real token
    speculator.train([1, 0], [{1: 1.0}, {0: 1.0}])
    speculator.compress()

    probs = speculator.probs([1])
    # Token 0 should have nonzero probability
    assert 0 in probs, f"Token 0 missing from probs: {probs}"
    assert probs[0] > 0.0, f"Token 0 has zero probability: {probs}"

    # Distribution should sum close to 1.0
    total = sum(probs.values())
    assert abs(total - 1.0) < 0.01, f"Probs sum to {total}, expected ~1.0"


def test_ngram_probs_sum_to_one() -> None:
    speculator = NGramSpeculator.new(512, 8, max_order=3)

    token_ids = [0, 1, 2, 3, 4] * 100
    token_logits = [{tid: 1.0} for tid in token_ids]
    speculator.train(token_ids, token_logits)
    speculator.compress()

    for ctx in [[0], [1, 2], [3, 4, 0]]:
        probs = speculator.probs(ctx)
        if probs:
            total = sum(probs.values())
            assert abs(total - 1.0) < 0.05, f"Probs for {ctx} sum to {total}"
