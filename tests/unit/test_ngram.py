from lalamo.speculator.ngram import NGramSpeculator


def test_ngram_simple() -> None:
    speculator = NGramSpeculator.new(1024, 16, 2)

    for seq in [[], [0], [1], [0, 1]]:
        assert all(v == 0.0 for v in speculator.probs(seq).values())

    speculator.train(range(100), ({k: 1.0} for k in range(100)))

    for seq in [[0], [1], [0, 1]]:
        probs = speculator.probs(seq)
        assert any(v != 0.0 for v in probs.values()), probs
        assert sorted(probs.items(), key=lambda x: x[1], reverse=True)[0] == (next(reversed(seq), -1) + 1, 1.0)
