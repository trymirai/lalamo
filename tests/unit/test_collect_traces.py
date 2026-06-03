from tests.conftest import RunLalamo


def test_collect_traces_help(run_lalamo: RunLalamo) -> None:
    output = run_lalamo("collect-traces", "--help")

    assert "MODEL_PATH" in output
    assert "DATASET_PATH" in output
    assert "--output-path" in output
    assert "--num-tokens-to-generate" not in output
