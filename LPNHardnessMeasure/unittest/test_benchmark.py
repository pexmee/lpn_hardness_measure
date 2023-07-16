import json
import math
from tempfile import NamedTemporaryFile
from unittest.mock import MagicMock

from benchmarking.benchmark import LPNBenchmark
from pytest import MonkeyPatch, mark


@mark.parametrize(
    "n, p, expected",
    [
        (100, 0.1, math.sqrt(-3 * math.log(1 - 0.99) / (100 * 0.1))),
        (500, 0.05, math.sqrt(-3 * math.log(1 - 0.99) / (500 * 0.05))),
        (1000, 0.01, math.sqrt(-3 * math.log(1 - 0.99) / (1000 * 0.01))),
    ],
)
def test_calculate_eps(n, p, expected):
    """
    Tests the 'calculate_eps' method of the LPNBenchmark class.

    This test case checks if the method correctly calculates the value of epsilon.

    Args:
        n (int): The number of samples.
        p (float): The error rate.
        expected (float): The expected epsilon value.
    """
    assert math.isclose(LPNBenchmark.calculate_eps(n, p), expected, rel_tol=1e-9)


def test_write_ndjson(monkeypatch: MonkeyPatch):
    """
    Tests the '_write_ndjson' method of the LPNBenchmark class.

    This test case mocks the 'ConcurrentResult' object and checks if the method correctly writes
    the serialized concurrent results to a newline-delimited JSON file.

    Args:
        monkeypatch (MonkeyPatch): pytest's monkeypatch fixture.
    """
    with NamedTemporaryFile() as file_h:
        monkeypatch.setattr("benchmarking.benchmark.DATA_FILE", file_h.name)

        mock_result = MagicMock()
        expected = {"test": "test"}
        mock_result.serialize.return_value = expected
        # monkeypatch.setattr("benchmarking.benchmark.ConcurrentResult", mock_result)

        LPNBenchmark._write_ndjson(mock_result)
        file_h.flush()
        assert json.loads(file_h.read()) == expected
