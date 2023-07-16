from utilities.concurrency import (
    ConcurrentResult,
    CustomSyncManager,
    new_concurrent_result,
)

DT = "dt"
ET = "et"
RF = "rf"


def test_new_concurrent_result():
    """
    Tests the 'new_concurrent_result' function.

    This test case creates a new 'ConcurrentResult' object using 'new_concurrent_result'
    and checks if it is an instance of the 'ConcurrentResult' class.
    The test also verifies that the serialized result matches the expected output
    before and after appending additional data to the result.

    The test uses three different classifiers - DecisionTree (DT), ExtraTrees (ET), and RandomForest (RF) -
    and checks that results for each can be added and serialized correctly.
    """
    expected = {
        DT: [],
        ET: [],
        RF: [],
    }
    # Yes it is ugly but mocking this is even uglier
    # not to mention harder
    with CustomSyncManager() as manager:
        result = new_concurrent_result(
            expected,
            manager,
        )
        assert isinstance(result, ConcurrentResult)
        assert result.serialize() == expected

        expected[DT].append({"test": "testing"})
        expected[ET].append({"test": "testing"})
        expected[RF].append({"test": "testing"})
        result[DT].append({"test": "testing"})
        result[ET].append({"test": "testing"})
        result[RF].append({"test": "testing"})
        assert result.serialize() == expected
