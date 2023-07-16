from random import randint
from unittest.mock import MagicMock

import numpy as np
import pytest
from _pytest.monkeypatch import MonkeyPatch
from learning.classifiers import generate_samples, predict_with_classifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


@pytest.mark.parametrize(
    "secret_len,amount,error_rate",
    [
        (np.random.randint(2, 10), 50, 0.2),
        (np.random.randint(5, 10), 100, 0.3),
        (np.random.randint(10, 20), 200, 0.25),
        (np.random.randint(20, 30), 1_000, 0.2),
        (np.random.randint(1, 10), 5_000, 0.5),
    ],
)
def test_generate_samples(secret_len, amount, error_rate):
    """
    Tests the 'generate_samples' function.

    This test case generates a random secret and then generates sample data using the
    'generate_samples' function. The test validates that the generated samples have the expected shapes,
    contain only binary values, and have a mean approximately equal to the error rate.

    Args:
        secret_len (int): The length of the secret to be generated.
        amount (int): The number of samples to be generated.
        error_rate (float): The error rate for the samples.
    """
    s = np.random.randint(0, 2, secret_len)
    A, b = generate_samples(s, amount, error_rate)

    # It should be amount X secret_len
    assert A.shape == (amount, secret_len)
    # Should be vector of size amount
    assert b.shape == (amount,)
    # All elements in A and b should be binary
    assert np.all(np.logical_or(A == 0, A == 1))
    assert np.all(np.logical_or(b == 0, b == 1))


@pytest.mark.parametrize(
    "secret_len, amount, error_rate, classifier",
    [
        (np.random.randint(2, 10), 50, 0.2, ExtraTreesClassifier),
        (np.random.randint(5, 10), 100, 0.3, DecisionTreeClassifier),
        (np.random.randint(10, 20), 200, 0.25, RandomForestClassifier),
        (np.random.randint(20, 30), 1_000, 0.2, ExtraTreesClassifier),
        (np.random.randint(1, 10), 5_000, 0.5, RandomForestClassifier),
    ],
)
def test_predict_with_classifier(
    secret_len, amount, error_rate, classifier, monkeypatch: MonkeyPatch
):
    """
    Tests the 'predict_with_classifier' function.

    This test case generates a random secret, generates sample data using the 'generate_samples' function,
    and then uses the 'predict_with_classifier' function to predict the secret.
    The test validates that the predicted secret has the correct length, the returned duration is 0,
    the hamming weight is positive and less than or equal to the amount of data,
    and the predicted secret contains only binary values.

    Args:
        secret_len (int): The length of the secret to be generated.
        amount (int): The number of samples to be generated.
        error_rate (float): The error rate for the samples.
        classifier (ClassifierType): The type of classifier to be used for prediction.
        monkeypatch (MonkeyPatch): A monkeypatch object for mocking.
    """
    retval = randint(5, 20)  # for the lulz.
    monkeypatch.setattr(
        "learning.classifiers.default_timer",
        MagicMock(return_value=retval),
    )
    s = np.random.randint(0, 2, secret_len)
    A, b = generate_samples(s, amount, error_rate)
    cs, hw, duration = predict_with_classifier(classifier, A, b, len(s))

    # The predicted secret has the correct length
    assert cs.shape == (secret_len,)

    # The returned duration is 0
    assert duration == 0

    # The hamming weight is positive and less than or equal to the amount of data
    assert hw <= amount

    # All elements in the predicted secret are binary
    assert np.all(np.logical_or(cs == 0, cs == 1))
