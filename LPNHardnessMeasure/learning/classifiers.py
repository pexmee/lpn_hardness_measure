from timeit import default_timer
from typing import Tuple, Type, Union

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import BaseEnsemble, ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import BaseDecisionTree, DecisionTreeClassifier

ClassifierType = Union[
    Type[DecisionTreeClassifier],
    Type[RandomForestClassifier],
    Type[ExtraTreesClassifier],
    Type[BaseDecisionTree],
    Type[BaseEnsemble],
]


def generate_samples(
    s: NDArray, amount: int, error_rate: float
) -> Tuple[NDArray, NDArray]:
    """
    Generates the sample matrix A and the response b,
    i.e b = (A*s + e) mod 2

    Args:
        s (NDArray): A secret. A vector of binary numbers.
        amount (int): Amount of samples to generate
        error_rate (float): The error rate. Example: 0.1 = 10% error rate.

    Returns:
        Tuple[NDArray, NDArray]: A matrix of samples of size
        (amount X dim) and the response b = (A*s + e) mod 2.
        i.e returns A and b.
    """
    dim = len(s)

    # Randomly generate matrix A where each num ∈ {0,1}
    A = np.random.randint(0, 2, size=(amount, dim))

    # Add errors using binominal distribution
    e = np.random.binomial(1, error_rate, amount)

    # Multiply matrix A with secret key s and add error e
    As_with_err = (A @ s) + e

    # b the response, i.e (A*s + e) mod 2
    b = np.mod(As_with_err, 2)
    return A, b


def predict_with_classifier(
    classifier: ClassifierType,
    A: NDArray,
    b: NDArray,
    dim: int,
) -> Tuple[NDArray, int]:
    """
    Predicts a candidate secret using a provided classifier
    and calculates the hamming weight of A * candidate_secret + b,
    and measures the duration.

    Args:
        classifier (Union[ Type[DecisionTreeClassifier], Type[RandomForestClassifier], Type[ExtraTreesClassifier], ]): a classifier
        A (NDArray): Sample matrix
        b (NDArray): The response, i.e (A*s + e) mod 2
        dim (int): The length of the secret

    Returns:
        Tuple[NDArray, int, float]: Candidate secret, hamming weight and the duration
    """

    start = default_timer()
    c = classifier()
    c.fit(A, b)
    candidate_secret = c.predict(np.eye(dim))
    hamming_weight = np.mod(A @ candidate_secret + b, 2).sum()
    stop = default_timer()
    return candidate_secret, hamming_weight, stop - start
