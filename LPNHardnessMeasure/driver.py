from logging import DEBUG, INFO

from benchmarking.benchmark import LPNBenchmark
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from utilities.logging import set_logging_defaults


def driver():
    """
    Main driver function for the LPN Benchmarking process.

    Sets up logging, configures the classifiers to be used and
    initiates the benchmarking process by running them in parallel.
    """
    set_logging_defaults(INFO)

    # Define classifiers to be benchmarked
    classifiers = {
        "Decision Trees Classifier": ExtraTreesClassifier,
        "Random Forest Classifier": RandomForestClassifier,
        "Decision Tree Classifier": DecisionTreeClassifier,
    }
    benchmarker = LPNBenchmark(classifiers)
    benchmarker.run_parallel_benchmarks()


if __name__ == "__main__":
    driver()
