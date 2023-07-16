import gc
import json
import logging
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, wait
from itertools import product
from multiprocessing.synchronize import Event
from typing import Any, Callable, Dict, Union

import numpy as np
from learning.classifiers import (
    ClassifierType,
    generate_samples,
    predict_with_classifier,
)
from utilities.concurrency import (
    ConcurrentResult,
    CustomSyncManager,
    new_concurrent_result,
)
from utilities.config import DATA_FILE, Config
from utilities.logging import create_callback, log_memory_usage


class LPNBenchmark:
    """
    Class to handle benchmarking of various classifiers.

    This class uses multiprocessing to run benchmarks on multiple classifiers in parallel,
    generating samples and secrets, making predictions, and recording the results. The results
    are written to a newline-delimited JSON file.
    """

    def __init__(
        self,
        classifiers: Dict[str, ClassifierType],
        model: Callable = predict_with_classifier,
    ) -> None:
        """
        Initializes an instance of the LPNBenchmark class.

        Args:
            classifiers (Dict[str, ClassifierType]): A dictionary mapping classifier names to their corresponding
                                                     ClassifierType instances. These classifiers will be benchmarked.
            model (Callable, optional): The model function to be used for making predictions during the benchmark.
                                        By default, it uses the `predict_with_classifier` function.

        Note:
            The initialization of this class also involves loading the configuration via the `Config` class.
            The `_termination_event` attribute, which is used to signal termination of benchmarking, is declared but not initialized here.
        """
        self.config = Config()
        self.classifiers = classifiers
        self._termination_event: Event
        self.model = model

    @staticmethod
    def calculate_eps(n: int, p: float) -> float:
        """
        Calculates epsylon

        Args:
            n (int): The number of samples.
            p (float): The error rate.

        Returns:
            float: The calculated epsylon.
        """
        probability_threshhold = 0.99
        return math.sqrt(-3 * math.log(1 - probability_threshhold) / (n * p))

    def run_parallel_benchmarks(self):
        """
        Runs benchmarks for different parameters in parallel using multiprocessing.

        This method will start a process pool with the number of CPUs as the number of workers,
        then submit tasks to it for different combinations of error rates, sample amounts, and dimensions.
        """
        logging.debug("Entering run_parallel_benchmarks")

        with CustomSyncManager() as manager:
            self._termination_event = manager.Event()

            cpus = mp.cpu_count()
            logging.info(f"Using {cpus} processes")

            with ProcessPoolExecutor(cpus) as exe:
                self._parallelize(exe, manager, cpus)

    def _parallelize(
        self, exe: ProcessPoolExecutor, manager: CustomSyncManager, cpu_cnt: int
    ):
        """
        Submit tasks to the executor for each combination of error rates, sample amounts, and dimensions.

        Args:
            exe (ProcessPoolExecutor): The executor for running tasks in parallel.
            manager (CustomSyncManager): The manager for shared state among processes.
            cpu_cnt (int): The number of CPUs to use for the process pool.

        Raises:
            Exception: If an error is encountered during the parallelization, the termination event is set.
        """

        futures = []
        concurrent_result = new_concurrent_result(self.classifiers, manager)
        for error_rate, sample_amount, dim in product(
            self.config.error_rates,
            self.config.sample_amounts,
            self.config.dimensions,
        ):
            try:
                if len(futures) == cpu_cnt:
                    # We wait if we reach cpu count
                    self._wait(futures, concurrent_result)
                    concurrent_result = new_concurrent_result(self.classifiers, manager)
                    futures = []

                logging.info(
                    f"Submitting task for dim={dim}, sample_amount={sample_amount}, error_rate={error_rate}"
                )
                future = exe.submit(
                    self._run_benchmarks,
                    dim,
                    sample_amount,
                    error_rate,
                    concurrent_result,
                )
                future.add_done_callback(
                    create_callback(dim, sample_amount, error_rate)
                )
                futures.append(future)

                if self._termination_event.is_set():
                    logging.info("termination event was set, breaking")
                    break

            # Probably unnecessary but better safe than sorry
            except Exception as exc:
                logging.warning(f"got exception in_parallelize", exc_info=exc)
                self._termination_event.set()

        if futures:
            self._wait(futures, concurrent_result)

    def _wait(self, futures: Any, concurrent_result: ConcurrentResult):
        """
        Waits for all futures to complete and writes the results to file.

        Args:
            futures (Any): The list of futures returned from the executor.
            concurrent_result (ConcurrentResult): The concurrent results to be written to the file.
        """
        wait(futures)
        self._write_ndjson(concurrent_result)

    @staticmethod
    def _write_ndjson(concurrent_result: ConcurrentResult):
        """
        Writes the concurrent results to a newline-delimited JSON file.

        The method writes the results to a file specified by DATA_FILE and also logs memory usage.
        """
        log_memory_usage("write_ndjson (before write)")
        with open(DATA_FILE, "a") as write_h:
            print(
                json.dumps(concurrent_result.serialize()),
                file=write_h,
            )

    def _run_benchmarks(
        self,
        dim: int,
        sample_amount: int,
        error_rate: float,
        concurrent_result: ConcurrentResult,
    ):
        """
        Runs benchmarks for a specific set of parameters.

        This method will generate samples and perform predictions with different classifiers,
        then record the results in the concurrent_result object. If an exception is encountered during
        benchmarking, the termination event will be set, halting the benchmarking process.

        Args:
            dim (int): The dimension, i.e the length of the secret.
            sample_amount (int): The number of samples.
            error_rate (float): The error rate.
            concurrent_result (ConcurrentResult): Shared proxy object to store the benchmark results.

        """

        logging.info("Entering run_benchmarks")
        log_memory_usage(
            f"run_benchmarks (start) - dim={dim}, sample_amount={sample_amount}"
        )
        if self._termination_event.is_set():
            return

        try:
            for key, classifier in self.classifiers.items():
                logging.info(f"Running benchmark for classifier {classifier}")
                result = self._benchmark_classifier(
                    error_rate,
                    dim,
                    sample_amount,
                    classifier,
                )
                concurrent_result[key].append(result)
                logging.debug(f"concurrent_result: {concurrent_result.serialize()}")

            log_memory_usage(
                f"run_benchmarks (end) - dim={dim}, sample_amount={sample_amount}"
            )

        except Exception as exc:
            logging.warning(
                f"got exception in run_benchmarks, terminating..",
                exc_info=exc,
            )
            self._termination_event.set()  # Set termination event.

    def _benchmark_classifier(
        self,
        error_rate: float,
        dim: int,
        sample_amount: int,
        classifier: ClassifierType,
    ) -> Dict[str, Union[int,float]]:
        """
        Performs benchmarking on a specific classifier.

        Generates random secrets and samples, performs predictions with the provided classifier,
        measures total duration of predictions, and returns results in a serialized format.

        Args:
            error_rate (float): Error rate for the benchmark. 0.1 = 10% error rate.
            dim (int): Dimension, i.e. the length of the secret.
            sample_amount (int): Number of samples to be generated.
            classifier (ClassifierType): The classifier object to be used for the benchmark.

        Returns:
            Dict[str, str]: A dictionary containing benchmark results, including
                            number of successful predictions, failed predictions, error rate,
                            and total prediction duration for a set of sample amounts and dimensions.

        Note:
            This method includes garbage collection after each iteration to manage memory usage.
        """
        p = error_rate
        n_success = 0
        n_failure = 0
        threshhold = (
            (1 + self.calculate_eps(sample_amount, error_rate)) * sample_amount * p
        )
        result: Dict[str, Union[int,float]] = {}

        total_duration = 0.0
        for _ in range(self.config.amount_of_secrets):
            log_memory_usage(
                "benchmark_classifier generating samples and predictions (loop start)"
            )
            secret = np.random.randint(0, 2, dim)
            A, b = generate_samples(secret, sample_amount, error_rate)
            cs, hw, duration = self.model(classifier, A, b, dim)
            total_duration += duration

            if hw < threshhold:
                if np.array_equal(cs, secret):
                    n_success += 1

                else:
                    # This is just a sanity check. This should not happen.
                    # If the hamming weight is below the threshold, we should have found
                    # the real secret.
                    n_failure += 1
            log_memory_usage(
                "benchmark_classifier  generating samples and predictions (loop end). Calling garbage collector.."
            )
            gc.collect()

        result = {
            "sample_amount": sample_amount,
            "dim": dim,
            "n_failure": n_failure,
            "error_rate": error_rate,
            "total_duration": total_duration,
        }
        log_memory_usage(
            "benchmark_classifier generating samples and predictions (end)"
        )

        return result
