from multiprocessing.managers import DictProxy, SyncManager
from typing import Any, Dict, List


class ConcurrentResult(DictProxy):
    """
    A wrapper class around DictProxy to add a serialize method.

    Methods:
        serialize: Serializes the concurrent result into a format that can be written to disk.
    """

    def serialize(self) -> Dict[str, List[Any]]:
        """
        Serializes the current ConcurrentResult into a dictionary where each value is a list.

        Returns:
            dict: A dictionary with keys from the current object and values converted to lists.
        """
        return {key: list(value) for key, value in self.items()}


class CustomSyncManager(SyncManager):
    """
    A custom SyncManager that overwrites the dict method with ConcurrentResult.
    """

    pass  # Just to be explicit.


CustomSyncManager.register("dict", dict, ConcurrentResult)


def new_concurrent_result(
    classifiers: Dict[str, Any],
    manager: CustomSyncManager,
) -> ConcurrentResult:
    """
    Creates a new ConcurrentResult using the provided classifiers and manager.

    Args:
        classifiers (Dict[str, Any]): A dictionary of classifier objects.
        manager (CustomSyncManager): A CustomSyncManager to create the ConcurrentResult with.

    Returns:
        ConcurrentResult: A new ConcurrentResult object.
    """
    return manager.dict({key: manager.list() for key in classifiers.keys()})
