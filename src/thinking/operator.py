from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import Any


class Operator(ABC):
    @abstractmethod
    def __init__(self, task: Callable[[], None]):
        ...

    @abstractmethod
    def run(self):
        ...

    @abstractmethod
    def wait_to_stop(self):
        ...

    @abstractmethod
    def send_from_worker(self, value: Any):
        ...

    @abstractmethod
    def receive_from_worker(self, no_wait: bool = False) -> Iterator[Any]:
        ...
