from threading import Thread
from queue import Queue, Empty

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


class SyncOperator(Operator):
    def __init__(self, task: Callable[[], None]):
        self._func = task
        self._pipe = Queue()

    def run(self):
        self._func()

    def wait_to_stop(self):
        pass

    def send_from_worker(self, value: Any):
        self._pipe.put(value)

    def receive_from_worker(self, no_wait: bool = False) -> Iterator[Any]:
        reception_method = self._pipe.get_nowait if no_wait else self._pipe.get

        while True:
            try:
                yield reception_method()
            except Empty:
                return


class ThreadOperator(Operator):
    def __init__(self, task: Callable[[], None]):
        self._thread = Thread(target=task)
        self._pipe = Queue()

    def run(self):
        self._thread.start()

    def wait_to_stop(self):
        self._thread.join()

    def send_from_worker(self, value: Any):
        self._pipe.put(value)

    def receive_from_worker(self, no_wait: bool = False) -> Iterator[Any]:
        reception_method = self._pipe.get_nowait if no_wait else self._pipe.get

        while True:
            try:
                yield reception_method()
            except Empty:
                return
