from collections.abc import Callable, Iterator
from queue import Queue, Empty
from typing import Any

from .operator_abc import Operator, RunningUnit


class SyncTask(RunningUnit):
    def __init__(self, task: Callable[[Callable[[Any], None]], None]):
        self._func = task
        self._pipe = Queue()

    def run(self):
        self._func(self.send_from_worker)

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


class SyncOperator(Operator):
    def create_running_unit(self, task: Callable[[Callable[[Any], None]], None]) -> SyncTask:
        return SyncTask(task)
