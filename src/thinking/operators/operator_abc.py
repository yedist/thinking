from collections.abc import Callable, Iterator
from typing import Any, Protocol


class RunningUnit(Protocol):

    def run(self) -> None:
        ...

    def wait_to_stop(self) -> None:
        ...

    def send_from_worker(self, value: Any) -> None:
        ...

    def receive_from_worker(self, no_wait: bool = False) -> Iterator[Any]:
        ...


class Operator(Protocol):
    def create_running_unit(self, task: Callable[[Callable[[Any], None]], None]) -> RunningUnit:
        ...
