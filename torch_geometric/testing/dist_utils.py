import pytest
import traceback

from typing import Collection, Callable, Any
from typing_extensions import Self

from torch.multiprocessing import (Queue, Manager)

from io import StringIO
import sys


class MPCaptOutput:
    def __enter__(self) -> Self:
        self.stdout = StringIO()
        self.stderr = StringIO()
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        sys.stdout = self.stdout
        sys.stderr = self.stderr

        return self

    def __exit__(self, *args: Any) -> None:
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

    @property
    def stdout_str(self) -> str:
        return self.stdout.getvalue()

    @property
    def stderr_str(self) -> str:
        return self.stderr.getvalue()


def ps_std_capture(
        func: Callable,
        queue: Queue,
        *args: Any,
        **kwargs: Any
) -> None:
    with MPCaptOutput() as capt:
        try:
            func(*args, **kwargs)
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            raise e
        finally:
            queue.put((capt.stdout_str, capt.stderr_str))


def assert_run_mproc(
        procs: Collection[Any],
        full_trace: bool = False,
        timeout: int = 5,
) -> None:
    manager = Manager()
    world_size = len(procs)
    queues = [manager.Queue() for _ in procs]
    results = []

    for p, q in zip(procs, queues):
        target = p._target
        p._target = ps_std_capture
        new_args = [target, q, world_size] + list(p._args)
        p._args = new_args
        assert p._target == ps_std_capture
        assert p._args == new_args

        p.start()

    for p, q in zip(procs, queues):
        p.join()
        stdout, stderr = q.get(timeout=timeout)
        results.append((p, stdout, stderr))

    for p, stdout, stderr in results:
        if stdout:
            print(stdout)
        if stderr:  # can be a warning as well => exitcode == 0
            print(stderr)
        if p.exitcode != 0:
            pytest.fail(
                pytrace=full_trace,
                reason=stderr.splitlines()[-1] if stderr else f"exitcode {p.exitcode}")
