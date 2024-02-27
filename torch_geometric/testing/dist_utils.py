import pytest
import traceback

from typing import Iterable

from torch.multiprocessing import (Queue, Process, Manager)

from io import StringIO
import sys


class MPCaptOutput:
    def __enter__(self):
        self._out_value = ''
        self._err_value = ''

        self._stdout = sys.stdout
        sys.stdout = StringIO()

        self._stderr = sys.stderr
        sys.stderr = StringIO()

        return self

    def __exit__(self, *args):
        self._out_value = sys.stdout.getvalue()
        sys.stdout = self._stdout

        self._err_value = sys.stderr.getvalue()
        sys.stderr = self._stderr

    def get_stdout(self):
        return self._out_value

    def get_stderr(self):
        return self._err_value


def ps_std_capture(func, queue: Queue, *args, **kwargs):
    try:
        with MPCaptOutput() as capt:
            try:
                func(*args, **kwargs)
            except Exception as e:
                traceback.print_exc(file=sys.stderr)
                raise e
    finally:
        queue.put((capt.get_stdout(), capt.get_stderr()))


def assert_run_mproc(
        procs: Iterable[Process],
        full_trace=False,
        timeout=5,
):
    manager = Manager()
    world_size = len(procs)
    queues = [manager.Queue() for _ in procs]

    results = []

    for p, q in zip(procs, queues):
        target = p._target
        p._target = ps_std_capture
        p._args = [target, q, world_size] + list(p._args)
        p.start()

    for p, q in zip(procs, queues):
        p.join()
        stdout, stderr = q.get(timeout=timeout)
        results.append((p, stdout, stderr))

    for p, stdout, stderr in results:
        if stdout:
            print(stdout)
        if stderr:  # can be warning as well => exitcode == 0
            print(stderr)
        if p.exitcode != 0:
            pytest.fail(pytrace=full_trace, reason=stderr.splitlines()[-1])
