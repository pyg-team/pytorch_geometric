import asyncio
import logging
from threading import BoundedSemaphore, Thread
from typing import Callable, Optional

import torch

# This code is taken from alibaba graphlearn-for-pytorch repository:
# https://github.com/alibaba/graphlearn-for-pytorch/blob/main/graphlearn_torch/
# python/distributed/event_loop.py


def to_asyncio_future(future: torch.futures.Future) -> asyncio.futures.Future:
    r"""Convert a :class:`torch.futures.Future` to a standard asyncio future.
    """
    loop = asyncio.get_event_loop()
    asyncio_future = loop.create_future()

    def on_done(*_):
        try:
            result = future.wait()
        except Exception as e:
            loop.call_soon_threadsafe(asyncio_future.set_exception, e)
        else:
            loop.call_soon_threadsafe(asyncio_future.set_result, result)

    future.add_done_callback(on_done)

    return asyncio_future


class ConcurrentEventLoop:
    r"""Concurrent event loop context.

    Args:
        concurrency: max processing concurrency.
    """
    def __init__(self, concurrency: int):
        self._concurrency = concurrency
        self._sem = BoundedSemaphore(concurrency)
        self._loop = asyncio.new_event_loop()
        self._runner_t = Thread(target=self._run_loop)
        self._runner_t.daemon = True

    def start_loop(self):
        if not self._runner_t.is_alive():
            self._runner_t.start()

    def shutdown_loop(self):
        self.wait_all()
        if self._runner_t.is_alive():
            self._loop.stop()
            self._runner_t.join(timeout=1)

    def wait_all(self):
        r"""Wait for all pending tasks to be finished."""
        for _ in range(self._concurrency):
            self._sem.acquire()
        for _ in range(self._concurrency):
            self._sem.release()

    def add_task(self, coro, callback: Optional[Callable] = None):
        r"""Adds an asynchronized coroutine task to run.

        Args:
            coro: The asynchronous coroutine function.
            callback (callable, optional): The callback function applied on the
                returned results after the coroutine task is finished.
                (default: :obj:`None`)

        Note that any result returned by :obj:`callback` will be ignored.
        """
        def on_done(f: asyncio.futures.Future):
            try:
                res = f.result()
                if callback is not None:
                    callback(res)
            except Exception as e:
                logging.error(f"Coroutine task failed with error: {e}")
            self._sem.release()

        self._sem.acquire()
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        fut.add_done_callback(on_done)

    def run_task(self, coro):
        r"""Runs a coroutine task synchronously.

        Args:
            coro: The synchronous coroutine function.
        """
        with self._sem:
            fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
            return fut.result()

    def _run_loop(self):
        self._loop.run_forever()
