import asyncio
import logging
from threading import BoundedSemaphore, Thread

import torch


def wrap_torch_future(f: torch.futures.Future) -> asyncio.futures.Future:
    r""" Convert a torch future to a standard asyncio future."""
    loop = asyncio.get_event_loop()
    aio_future = loop.create_future()

    def on_done(*_):
        try:
            result = f.wait()
        except Exception as e:
            loop.call_soon_threadsafe(aio_future.set_exception, e)
        else:
            loop.call_soon_threadsafe(aio_future.set_result, result)

    f.add_done_callback(on_done)
    return aio_future


class ConcurrentEventLoop(object):
    r""" Concurrent event loop context.

    Args:
        concurrency: max processing concurrency.
    """
    def __init__(self, concurrency):
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
        r""" Wait all pending tasks to be finished."""
        for _ in range(self._concurrency):
            self._sem.acquire()
        for _ in range(self._concurrency):
            self._sem.release()

    def add_task(self, coro, callback=None):
        r""" Add an asynchronized coroutine task to run.

        Args:
        coro: the async coroutine func.
        callback: the callback func applied on the returned results
            after the coroutine task is finished.

        Note that any results returned by `callback` func will be ignored,
        so it is preferable to handle all in your `callback` func and do
        not return any results.
        """
        def on_done(f: asyncio.futures.Future):
            try:
                res = f.result()
                if callback is not None:
                    callback(res)
            except Exception as e:
                logging.error("coroutine task failed: %s", e)
            self._sem.release()

        self._sem.acquire()
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        fut.add_done_callback(on_done)

    def run_task(self, coro):
        r""" Run a coroutine task synchronously.
        """
        with self._sem:
            fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
            return fut.result()

    def _run_loop(self):
        self._loop.run_forever()
