from __future__ import division

import os
import sys
from itertools import cycle
import time
import threading

BOLD = '\033[1m'
SUCCESS = '\033[92m'
FAIL = '\033[91m'
RUN = '\033[93m'
ENDC = '\033[0m'
CLEAR = '\r\033[K'


class Spinner(object):
    spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    spinner_cycle = cycle(spinner)

    def __init__(self, title, name='', interval=1):
        col = os.get_terminal_size().columns
        self.title = '{}{}{} {}'.format(BOLD, title, ENDC, name)[:col + 4]
        self.interval = interval

        self.thread = threading.Thread(target=self.init_thread)
        self.thread.daemon = True
        self.stop_running = threading.Event()
        self.lock = threading.Lock()

        self._print(self._next_spinner())
        self.reset_cursor = CLEAR
        self.restore_restore = ''

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_running.set()
        self.thread.join()

    def _print(self, marker, end=''):
        str = '{} {}{}'.format(marker, self.title, end)
        self.lock.acquire()
        sys.stdout.write(getattr(self, 'reset_cursor', ''))
        sys.stdout.write(str)
        sys.stdout.write(getattr(self, 'restore_cursor', ''))
        self.lock.release()
        sys.stdout.flush()

    def _next_spinner(self):
        return '{}{}{}'.format(RUN, next(self.spinner_cycle), ENDC)

    def init_thread(self):
        while not self.stop_running.is_set():
            self._print(self._next_spinner())
            time.sleep(1 / (self.interval * len(self.spinner)))

    def success(self):
        self.stop()
        self._print('{}✔{}'.format(SUCCESS, ENDC), end='\n')

    def fail(self):
        self.stop()
        self._print('{}✖{}'.format(FAIL, ENDC), end='\n')
