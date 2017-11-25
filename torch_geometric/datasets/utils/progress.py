from __future__ import division

import os
import sys
from itertools import cycle
import time
import threading

BOLD = '\033[1m'
SUCCESS = '\033[92m'
FAIL = '\033[91m'
ACTIVE = '\033[93m'
ENDC = '\033[0m'

CLEAR = '\r\033[K'
UP = '\033[F'
BEGIN = '\r'
SAVE = '\033[s'
RESTORE = '\033[u'


class Progress(object):
    spinner = cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])

    def __init__(self, title, name, end=100, type='%'):
        col = os.get_terminal_size().columns
        print('  {}{}{} {}'.format(BOLD, title, ENDC, name)[:col + 8])

        self.count = '{}/{}{}'.format(type, end, type)
        self.curr_size = len(str(end))
        self.bar_size = col - self.curr_size - len(self.count) - 2
        self.incomplete = '░' * self.bar_size
        self.complete = '▓' * self.bar_size
        self.end = end

        self.update(0)

        self.thread = threading.Thread(target=self.init_thread)
        self.stop_running = threading.Event()
        self.thread.start()

    def init_thread(self):
        while not self.stop_running.is_set():
            spinner = '{}{}{}'.format(ACTIVE, next(self.spinner), ENDC)
            str = '{}{}{}{}{}'.format(SAVE, UP, BEGIN, spinner, RESTORE)
            sys.stdout.write(str)
            sys.stdout.flush()
            time.sleep(1 / 12)

    def stop_thread(self):
        self.stop_running.set()
        self.thread.join()

    def update(self, curr):
        per = round((curr / self.end) * self.bar_size)
        bar = '{}{}'.format(self.complete[:per], self.incomplete[per:])
        count = '{}{}'.format(str(curr).rjust(self.curr_size), self.count)
        sys.stdout.write('{}{} {}'.format(CLEAR, bar, count))
        sys.stdout.flush()

    def clear(self):
        sys.stdout.write('{}{}'.format(CLEAR, UP))

    def success(self):
        self.stop_thread()
        self.clear()
        str = '{}✔{}\n'.format(SUCCESS, ENDC)
        sys.stdout.write(str)
        sys.stdout.flush()

    def fail(self):
        self.stop_thread()
        self.clear()
        str = '{}✖{}\n'.format(FAIL, ENDC)
        sys.stdout.write(str)
        sys.stdout.flush()
