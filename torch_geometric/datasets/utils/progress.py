from __future__ import division

import os
import sys

from .spinner import Spinner

CLEAR = '\r\033[K'
UP = '\033[1A'
DOWN = '\033[1B'
STORE = '\033[s'
RESTORE = '\033[u'


class Progress(object):
    def __init__(self, title, name='', end=100, type='%', interval=1):
        self.spinner = Spinner(title, name, interval)
        self.lock = self.spinner.lock

        col = os.get_terminal_size().columns
        self.count = '{}/{}{}'.format(type, end, type)
        self.curr_size = len(str(end))
        self.bar_size = col - self.curr_size - len(self.count) - 3
        self.incomplete = '░' * self.bar_size
        self.complete = '▓' * self.bar_size
        self.end = end

        sys.stdout.write('\n')
        self.update(0)

        self.spinner.reset_cursor = '{}{}{}'.format(STORE, UP, CLEAR)
        self.spinner.restore_cursor = '{}{}'.format(DOWN, RESTORE)
        self.spinner.start()

    def update(self, curr):
        per = round((curr / self.end) * self.bar_size)
        bar = '{}{}'.format(self.complete[:per], self.incomplete[per:])
        count = '{}{}'.format(str(curr).rjust(self.curr_size), self.count)

        self.lock.acquire()
        sys.stdout.write('{}{} {}'.format(CLEAR, bar, count))
        self.lock.release()
        sys.stdout.flush()

    def success(self):
        self.spinner.success()
        sys.stdout.write(CLEAR)
        sys.stdout.flush()

    def fail(self):
        self.spinner.fail()
        sys.stdout.write(CLEAR)
        sys.stdout.flush()
