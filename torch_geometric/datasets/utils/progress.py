from __future__ import division

import os
import sys
import itertools

BOLD = '\033[1m'
SUCCESS = '\033[92m'
FAIL = '\033[91m'
SPINNER = '\033[93m'
ENDC = '\033[0m'
CLEAR = '\r\033[K'
UP = '\033[F'


class Progress(object):
    spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, title, name, end=100, type='%'):
        col = os.get_terminal_size().columns
        self.spinner = itertools.cycle(self.spinner)
        self.header = '{}{}{} {}'.format(BOLD, title, ENDC, name)[:col + 6]

        self.count = '{}/{}{}'.format(type, end, type)
        self.curr_size = len(str(end))

        self.bar_size = col - self.curr_size - len(self.count) - 2
        self.incomplete = '░' * self.bar_size
        self.complete = '▓' * self.bar_size
        self.end = end

        self._update(0)

    def _next_spinner(self):
        return '{}{}{}'.format(SPINNER, next(self.spinner), ENDC)

    def clear(self):
        sys.stdout.write('{}{}{}'.format(CLEAR, UP, CLEAR))

    def update(self, curr):
        self.clear()
        self._update(curr)

    def _update(self, curr):
        header = '{} {}'.format(self._next_spinner(), self.header)
        per = round((curr / self.end) * self.bar_size)
        bar = '{}{}'.format(self.complete[:per], self.incomplete[per:])
        count = '{}{}'.format(str(curr).rjust(self.curr_size), self.count)

        sys.stdout.write('{}\n{} {}'.format(header, bar, count))
        sys.stdout.flush()

    def success(self):
        self.clear()
        header = '{}✔{} {}\n'.format(SUCCESS, ENDC, self.header)
        sys.stdout.write(header)

    def fail(self):
        self.clear()
        header = '{}✖{} {}\n'.format(FAIL, ENDC, self.header)
        sys.stdout.write(header)
