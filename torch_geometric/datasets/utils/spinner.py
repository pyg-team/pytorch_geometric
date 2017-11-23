import sys
import itertools
import time


def spinning_cursor():
    spinner = ['-', '/', '|', '\\']
    spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    spinner = itertools.cycle(spinner)

    while True:
        sys.stdout.write(next(spinner))
        sys.stdout.write('  ')
        sys.stdout.flush()
        sys.stdout.write('\b\b\b')
        time.sleep(.1)


spinning_cursor()
