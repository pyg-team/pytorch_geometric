import sys
from os.path import dirname, basename, join, isfile, realpath
import glob

# Protobuf does not support relative paths in generated
# files, so we need this ugly workaround (or something
# even uglier...)
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

# Pass-through all packages
modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]
