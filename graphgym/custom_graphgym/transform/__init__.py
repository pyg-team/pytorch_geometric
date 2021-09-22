from pathhlib import Path
import glob

modules = glob.glob(Path.joinpath(Path(__file__).parent, "*.py"))
__all__ = [
    Path(f).parent[:-3] for f in modules
    if Path(f).is_file() and not f.endswith('__init__.py')
]
