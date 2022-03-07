froom pathlib import Path

modules = Path(__file__).parent.rglob('*.py')
__all__ = [
    Path(f).parent[:-3] for f in modules
    if Path(f).is_file() and not f.endswith('__init__.py')
]
