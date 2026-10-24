import sys
from pathlib import Path

_PYG_ROOT = Path(__file__).resolve().parent.parent.parent
_NEO4J_DATA = _PYG_ROOT / "examples" / "neo4j" / "data"
for _p in (str(_PYG_ROOT), str(_NEO4J_DATA)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
