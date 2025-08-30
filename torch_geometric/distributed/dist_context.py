from dataclasses import dataclass
from enum import Enum

from torch_geometric.deprecation import deprecated


class DistRole(Enum):
    WORKER = 1


@dataclass
@deprecated("Use 'CuGraph' instead.\
    See https://github.com/rapidsai/cugraph-gnn/tree/main/python/cugraph-pyg/cugraph_pyg/examples") # noqa
class DistContext:
    r"""Context information of the current process.
    Deprecated, use 'CuGraph' instead.
    See `cuGraph examples repo <https://github.com/rapidsai/cugraph-gnn/tree/main/python/cugraph-pyg/cugraph_pyg/examples>`_. # noqa
    """
    rank: int
    global_rank: int
    world_size: int
    global_world_size: int
    group_name: str
    role: DistRole = DistRole.WORKER

    @property
    def worker_name(self) -> str:
        return f'{self.group_name}-{self.rank}'
