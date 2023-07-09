from enum import Enum
from typing import Optional


class DistRole(Enum):
    WORKER = 1  # for worker mode


class DistContext(object):
    r""" Context info for distributed Info in the current process."""
    def __init__(self, role: DistRole, rank: int, group_name: str,
                 world_size: int, global_world_size: int, global_rank: int):
        self.role = role
        self.rank = rank
        self.group_name = group_name
        self.world_size = world_size
        self.global_world_size = global_world_size
        self.global_rank = global_rank

    @property
    def worker_name(self) -> str:
        return f"{self.group_name}-{self.rank}"


_dist_context: DistContext = None


def get_context() -> DistContext:
    return _dist_context


def init_worker_group(world_size: int, rank: int,
                      group_name: Optional[str] = None):
    global _dist_context
    _dist_context = DistContext(
        role=DistRole.WORKER, world_size=world_size, rank=rank,
        group_name=(group_name
                    if group_name is not None else '_default_worker'),
        global_world_size=world_size, global_rank=rank)
