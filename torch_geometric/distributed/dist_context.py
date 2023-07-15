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
