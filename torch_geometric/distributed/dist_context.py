from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DistRole(Enum):
    WORKER = 1  # for worker mode


@dataclass
class DistContext:
    r""" Context info for distributed Info in the current process."""
    def __init__(self, rank: int, global_rank: int, world_size: int,
                 global_world_size: int, group_name: str,
                 role: DistRole = DistRole.WORKER):
        self.rank = rank
        self.global_rank = global_rank
        self.world_size = world_size
        self.global_world_size = global_world_size
        self.group_name = group_name
        self.role = role

    @property
    def worker_name(self) -> str:
        return f"{self.group_name}-{self.rank}"
