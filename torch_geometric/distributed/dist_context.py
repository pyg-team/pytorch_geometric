from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DistRole(Enum):
    WORKER = 1  # for worker mode


@dataclass
class DistContext:
    r""" Context info for distributed Info in the current process."""
    rank: int
    global_rank: int
    world_size: int
    global_world_size: int
    group_name: str
    role: DistRole = DistRole.WORKER

    @property
    def worker_name(self) -> str:
        return f"{self.group_name}-{self.rank}"
