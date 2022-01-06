from typing import List, Optional, Any

import re
import inspect
import importlib


def paper_title(cls: str) -> str:
    cls = importlib.import_module('torch_geometric.datasets').__dict__[cls]
    match = re.search('`\".+?\"', inspect.getdoc(cls), flags=re.DOTALL)
    return None if match is None else match.group().replace('\n', ' ')[2:-1]


def paper_link(cls: str) -> str:
    cls = importlib.import_module('torch_geometric.datasets').__dict__[cls]
    match = re.search('<.+?>', inspect.getdoc(cls), flags=re.DOTALL)
    return None if match is None else match.group().replace('\n', ' ')[1:-1]


def get_stats_table(cls: str) -> str:
    cls = importlib.import_module('torch_geometric.datasets').__dict__[cls]
    match = re.search('[*] - Name\n.*$', inspect.getdoc(cls), flags=re.DOTALL)
    return '' if match is None else match.group()


def num_children(cls: str) -> int:
    return max(get_stats_table(cls)[1:].count('*'), 0)


def get_children(cls: str) -> List[str]:
    matches = re.findall('[*] -.*', get_stats_table(cls)[1:])
    return [match[4:] for match in matches]


def get_stat(cls: str, name: str, child: Optional[str] = None,
             default: Any = None) -> str:
    stats_table = get_stats_table(cls)

    if child is None:  #  or child == '...':
        return default

    match = re.search(f'^[*] -.*{name}', stats_table, flags=re.DOTALL)
    if match is None:
        return default

    column = match.group().count(' -')

    if child is not None:
        match = re.search(f'[*] - {child}\n.*$', stats_table, flags=re.DOTALL)
        stats_row = match.group()

    return stats_row.split(' -')[column].strip()
