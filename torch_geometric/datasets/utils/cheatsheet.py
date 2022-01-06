from typing import List, Optional, Any

import re
import inspect
import importlib


def paper_link(cls: str) -> str:
    cls = importlib.import_module('torch_geometric.datasets').__dict__[cls]
    match = re.search('<.+?>', inspect.getdoc(cls), flags=re.DOTALL)
    return None if match is None else match.group().replace('\n', ' ')[1:-1]


def get_stats_table(cls: str) -> str:
    cls = importlib.import_module('torch_geometric.datasets').__dict__[cls]
    match = re.search('Stats:\n.*$', inspect.getdoc(cls), flags=re.DOTALL)
    return '' if match is None else match.group()


def has_stats(cls: str) -> bool:
    return len(get_stats_table(cls)) > 0


def get_stat(cls: str, name: str, child: Optional[str] = None,
             default: Any = None) -> str:
    if child is None and len(get_children(cls)) > 0:
        return ''

    stats_table = get_stats_table(cls)

    match = re.search(f'^.*- {name}', stats_table, flags=re.DOTALL)
    if match is None:
        return default

    column = match.group().count(' -')

    if child is not None:
        match = re.search(f'[*] - {child}\n.*$', stats_table, flags=re.DOTALL)
        stats_row = match.group()
    else:
        stats_row = '*' + stats_table.split('*')[2]

    return stats_row.split(' -')[column].split('\n')[0].strip()


def get_children(cls: str) -> List[str]:
    matches = re.findall('[*] -.*', get_stats_table(cls))
    return [match[4:] for match in matches[1:]] if len(matches) > 2 else []
