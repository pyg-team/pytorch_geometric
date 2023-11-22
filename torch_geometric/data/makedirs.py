from torch_geometric.io import fs


def makedirs(path: str):
    r"""Recursively creates a directory.

    Args:
        path (str): The path to create.
    """
    fs.makedirs(path, exist_ok=True)
