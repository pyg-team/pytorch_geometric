"""Module to load greyc datasets as list of networkx graphs
"""

import os
from typing import List, Union

from torch_geometric.io.file_managers import DataLoader

PATH = os.path.dirname(__file__)


def one_hot_encode(val: Union[int, str], allowable_set: Union[List[str],
                                                              List[int]],
                   include_unknown_set: bool = False) -> List[float]:
    """One hot encoder for elements of a provided set.

    Examples:
    --------
    >>> one_hot_encode("a", ["a", "b", "c"])
    [1.0, 0.0, 0.0]
    >>> one_hot_encode(2, [0, 1, 2])
    [0.0, 0.0, 1.0]
    >>> one_hot_encode(3, [0, 1, 2])
    [0.0, 0.0, 0.0]
    >>> one_hot_encode(3, [0, 1, 2], True)
    [0.0, 0.0, 0.0, 1.0]

    Parameters
    ----------
    val: int or str
            The value must be present in `allowable_set`.
    allowable_set: List[int] or List[str]
            List of allowable quantities.
    include_unknown_set: bool, default False
            If true, the index of all values not in `allowable_set` is `len(allowable_set)`.

    Returns:
    -------
    List[float]
            An one-hot vector of val.
            If `include_unknown_set` is False, the length is `len(allowable_set)`.
            If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.

    Raises:
    ------
    ValueError
            If include_unknown_set is False and `val` is not in `allowable_set`.
    """
    if include_unknown_set is False:
        if val not in allowable_set:
            logger.info("input {} not in allowable set {}:".format(
                val, allowable_set))

    # init an one-hot vector
    if include_unknown_set is False:
        one_hot_legnth = len(allowable_set)
    else:
        one_hot_legnth = len(allowable_set) + 1
    one_hot = [0.0 for _ in range(one_hot_legnth)]

    try:
        one_hot[allowable_set.index(val)] = 1.0  # type: ignore
    except:
        if include_unknown_set:
            # If include_unknown_set is True, set the last index is 1.
            one_hot[-1] = 1.0
        else:
            pass
    return one_hot


def prepare_graph(
        graph, atom_list=['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'H']):
    """Prepare graph to include all data before pyg conversion
    Parameters
    ----------
    graph : networkx graph
        graph to be prepared
    atom_list : list[str], optional
        List of node attributes to be converted into one hot encoding
    """
    # Convert attributes.
    graph.graph = {}
    for node in graph.nodes:
        graph.nodes[node]['atom_symbol'] = one_hot_encode(
            graph.nodes[node]['atom_symbol'],
            atom_list,
            include_unknown_set=True,
        )
        graph.nodes[node]['degree'] = float(graph.degree[node])
        for attr in ['x', 'y', 'z']:
            graph.nodes[node][attr] = float(graph.nodes[node][attr])

    for edge in graph.edges:
        for attr in ['bond_type', 'bond_stereo']:
            graph.edges[edge][attr] = float(graph.edges[edge][attr])

    return graph


def _load_greyc_networkx_graphs(dir: str, name: str):
    """Load the dataset as a llist of networkx graphs and
    returns list of graphs and list of properties

    Args:
       dataset:str the dataset to load (Alkane,Acyclic,...)

    Returns:
       list of nx graphs
       list of properties (float or int)
    """
    loaders = {
        "alkane": _loader_alkane,
        "acyclic": _loader_acyclic,
        "mao": _loader_mao
    }
    loader_f = loaders.get(name, None)
    loader = loader_f(dir)
    if loader is None:
        raise Exception("Dataset Not Found")

    graphs = [prepare_graph(graph) for graph in loader.graphs]
    return graphs, loader.targets


def read_greyc(dir: str, name: str):
    return _load_greyc_networkx_graphs(dir, name)


def _loader_alkane(dir: str):
    """Load the 150 graphs of Alkane datasets
    returns two lists
    - 150 networkx graphs
    - boiling points
    """
    dloader = DataLoader(
        os.path.join(dir, 'dataset.ds'),
        filename_targets=os.path.join(dir, 'dataset_boiling_point_names.txt'),
        dformat='ds', gformat='ct', y_separator=' ')
    return dloader


def _loader_acyclic(dir: str):
    dloader = DataLoader(os.path.join(dir,
                                      'dataset_bps.ds'), filename_targets=None,
                         dformat='ds', gformat='ct', y_separator=' ')
    return dloader


def _loader_mao(dir: str):
    dloader = DataLoader(os.path.join(dir,
                                      'dataset.ds'), filename_targets=None,
                         dformat='ds', gformat='ct', y_separator=' ')
    dloader._targets = [int(yi) for yi in dloader.targets]
    return dloader
