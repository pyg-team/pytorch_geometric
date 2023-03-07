from typing import Any, Callable, Dict, List, Tuple, Union

import networkx as nx
import pandas as pd
import torch
from bidict import bidict

from torch_geometric.data import HeteroData
from torch_geometric.typing import HeteroColumnSet, HeteroDF


def _groupby_to_dict(df: pd.DataFrame, groupbycols: Union[str, List[str]]) \
        -> Dict[Any, pd.DataFrame]:
    r"""groups a dataframe by a col or list of cols and returns a dictionary
    of {groupby_values:group_df}, return a dictionary with the group by
    values as keys and the grouped dfs as values.
    Examples:

        >>> df = pd.DataFrame({
        ...  "student": ["Ram", "Noa", "Song", "Ram", "Ram", "Song", "Noa",
                                "Noa"],
        ...  "grades" : [100, 80, 90, 55, 50, 60, 70, 80],
        ...  "class" : ["history", "history", "history", "math",
                        "art", "history", "math", "art"]
                        })
        >>> print(pd.concat(
        ...     _groupby_to_dict(df, "student"), axis=0).to_markdown())
        ... |             |   grades | class   |
        ... |:------------|---------:|:--------|
        ... | ('Noa', 1)  |       80 | history |
        ... | ('Noa', 6)  |       70 | math    |
        ... | ('Noa', 7)  |       80 | art     |
        ... | ('Ram', 0)  |      100 | history |
        ... | ('Ram', 3)  |       55 | math    |
        ... | ('Ram', 4)  |       50 | art     |
        ... | ('Song', 2) |       90 | history |
        ... | ('Song', 5) |       60 | history |

    """
    dict_df = dict(list(df.groupby(groupbycols)))
    dict_df = {k: df.drop(groupbycols, axis=1) for k, df in dict_df.items()}
    return dict_df


def _node_edge_dfs_to_sig_df(nodes: pd.DataFrame,
                             edges: pd.DataFrame) -> pd.DataFrame:
    r"""
    Takes a nodes DF and edges DF and create a DF of all signatures by
    adding node type info to each row in the edge DF assumes nodes have
    "node_type" and "id" attributes and edges have "source","target" and
    "edge_type" attributes returns a signature df which is like the edge df
    except with "source_type" and "target_type" columns derived from the
    nodes df a signature df which is like the edge df except with
    "source_type" and "target_type" columns derived from the nodes df.

    Examples:
    >>> nodes = pd.DataFrame({
    ... "id": [1,2,3,4,5],
    ... "node_type": ["a", "a", "a", "b", "b"]})
    >>> edges = pd.DataFrame({
    ... "source": [1,2,3,5], "target":[2,4,5,1],
    ... "edge_type": ["c","c","c","c"]})
    >>> print(_node_edge_dfs_to_sig_df(nodes, edges).to_markdown())
    ... |    | source | target | source_type | edge_type | target_type|
    ... |---:|-------:|-------:|:------------|:----------|:-----------|
    ... |  0 |      1 |      2 | a           | c         | a          |
    ... |  1 |      2 |      4 | a           | c         | b          |
    ... |  2 |      3 |      5 | a           | c         | b          |
    ... |  3 |      5 |      1 | b           | c         | a          |
    """
    sig_df = \
        edges.merge(nodes, how="left", left_on="source", right_on="id").merge(
            nodes,
            how="left",
            left_on="target",
            right_on="id",
            suffixes=("_source", "_target"),
        ).drop(["id_source", "id_target"], axis=1).rename(
            {
                "node_type_source": "source_type",
                "node_type_target": "target_type"
            },
            axis="columns",
        )[["source", "target", "source_type", "edge_type", "target_type"]]
    return sig_df


def nx2heteroDf(g: nx.MultiDiGraph, node_groupbycols: str = "node_type",
                edge_groupbycols: List[str] = None) -> HeteroDF:
    r"""Convert a networkx graph to
    :class:`~torch_geometric.typing.HeteroDF`. assumes nodes have
    "node_type" and "id" attributes and edges have "edge_type" attributes.
    Returns a HeteroDF representation of the graph.

    Args:
        g (:class:`~nx.classes.multidigraph.MultiDiGraph`): Heterogeneous
            Graph (A directed graph class that can store multiedges)
        node_groupbycols: the column name representing the node types
        edge_groupbycols: the columns names representing the edge source ,
            type and target

    Returns:
        :class:`~torch_geometric.typing.HeteroDF`

    Example:
        >>> college_nx = nx.MultiDiGraph()
        >>> college_nx.add_nodes_from([
        ... (1, {"id": 1, "name": "bob", "node_type": "student", }),
        ... (2, {"id": 2, "name": "dan", "node_type": "student", }),
        ... (3, {"id": 3, "name": "tal", "node_type": "ta", }),
        ... (4, {"id": 4, "name": "profG", "node_type": "lecturer", }),
        ... (5, {"id": 5, "name": "phys101", "node_type": "class", }),
        ... (6, {"id": 6, "name": "phys102", "node_type": "class", }),
        ... (7, {"id": 7, "name": "chem101", "node_type": "class", }),
        ... (8, {"id": 8, "name": "physics", "node_type": "topic"}),
        ... (9, {"id": 9, "name": "chemistry", "node_type": "topic"}),
        ... ])
        >>> college_nx.add_edges_from([
        ... (1, 5, {"edge_type": "takes"}),
        ... (1, 7, {"edge_type": "takes"}),
        ... (2, 6, {"edge_type": "takes"}),
        ... (3, 5, {"edge_type": "teaches"}),
        ... (4, 6, {"edge_type": "teaches"}),
        ... (5, 8, {"edge_type": "is_about"}),
        ... (6, 8, {"edge_type": "is_about"}),
        ... (6, 9, {"edge_type": "is_about"}),
        ... (7, 9, {"edge_type": "is_about"}),
        ... ])
        >>> nx_hetero_df = nx2heteroDf(college_nx)
        >>> print(nx_hetero_df.keys())
        >>> dict_keys(['class', 'lecturer', 'student', 'ta', 'topic',
        ... ('class', 'is_about', 'topic'),
        ... ('lecturer', 'teaches', 'class'),
        ... ('student', 'takes', 'class'),
        ... ('ta', 'teaches', 'class')])
    """

    # get dataframes of edge and node type and ids, without all features
    if edge_groupbycols is None:
        edge_groupbycols = ["source_type", "edge_type", "target_type"]
    node_df = pd.DataFrame.from_dict(dict(g.nodes.data()), orient="index")

    #
    node_type_df = node_df[["id", "node_type"]]
    edge_df = nx.to_pandas_edgelist(g)

    # join the node dfs on source and target, to get a signature df
    sig_df = _node_edge_dfs_to_sig_df(node_type_df, edge_df)

    # split group nodes and edges data and wrap in dict
    heterodf = {
        **_groupby_to_dict(node_df, groupbycols=node_groupbycols),
        **_groupby_to_dict(sig_df, groupbycols=edge_groupbycols),
    }
    return heterodf


def heteroDf_merge(hdf1: HeteroDF, hdf2: HeteroDF, **kwargs) -> HeteroDF:
    r"""merges 2 hetero dfs into 1, same keys are merged on "id" for nodes
    and ("source","target") for edges.
    Args:

        hdf1: :class:`~torch_geometric.typing.HeteroDF`.
        hdf2: :class:`~torch_geometric.typing.HeteroDF`.
        kwargs: Additional kwargs for function.

    Returns:
            merged :class:`~torch_geometric.typing.HeteroDF`
    Example:
        >>> het_df0 = {"A":pd.DataFrame.from_records([
        ... {"id": 0, "f0": 1, "f1": 2},
        ... {"id": 1, "f0": 3, "f1": 4}])}
        >>> het_df1 = {"A":pd.DataFrame.from_records([
        ... {"id": 0, "f2": 1, "f3": 2},
        ... {"id": 2, "f2": 2, "f3": 4}])}
        >>> print(pd.concat(
        ...     heteroDF_merge(het_df0,het_df1),axis=0).to_markdown())
        >>> |          |   id |   f0 |   f1 |   f2 |   f3 |
        ... |:---------|-----:|-----:|-----:|-----:|-----:|
        ... | ('A', 0) |    0 |    1 |    2 |    1 |    2 |
    """

    keys1 = set(hdf1.keys())
    keys2 = set(hdf2.keys())
    merge_het = {}

    for key in keys1.difference(keys2):
        merge_het[key] = hdf1[key]
    for key in keys2.difference(keys1):
        merge_het[key] = hdf2[key]
    for key in keys1.intersection(keys2):
        # merge node dfs
        if isinstance(key, str):
            merge_het[key] = hdf1[key]\
                .merge(hdf2[key], on="id", **kwargs)\
                .fillna(0)
        # merge signature dfs
        elif isinstance(key, tuple):
            merge_het[key] = hdf1[key]\
                .merge(hdf2[key], on=["source", "target"], **kwargs)

    return merge_het


def heteroDf_split(
    hdf: HeteroDF,
    split_by: Union[List[str], HeteroColumnSet],
    keep_index_in_matched: bool = False,
    keep_index_in_unmatched: bool = False,
) -> Tuple[HeteroDF, HeteroDF]:
    r"""
    splits a HeteroDf into two HeteroDFs according split_by. In cases where
    all columns of a certain df were matched/not matched, we can decide if
    we want a dataframe with node_id/source target pairs to still exist in
    the not matched/matched part.

    Args: hdf (:class:`~torch_geometric.typing.HeteroDF`): HeteroDf to
    split. split_by (:obj:`Union[List[str]]`,
    :class:`~torch_geometric.typing.HeteroColumnSet`): keep_index_in_matched
    (:obj:`bool`): keeping index in matched HeteroDF keep_index_in_unmatched
    (:obj:`bool`): keeping index in unmatched HeteroDF
    Example:
        >>> het_df = {
        ... "A":pd.DataFrame.from_records(
        ...     [{"id": 0, "f0": 1, "f1": 2},
        ...     {"id": 1, "f0": 3, "f1": 4}])
        ... ,"B":pd.DataFrame.from_records(
        ...     [{"id": 0, "f0": 1, "f2": 2},
        ...     {"id": 1, "f0": 2, "f2": 4}])
        >>> matched, unmatched = heteroDF_split(het_df,["f0"])
        >>> print(pd.concat(matched,axis=0).to_markdown())
        >>> |          |   id |   f0 |
        ... |:---------|-----:|-----:|
        ... | ('A', 0) |    0 |    1 |
        ... | ('A', 1) |    1 |    3 |
        ... | ('B', 0) |    0 |    1 |
        ... | ('B', 1) |    1 |    2 |
        >>> print(pd.concat(unmatched,axis=0).to_markdown())
        >>> |          |   id |   f1 |   f2 |
        ... |:---------|-----:|-----:|-----:|
        ... | ('A', 0) |    0 |    2 |  nan |
        ... | ('A', 1) |    1 |    4 |  nan |
        ... | ('B', 0) |    0 |  nan |    2 |
        ... | ('B', 1) |    1 |  nan |    4 |
}
    """
    # turn splitby into form dict
    if isinstance(split_by, list):
        split_by = {key: split_by for key in hdf.keys()}

    matchedHet = {}
    unmatchedHet = {}
    # TODO add keep ID_cols flag

    for key, df in hdf.items():
        # compute cols to split by
        split_cols = split_by.get(key, list())
        # get common columns
        if isinstance(key, str):
            index_cols = ["id"]
        elif isinstance(key, tuple):
            index_cols = ["source", "target"]
        else:
            raise KeyError(
                f"the column type is {type(key)} is not supported please "
                f"use string or a tuple")
        # if no data is matched, keep everything in the unmatched
        only_index = list(df.columns) == index_cols
        if only_index:
            if keep_index_in_unmatched:
                unmatchedHet[key] = df
            elif keep_index_in_matched:
                matchedHet[key] = df
            continue

        # matched
        matched_df = df.filter(items=index_cols + split_cols, axis=1)
        if list(matched_df.columns) != index_cols or keep_index_in_matched:
            matchedHet[key] = matched_df

        # unmatched
        unmatched_df = df.drop(split_cols, axis=1, errors="ignore")
        if list(unmatched_df.columns) != index_cols or keep_index_in_unmatched:
            unmatchedHet[key] = unmatched_df

    return matchedHet, unmatchedHet


def _reset_index_map(df: pd.DataFrame) -> bidict:
    # reset index to maintain continuous index and get mapping to id
    newID_to_oldID = df.reset_index()["id"].to_dict()
    return bidict(newID_to_oldID)


def _edge_df_to_pygEdgeData(df: pd.DataFrame) -> Dict:
    r"""
    Converts an edgeDF into the format required by the HeteroData
    constructor Assumes "source" and "target" columns are available. returns
    dictionary with either only edge_index values or both edge_index and
    edge_attributes matrices

   """

    edge_index = df[["source", "target"]]
    edge_attributes = df.drop(["source", "target"], axis=1)
    if len(edge_attributes.columns) == 0:
        edge_attributes = None

    if edge_attributes is None:
        return {'edge_index': torch.from_numpy(edge_index.T.values)}
    else:
        return {
            'edge_index':
            torch.from_numpy(edge_index.T.values),
            'edge_attr':
            torch.from_numpy(edge_attributes.values).type(torch.FloatTensor)
        }


def _multi_map(df: pd.DataFrame, dict_map: Dict[str,
                                                Callable]) -> pd.DataFrame:
    r"""
    Apply multiple dict maps on multiple columns
    (a dict of value mapping on each column defined in dict_map).
    returns the df after applying the
    mapping
    Examples:
        >>> cat_df = pd.DataFrame([
        ... ["a", "a", "b", "c"],
        ... ["d","d","d","a"]]) c
        >>> at_df_map = {
        ... 1:{'a':2, 'd':4},
        ... 2:{'b':0, 'd':6},
        ... 3:{'c':3, 'a':1}}
        >>> print(multi_map(cat_df, cat_df_map).to_markdown())
        ... |    | 0   |   1 |   2 |   3 |
        ... |---:|:----|----:|----:|----:|
        ... |  0 | a   |   2 |   0 |   3 |
        ... |  1 | d   |   4 |   6 |   1 |
    """
    new_columns = {
        col: series.map(dict_map[col]) if col in dict_map else series
        for col, series in df.items()
    }
    return pd.DataFrame.from_dict(new_columns)


def heteroDf2heteroData(hdf: HeteroDF, ignore=None,
                        keep=None) -> Tuple[HeteroData, Dict]:
    r"""Converts a :class:`~torch_geometric.typing.HeteroDF` into a
    :class:`~torch_geometric.data.HeteroData`.

    Where :class:`~torch_geometric.typing.HeteroDF` is a dictionary with
    keys: :class:`~torch_geometric.typing.NodeType` (str) keys are str that
    defines node types values are pd.Dataframes with at least an "id" column
    and other feature columns. :class:`~torch_geometric.typing.EdgeType` (
    Tuple[str,str,str]) Keys are EdgeType of the form (sourceType,edgeType,
    targetType) values are pd.Dataframes with source and target columns that
    contain the node ids that define the edge and other feature columns.

    Args:
        hdf: :class:`~torch_geometric.typing.HeteroDF`. ignore: whether to
        ignore some features of the heteroDF. takes either a list of columns to
        ignore for each dataframe or a heteroColumnSet. keep: used only if
        ignore is None. whether to keep some features of the heteroDF. takes
        either a list of columns to keep for each dataframe or a
        :class:`~torch_geometric.typing.HeteroColumnSet`.

    Returns:
        (:class:`~torch_geometric.data.HeteroData`,:obj:`ContextObj`)
        Where the ContextObj is a dict with keys and values as:
        'index_map' : a map from node type to a bidict that
            maps node id form the HeteroDF to node index
            in the heteroData node features tensor.
        'ignored_features' : a heteroDF of all features that were ignored.
        'node_features' : a map from node type to list of
            feature names in each node type.
        'edge_attributes' : a map from edge type to list of feature names.

    Example:
        >>> het_df={
        ... 'A':pd.DataFrame.from_records([
        ...     {"id": 0, "feat1":1,"feat2":2,'name':'A0'},
        ...     {"id": 1, "feat1":11,"feat2":12,'name':'A1'}
        ... ]),
        ... 'B':pd.DataFrame.from_records([
        ...     {"id": "b0", "feat1":101,"feat2":102,'name':'B0'},
        ...     {"id": "b1", "feat1":111,"feat2":112,'name':'B1'}
        ... ]),
        ... ('B','to','A'):pd.DataFrame.from_records([
        ...     {"source": "b0","target":0,'capacity':10}
        ... ]),
        ... ('A','self_edge','A'):pd.DataFrame.from_records([
        ...     {"source": 0,"target":0}
        ... ]),}
        >>> het_data,context = heteroDf2heteroData(het_df,ignore=['name'])
        >>> het_data
        >>> HeteroData(
        ...     A={
        ...         x=[2, 2],
        ...         num_nodes=2
        ...     },
        ...     B={
        ...         x=[2, 2],
        ...         num_nodes=2
        ...     },
        ...     (B, to, A)={
        ...         edge_index=[2, 1],
        ...         edge_attr=[1, 1]
        ...     },
        ...     (A, self_edge, A)={ edge_index=[2, 1] }
        ...     )
        >>> context["index_map"]
        >>> {'A': bidict({0: 0, 1: 1}), 'B': bidict({0: 'b0', 1: 'b1'})}
        >>> print(pd.concat(context["ignored_features"]).to_markdown())
        >>> |          | id   | name   |
        ... |:---------|:-----|:-------|
        ... | ('A', 0) | 0    | A0     |
        ... | ('A', 1) | 1    | A1     |
        ... | ('B', 0) | b0   | B0     |
        ... | ('B', 1) | b1   | B1     |
        >>> context["node_features"]
        >>> {'A': ['feat1', 'feat2'], 'B': ['feat1', 'feat2']}
        >>> context["edges_attributes"]
        >>> {('B', 'to', 'A'): ['capacity']}
    """
    # split hdf to remove ignored features
    if ignore is None and keep is not None:
        hdf, ignored_features = heteroDf_split(hdf, split_by=keep,
                                               keep_index_in_matched=True)

    elif ignore is not None:
        ignored_features, hdf = heteroDf_split(hdf, split_by=ignore,
                                               keep_index_in_unmatched=True)
    else:
        ignored_features = None

    # separate hdfs to node and edge hdfs
    nodes_hdf = {key: df for key, df in hdf.items() if isinstance(key, str)}
    edges_hdf = {key: df for key, df in hdf.items() if isinstance(key, tuple)}

    # compute index mapping for each key
    # looks like {type:{int_index:old_index}}
    index_map = {k: _reset_index_map(v) for k, v in nodes_hdf.items()}
    # transform edge source and target to new according to index map
    new_edge_hdf = {}
    edge_attributes_dict = {}
    for key, df in edges_hdf.items():
        (source_type, edge_type, target_type) = key
        new_edge_hdf[key] = _multi_map(
            df, dict_map={
                "source": index_map[source_type].inverse,
                "target": index_map[target_type].inverse
            })
        edges_attributes = df.columns.drop(["source", "target"]).to_list()
        if edges_attributes:
            # meaning there are edge attributes (the list is not empty)
            edge_attributes_dict[key] = edges_attributes

    heterodata_init_dict = {}
    node_features_dict = {}
    # node df, cast to numpy
    for key, df in nodes_hdf.items():
        if list(df.columns) == ["id"]:
            df["values"] = 0.0  # add default value if there are no features
        heterodata_init_dict[key] = {
            'x':
            torch.from_numpy(df.drop(['id'],
                                     axis=1).values).type(torch.FloatTensor),
            'num_nodes':
            df.shape[0]
        }
        node_features_dict[key] = df.drop(['id'], axis=1).columns.tolist()
    # edge df, split to index array and attribute array
    for key, df in new_edge_hdf.items():
        heterodata_init_dict[key] = _edge_df_to_pygEdgeData(df)

    return (HeteroData(heterodata_init_dict), {
        'index_map': index_map,
        'ignored_features': ignored_features,
        'node_features': node_features_dict,
        'edges_attributes': edge_attributes_dict
    })


def heteroData2heteroDf(hetero_data: HeteroData, context: Dict) -> HeteroDF:
    """
    heteroData2heteroDf Converts a :class:`~torch_geometric.data.HeteroData`
        into a :class:`~torch_geometric.typing.HeteroDF`.

    Args:
        hetero_data (:class:`~torch_geometric.data.HeteroData`):
            hetero_data to convert
        context (Dict): a dict with keys and values as:
        'index_map' : a map from node type to a bidict that
            maps node id form the HeteroDF to node index
            in the heteroData node features tensor.
        'ignored_features' : a heteroDF of all features that were ignored.
        'node_features' : a map from node type to list of
            feature names in each node type.
        'edge_attributes' : a map from edge type to list of feature names

    Returns:
        (:class:`~torch_geometric.typing.HeteroDF`)

    Example:
        >>> data = HeteroData()
        ... data["A"].x = torch.tensor([[1,2],[11,12]])
        ... data["B"].x = torch.tensor([[101,102],[111,112]])
        ... data["B","to","A"].edge_index = torch.tensor([[0,1],[0,0]])
        ... data["B","to","A"].edge_attr = torch.tensor([[10],[100]])

        >>> context ={
        ... "index_map":{'A': bidict({0: 0, 1: 1}),
        ...              'B': bidict({0: 'b0', 1: 'b1'})},
        ... "ignored_features":{
        ...     "A":pd.DataFrame.from_records([
        ...         {"id": 0, 'name':'A0'},
        ...         {"id": 1, 'name':'A1'}
        ...     ]),
        ...     "B":pd.DataFrame.from_records([
        ...         {"id": "b0",'name':'B0'},
        ...         {"id": "b1",'name':'B1'}
        ...     ])},
        ... "node_features":{'A': ['f1', 'f2'],
        ...                  'B': ['f1', 'f2']},
        ... "edges_attributes": {('B','to','A'): ['attr1']}
        ... }
        >>> print(
        ... pd.concat(heteroData2heteroDf(data,context),axis=0).to_markdown())
        >>> |                      |id |f1 |f2 |name|source|target|attr1|
        ... |:---------------------|:--|--:|--:|:---|:-----|-----:|----:|
        ... | ('A', 0)             |0  |  1|  2|A0  |nan   |nan   | nan |
        ... | ('A', 1)             |1  | 11| 12|A1  |nan   |nan   | nan |
        ... | ('B', 0)             |b0 |101|102|B0  |nan   |nan   | nan |
        ... | ('B', 1)             |b1 |111|112|B1  |nan   |nan   | nan |
        ... | (('B', 'to', 'A'), 0)|nan|nan|nan|nan |b0    |0     |  10 |
        ... | (('B', 'to', 'A'), 1)|nan|nan|nan|nan |b1    |0     |  00 |
    """
    hetero_df = {}
    for node_key, node_data in hetero_data.node_items():
        nodes_df = pd.DataFrame(node_data.x.numpy(),
                                columns=context["node_features"][node_key])
        nodes_df.insert(0, "id",
                        nodes_df.index.map(context["index_map"][node_key]))
        if context.get("ignored_features", None) is not None:
            hetero_df[node_key] = pd.merge(
                nodes_df, context["ignored_features"][node_key], on="id")

    for edge_key, edge_data in hetero_data.edge_items():
        if not bool(edge_data):
            continue  # we can receive an empty dict for some reason
        edges_df = pd.DataFrame(edge_data.edge_index.T.numpy(),
                                columns=['source', 'target'])
        src_node_name, _, tgt_node_name = edge_key
        edges_df["source"] = edges_df["source"].map(
            context["index_map"][src_node_name])
        edges_df["target"] = edges_df["target"].map(
            context["index_map"][tgt_node_name])

        if edge_data.num_edge_features > 0:
            edges_df = edges_df.join(
                pd.DataFrame(edge_data.edge_attr.numpy(),
                             columns=context["edges_attributes"][edge_key]))

        hetero_df[edge_key] = edges_df
    return hetero_df
