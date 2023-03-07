from typing import Set

import networkx as nx
import pandas as pd
import torch

from torch_geometric.utils.hetero_df import (
    _edge_df_to_pygEdgeData,
    _groupby_to_dict,
    _multi_map,
    _node_edge_dfs_to_sig_df,
    heteroDf2heteroData,
    heteroDf_merge,
    heteroDf_split,
    nx2heteroDf,
)


def _to_tuple_set(df: pd.DataFrame) -> Set:
    records = list(df.to_records(index=False))
    return {tuple(rec) for rec in records}


student_data = pd.DataFrame.from_records([
    {
        "id": 1,
        "gpa": 80,
        "year": 4
    },
    {
        "id": 2,
        "gpa": 85,
        "year": 3
    },
])
ta_data = pd.DataFrame.from_records([
    {
        "id": 3,
        "teaching_score": 5
    },
])
lec_data = pd.DataFrame.from_records([
    {
        "id": 4,
        "teaching_score": 4
    },
])
course_data = pd.DataFrame.from_records([
    {
        "id": 5,
        "times_taught": 3
    },
    {
        "id": 6,
        "times_taught": 5
    },
    {
        "id": 7,
        "times_taught": 2
    },
])
participate_in_data = pd.DataFrame.from_records([
    {
        "source": 1,
        "target": 5,
        "score": 65
    },
    {
        "source": 1,
        "target": 7,
        "score": 80
    },
    {
        "source": 2,
        "target": 6,
        "score": 70
    },
])

college_data_dict = {
    "student": student_data,
    "ta": ta_data,
    "lecturer": lec_data,
    "class": course_data,
    ("student", "takes", "class"): participate_in_data
}
college_nx = nx.MultiDiGraph()
college_nx.add_nodes_from([
    (1, {
        "id": 1,
        "name": "bob",
        "node_type": "student",
    }),
    (2, {
        "id": 2,
        "name": "dan",
        "node_type": "student",
    }),
    (3, {
        "id": 3,
        "name": "tal",
        "node_type": "ta",
    }),
    (4, {
        "id": 4,
        "name": "profG",
        "node_type": "lecturer",
    }),
    (5, {
        "id": 5,
        "name": "phys101",
        "node_type": "class",
    }),
    (6, {
        "id": 6,
        "name": "phys102",
        "node_type": "class",
    }),
    (7, {
        "id": 7,
        "name": "chem101",
        "node_type": "class",
    }),
    (8, {
        "id": 8,
        "name": "physics",
        "node_type": "topic"
    }),
    (9, {
        "id": 9,
        "name": "chemistry",
        "node_type": "topic"
    }),
])
college_nx.add_edges_from([
    (1, 5, {
        "edge_type": "takes"
    }),
    (1, 7, {
        "edge_type": "takes"
    }),
    (2, 6, {
        "edge_type": "takes"
    }),
    (3, 5, {
        "edge_type": "teaches"
    }),
    (4, 6, {
        "edge_type": "teaches"
    }),
    (5, 8, {
        "edge_type": "is_about"
    }),
    (6, 8, {
        "edge_type": "is_about"
    }),
    (6, 9, {
        "edge_type": "is_about"
    }),
    (7, 9, {
        "edge_type": "is_about"
    }),
])

# %% tests


def test_groupby_to_dict():
    df = pd.DataFrame({
        "student": ["Gal", "Noa", "Song", "Gal", "Gal", "Song", "Noa", "Noa"],
        "grades": [100, 90, 90, 50, 50, 60, 60, 100],
        "class": [
            "history", "history", "history", "math", "art", "history", "math",
            "art"
        ]
    })

    dict_of_dfs = _groupby_to_dict(df, "student")
    assert list(dict_of_dfs.keys()) == ["Gal", "Noa", "Song"]
    assert list(dict_of_dfs["Gal"].columns) == ["grades", "class"]
    assert list(dict_of_dfs["Gal"]['grades']) == [100, 50, 50]
    assert list(dict_of_dfs["Noa"]['class']) == ["history", "math", "art"]

    dict_of_dfs = _groupby_to_dict(df, "grades")
    assert list(dict_of_dfs.keys()) == [50, 60, 90, 100]
    assert list(dict_of_dfs[50].columns) == ["student", "class"]
    assert list(dict_of_dfs[100]["student"]) == ["Gal", "Noa"]
    assert list(dict_of_dfs[90]["class"]) == ["history", "history"]

    dict_of_dfs = _groupby_to_dict(df, ["grades", "class"])
    assert list(dict_of_dfs.keys()) == [(50, 'art'), (50, 'math'),
                                        (60, 'history'), (60, 'math'),
                                        (90, 'history'), (100, 'art'),
                                        (100, 'history')]
    assert list(dict_of_dfs[(50, 'art')]["student"]) == ["Gal"]


def test_node_edge_dfs_to_sig_df():
    nodes = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "node_type": ["a", "a", "a", "b", "b"]
    })
    edges = pd.DataFrame({
        "source": [1, 2, 3, 5],
        "target": [2, 4, 5, 1],
        "edge_type": ["c", "c", "c", "c"]
    })

    df = _node_edge_dfs_to_sig_df(nodes, edges)
    assert list(df["source"]) == [1, 2, 3, 5]
    assert list(df["target"]) == [2, 4, 5, 1]
    assert list(df["source_type"]) == ["a", "a", "a", "b"]
    assert list(df["edge_type"]) == ["c", "c", "c", "c"]
    assert list(df["target_type"]) == ["a", "b", "b", "a"]


def test_nx2heteroDf():
    nxHeteroDF = nx2heteroDf(college_nx)

    recordsDF = {k: _to_tuple_set(v) for k, v in nxHeteroDF.items()}
    assert recordsDF == {
        "class": {(5, "phys101"), (6, "phys102"), (7, "chem101")},
        "lecturer": {(4, "profG")},
        "student": {(1, "bob"), (2, "dan")},
        "ta": {(3, "tal")},
        "topic": {(8, "physics"), (9, "chemistry")},
        ("class", "is_about", "topic"): {(5, 8), (6, 8), (6, 9), (7, 9)},
        ("lecturer", "teaches", "class"): {(4, 6)},
        ("student", "takes", "class"): {(1, 5), (1, 7), (2, 6)},
        ("ta", "teaches", "class"): {(3, 5)},
    }


def test_heteroDf_merge():
    nxHeteroDF = nx2heteroDf(college_nx)
    mergedHeteroDF = heteroDf_merge(nxHeteroDF, college_data_dict)

    recordsDF = {k: _to_tuple_set(v) for k, v in mergedHeteroDF.items()}
    assert recordsDF == {
        "student": {(1, "bob", 80, 4), (2, "dan", 85, 3)},
        "ta": {(3, "tal", 5)},
        "lecturer": {(4, "profG", 4)},
        "class": {(5, "phys101", 3), (6, "phys102", 5), (7, "chem101", 2)},
        "topic": {(8, "physics"), (9, "chemistry")},
        ("class", "is_about", "topic"): {(5, 8), (6, 8), (6, 9), (7, 9)},
        ("lecturer", "teaches", "class"): {(4, 6)},
        ("student", "takes", "class"): {(1, 5, 65), (1, 7, 80), (2, 6, 70)},
        ("ta", "teaches", "class"): {(3, 5)},
    }


def test_heteroDf_split():
    nxHeteroDF = nx2heteroDf(college_nx)
    mergedHeteroDF = heteroDf_merge(nxHeteroDF, college_data_dict)
    matchedHetDF, unmatchedHetDF = heteroDf_split(mergedHeteroDF,
                                                  split_by=["name"],
                                                  keep_index_in_unmatched=True)
    matchedRecords = {k: _to_tuple_set(v) for k, v in matchedHetDF.items()}
    assert matchedRecords == {
        "student": {(1, "bob"), (2, "dan")},
        "ta": {(3, "tal")},
        "lecturer": {(4, "profG")},
        "class": {(5, "phys101"), (6, "phys102"), (7, "chem101")},
        "topic": {(8, "physics"), (9, "chemistry")},
    }
    unmatchedRecords = {k: _to_tuple_set(v) for k, v in unmatchedHetDF.items()}
    assert unmatchedRecords == {
        "student": {(1, 80, 4), (2, 85, 3)},
        "ta": {(3, 5)},
        "topic": {(8, ), (9, )},
        "lecturer": {(4, 4)},
        "class": {(5, 3), (6, 5), (7, 2)},
        ("class", "is_about", "topic"): {(5, 8), (6, 8), (6, 9), (7, 9)},
        ("lecturer", "teaches", "class"): {(4, 6)},
        ("student", "takes", "class"): {(1, 5, 65), (1, 7, 80), (2, 6, 70)},
        ("ta", "teaches", "class"): {(3, 5)},
    }
    # Split different attributes for different keys

    matchedHetDF, unmatchedHetDF = heteroDf_split(
        mergedHeteroDF, split_by={
            'class': ['name'],
            ('student', 'takes', 'class'): ['score']
        }, keep_index_in_unmatched=True)
    matchedRecords = {k: _to_tuple_set(v) for k, v in matchedHetDF.items()}
    assert matchedRecords == {
        "class": {(5, "phys101"), (6, "phys102"), (7, "chem101")},
        ("student", "takes", "class"): {(1, 5, 65), (1, 7, 80), (2, 6, 70)},
    }
    unmatchedRecords = {k: _to_tuple_set(v) for k, v in unmatchedHetDF.items()}
    assert unmatchedRecords == {
        "student": {(1, "bob", 80, 4), (2, "dan", 85, 3)},
        "ta": {(3, "tal", 5)},
        "lecturer": {(4, "profG", 4)},
        "class": {(5, 3), (6, 5), (7, 2)},
        "topic": {(8, "physics"), (9, "chemistry")},
        ("student", "takes", "class"): {(1, 5), (1, 7), (2, 6)},
        ("class", "is_about", "topic"): {(5, 8), (6, 8), (6, 9), (7, 9)},
        ("lecturer", "teaches", "class"): {(4, 6)},
        ("ta", "teaches", "class"): {(3, 5)},
    }
    # Split different attributes but make sure to keep ids and source target
    # pairs in unmatched, even if all other fields matched. This is useful
    # if there is no feature data remaining in the unmatched heteroDF but we
    # still want to keep stuff like featureless edge data.
    matchedHetDF, unmatchedHetDF = heteroDf_split(
        mergedHeteroDF, split_by={
            'class': ['name'],
            ('student', 'takes', 'class'): ['score'],
            'topic': ['name']
        }, keep_index_in_unmatched=True)
    matchedRecords = {k: _to_tuple_set(v) for k, v in matchedHetDF.items()}
    assert matchedRecords == {
        "class": {(5, "phys101"), (6, "phys102"), (7, "chem101")},
        ("student", "takes", "class"): {(1, 5, 65), (1, 7, 80), (2, 6, 70)},
        "topic": {(8, "physics"), (9, "chemistry")},
    }
    unmatchedRecords = {k: _to_tuple_set(v) for k, v in unmatchedHetDF.items()}
    assert unmatchedRecords == {
        "student": {(1, "bob", 80, 4), (2, "dan", 85, 3)},
        "ta": {(3, "tal", 5)},
        "lecturer": {(4, "profG", 4)},
        "class": {(5, 3), (6, 5), (7, 2)},
        "topic": {(8, ), (9, )},
        ("class", "is_about", "topic"): {(5, 8), (6, 8), (6, 9), (7, 9)},
        ("lecturer", "teaches", "class"): {(4, 6)},
        ("ta", "teaches", "class"): {(3, 5)},
        # note that we still have empty edge data when all fields
        # were moved to the matched part
        ("student", "takes", "class"): {(1, 5), (1, 7), (2, 6)},
    }
    matchedHetDF, unmatchedHetDF = heteroDf_split(
        mergedHeteroDF, split_by={
            'class': ['name'],
            ('student', 'takes', 'class'): ['score'],
            'topic': ['name']
        })

    matchedRecords = {k: _to_tuple_set(v) for k, v in matchedHetDF.items()}
    assert matchedRecords == {
        "class": {(5, "phys101"), (6, "phys102"), (7, "chem101")},
        ("student", "takes", "class"): {(1, 5, 65), (1, 7, 80), (2, 6, 70)},
        "topic": {(8, "physics"), (9, "chemistry")},
    }
    unmatchedRecords = {k: _to_tuple_set(v) for k, v in unmatchedHetDF.items()}
    assert unmatchedRecords == {
        "student": {(1, "bob", 80, 4), (2, "dan", 85, 3)},
        "ta": {(3, "tal", 5)},
        "lecturer": {(4, "profG", 4)},
        "class": {(5, 3), (6, 5), (7, 2)},
    }

    matchedHetDF, unmatchedHetDF = heteroDf_split(
        mergedHeteroDF, split_by={
            'class': ['name'],
            ('student', 'takes', 'class'): ['score'],
            'topic': ['name']
        }, keep_index_in_matched=True)

    matchedRecords = {k: _to_tuple_set(v) for k, v in matchedHetDF.items()}
    assert matchedRecords, {
        "class": {(5, "phys101"), (6, "phys102"), (7, "chem101")},
        ("student", "takes", "class"): {(1, 5, 65), (1, 7, 80), (2, 6, 70)},
        "topic": {(8, "physics"), (9, "chemistry")},
        ("class", "is_about", "topic"): {(5, 8), (6, 8), (6, 9), (7, 9)},
        ("lecturer", "teaches", "class"): {(4, 6)},
        ("ta", "teaches", "class"): {(3, 5)},
        "student": {(1, ), (2, )},
        "ta": {(3, )},
        "lecturer": {(4, )},
    }
    unmatchedRecords = {k: _to_tuple_set(v) for k, v in unmatchedHetDF.items()}
    assert unmatchedRecords == {
        "student": {(1, "bob", 80, 4), (2, "dan", 85, 3)},
        "ta": {(3, "tal", 5)},
        "lecturer": {(4, "profG", 4)},
        "class": {(5, 3), (6, 5), (7, 2)},
    }


def test_edge_df_to_pygEdgeData():
    edges = pd.DataFrame({
        "source": [1, 2, 3, 5],
        "target": [2, 4, 5, 1],
        "edge_data": [0.1, 0.2, 0.2, 0.5]
    })
    edge_dict = _edge_df_to_pygEdgeData(edges)
    assert list(edge_dict.keys()) == ['edge_index', 'edge_attr']
    assert edge_dict["edge_index"].equal(
        torch.as_tensor([[1, 2, 3, 5], [2, 4, 5, 1]]))
    assert edge_dict["edge_attr"].equal(
        torch.as_tensor([[0.1000], [0.2000], [0.2000], [0.5000]]))

    edges = pd.DataFrame({"source": [1, 2, 3, 5], "target": [2, 4, 5, 1]})
    edge_dict = _edge_df_to_pygEdgeData(edges)
    assert list(edge_dict.keys()) == ['edge_index']
    assert edge_dict["edge_index"].equal(
        torch.as_tensor([[1, 2, 3, 5], [2, 4, 5, 1]]))


def test_multi_map():
    cat_df = pd.DataFrame([["a", "a", "b", "c"], ["d", "d", "d", "a"]])
    cat_df_map = {
        1: {
            'a': 2,
            'd': 4
        },
        2: {
            'b': 0,
            'd': 6
        },
        3: {
            'c': 3,
            'a': 1
        }
    }
    mm = _multi_map(cat_df, cat_df_map)
    assert list(mm[0]) == ['a', 'd']
    assert list(mm[1]) == [2, 4]
    assert list(mm[2]) == [0, 6]
    assert list(mm[3]) == [3, 1]


def test_heteroDf2heteroData():
    tiny_heteroDF = {
        'A':
        pd.DataFrame.from_records([{
            "id": 0,
            "feat1": 1,
            "feat2": 2,
            'name': 'A0'
        }, {
            "id": 1,
            "feat1": 11,
            "feat2": 12,
            'name': 'A1'
        }]),
        'B':
        pd.DataFrame.from_records([{
            "id": "b0",
            "feat1": 101,
            "feat2": 102,
            'name': 'B0'
        }, {
            "id": "b1",
            "feat1": 111,
            "feat2": 112,
            'name': 'B1'
        }]),
        ('B', 'to', 'A'):
        pd.DataFrame.from_records([{
            "source": "b0",
            "target": 0,
            'capacity': 10
        }]),
        ('A', 'self_edge', 'A'):
        pd.DataFrame.from_records([{
            "source": 0,
            "target": 0
        }]),
    }

    tinyData, tinyContext = heteroDf2heteroData(tiny_heteroDF, ignore=['name'])

    assert tinyData["A"].num_nodes == 2
    assert tinyData["A"].x.equal(
        torch.as_tensor([[1, 2], [11, 12]]).to(torch.float))
    assert tinyData["B"].x.equal(
        torch.as_tensor([[101., 102.], [111., 112.]]).to(torch.float))
    assert tinyData[("B", "to",
                     "A")].edge_index.equal(torch.as_tensor([[0], [0]]))
    assert tinyData[("B", "to", "A")].edge_attr.equal(
        torch.as_tensor([[10]]).to(torch.float))
    assert tinyData[("A", "self_edge",
                     "A")].edge_index.equal(torch.as_tensor([[0], [0]]))

    assert tinyContext['index_map']["A"].keys() == {0, 1}
    assert tinyContext['index_map']["A"].values() == {0, 1}
    assert tinyContext['index_map']["B"].keys() == {0, 1}
    assert tinyContext['index_map']["B"].values() == {"b0", "b1"}

    assert list(tinyContext['ignored_features']["A"].keys()) == ["id", "name"]
    assert list(tinyContext['ignored_features']["A"]["id"]) == [0, 1]
    assert list(tinyContext['ignored_features']["A"]["name"]) == ["A0", "A1"]
    assert list(tinyContext['ignored_features']["B"]["id"]) == ["b0", "b1"]

    assert list(tinyContext['node_features']["A"]) == ["feat1", "feat2"]
    assert list(tinyContext['node_features']["B"]) == ["feat1", "feat2"]
