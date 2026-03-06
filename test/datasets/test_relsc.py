import json
import os
from unittest.mock import patch

import pandas as pd

from torch_geometric.data import Data, HeteroData
from torch_geometric.datasets import RelSCH, RelSCM


def create_dummy_data(raw_dir, project_name):
    os.makedirs(raw_dir, exist_ok=True)

    keys = [f"dummy_file_{i}.java" for i in range(4)]

    dummy_json = {}
    for key in keys:
        dummy_json[key] = [
            [
                [[6], [3]],  # node_features
                [[0, 1], [1, 0]],  # adj_list
                [[1], [2]]  # edge_features base
            ],
            2  # num_nodes
        ]

    dummy_json[keys[0]] = [
        [
            [[6], [3]],
            [[0, 1]] * 11,  # 11 collegamenti fittizi
            [[i] for i in range(1, 12)]  # 11 feature di archi distinte
        ],
        2
    ]

    json_path = os.path.join(raw_dir, f"{project_name}.json")
    with open(json_path, 'w') as f:
        json.dump(dummy_json, f)

    # Crea un finto file CSV con 4 valori per i nostri 4 grafi
    dummy_csv = pd.DataFrame({'Key': keys, 'Value': [10.0, 20.0, 30.0, 40.0]})
    csv_path = os.path.join(raw_dir, f"y_{project_name}.csv")
    dummy_csv.to_csv(csv_path, index=False)


@patch('torch_geometric.datasets.RelSCH.download')
def test_relsc_h(mock_download, tmp_path):
    project_name = 'rdf'
    raw_dir = os.path.join(tmp_path, 'RelSC', 'raw')

    create_dummy_data(raw_dir, project_name)

    dataset = RelSCH(root=str(tmp_path), project_name=project_name)

    assert len(dataset) == 4
    assert dataset.project_name == 'rdf'

    data = dataset[0]
    assert isinstance(data, Data)
    assert data.num_nodes == 2
    assert 'x' in data
    assert 'edge_index' in data
    assert 'y' in data


@patch('torch_geometric.datasets.RelSCM.download')
def test_relsc_m(mock_download, tmp_path):
    project_name = 'dubbo'
    raw_dir = os.path.join(tmp_path, 'RelSC', 'raw')

    create_dummy_data(raw_dir, project_name)

    dataset = RelSCM(root=str(tmp_path), project_name=project_name)

    assert len(dataset) == 4
    assert dataset.project_name == 'dubbo'

    data = dataset[0]
    assert isinstance(data, HeteroData)
    assert 'y' in data

    # Verifica che le chiavi semantiche siano state generate
    assert 'declarations' in data.node_types
    assert 'control_flow' in data.node_types
