import os.path as osp
from typing import Any, Dict

import torch

from torch_geometric.data import Data, OnDiskDataset
from torch_geometric.testing import withPackage


@withPackage('sqlite3')
def test_pickle(tmp_path):
    dataset = OnDiskDataset(tmp_path)
    assert len(dataset) == 0
    assert str(dataset) == 'OnDiskDataset(0)'
    assert osp.exists(osp.join(tmp_path, 'processed', 'sqlite.db'))

    data_list = [
        Data(
            x=torch.randn(5, 8),
            edge_index=torch.randint(0, 5, (2, 16)),
            num_nodes=5,
        ) for _ in range(4)
    ]

    dataset.append(data_list[0])
    assert len(dataset) == 1

    dataset.extend(data_list[1:])
    assert len(dataset) == 4

    out = dataset.get(0)
    assert torch.equal(out.x, data_list[0].x)
    assert torch.equal(out.edge_index, data_list[0].edge_index)
    assert out.num_nodes == data_list[0].num_nodes

    out_list = dataset.multi_get([1, 2, 3])
    for out, data in zip(out_list, data_list[1:]):
        assert torch.equal(out.x, data.x)
        assert torch.equal(out.edge_index, data.edge_index)
        assert out.num_nodes == data.num_nodes

    dataset.close()

    # Test persistence of datasets:
    dataset = OnDiskDataset(tmp_path)
    assert len(dataset) == 4

    out = dataset.get(0)
    assert torch.equal(out.x, data_list[0].x)
    assert torch.equal(out.edge_index, data_list[0].edge_index)
    assert out.num_nodes == data_list[0].num_nodes

    dataset.close()


@withPackage('sqlite3')
def test_custom_schema(tmp_path):
    class CustomSchemaOnDiskDataset(OnDiskDataset):
        def __init__(self, root: str):
            schema = {
                'x': dict(dtype=torch.float, size=(-1, 8)),
                'edge_index': dict(dtype=torch.long, size=(2, -1)),
                'num_nodes': int,
            }
            self.serialize_count = 0
            self.deserialize_count = 0
            super().__init__(root, schema=schema)

        def serialize(self, data: Data) -> Dict[str, Any]:
            self.serialize_count += 1
            return data.to_dict()

        def deserialize(self, mapping: Dict[str, Any]) -> Any:
            self.deserialize_count += 1
            return Data.from_dict(mapping)

    dataset = CustomSchemaOnDiskDataset(tmp_path)
    assert len(dataset) == 0
    assert str(dataset) == 'CustomSchemaOnDiskDataset(0)'
    assert osp.exists(osp.join(tmp_path, 'processed', 'sqlite.db'))

    data_list = [
        Data(
            x=torch.randn(5, 8),
            edge_index=torch.randint(0, 5, (2, 16)),
            num_nodes=5,
        ) for _ in range(4)
    ]

    dataset.append(data_list[0])
    assert dataset.serialize_count == 1
    assert len(dataset) == 1

    dataset.extend(data_list[1:])
    assert dataset.serialize_count == 4
    assert len(dataset) == 4

    out = dataset.get(0)
    assert dataset.deserialize_count == 1
    assert torch.equal(out.x, data_list[0].x)
    assert torch.equal(out.edge_index, data_list[0].edge_index)
    assert out.num_nodes == data_list[0].num_nodes

    out_list = dataset.multi_get([1, 2, 3])
    assert dataset.deserialize_count == 4
    for out, data in zip(out_list, data_list[1:]):
        assert torch.equal(out.x, data.x)
        assert torch.equal(out.edge_index, data.edge_index)
        assert out.num_nodes == data.num_nodes

    dataset.close()


@withPackage('sqlite3')
def test_index_select_get(tmp_path):
    dataset = OnDiskDataset(tmp_path)
    data_list = [Data(x=torch.tensor([i])) for i in range(10)]
    dataset.extend(data_list)

    subset = dataset.index_select([5, 6, 7, 8, 9])
    nested_subset = subset.index_select([1, 3])

    assert torch.equal(subset[0].x, data_list[5].x)
    assert torch.equal(subset.get(0).x, data_list[5].x)

    out_list = subset.multi_get([0, 2, 4])
    for out, data in zip(out_list, data_list[5:10:2]):
        assert torch.equal(out.x, data.x)

    out_list = nested_subset.__getitems__([0, 1])
    assert torch.equal(out_list[0].x, data_list[6].x)
    assert torch.equal(out_list[1].x, data_list[8].x)

    dataset.close()
