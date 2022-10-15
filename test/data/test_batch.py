import os
import random
import sys

import numpy as np
import torch
from torch_sparse import SparseTensor

import torch_geometric
from torch_geometric.data import Batch, Data, HeteroData


def get_edge_index(num_src_nodes, num_dst_nodes, num_edges):
    row = torch.randint(num_src_nodes, (num_edges, ), dtype=torch.long)
    col = torch.randint(num_dst_nodes, (num_edges, ), dtype=torch.long)
    return torch.stack([row, col], dim=0)


def test_batch():
    torch_geometric.set_debug(True)

    x1 = torch.tensor([1, 2, 3], dtype=torch.float)
    y1 = 1
    x1_sp = SparseTensor.from_dense(x1.view(-1, 1))
    e1 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    adj1 = SparseTensor.from_edge_index(e1)
    s1 = '1'
    array1 = ['1', '2']
    x2 = torch.tensor([1, 2], dtype=torch.float)
    y2 = 2
    x2_sp = SparseTensor.from_dense(x2.view(-1, 1))
    e2 = torch.tensor([[0, 1], [1, 0]])
    adj2 = SparseTensor.from_edge_index(e2)
    s2 = '2'
    array2 = ['3', '4', '5']
    x3 = torch.tensor([1, 2, 3, 4], dtype=torch.float)
    y3 = 3
    x3_sp = SparseTensor.from_dense(x3.view(-1, 1))
    e3 = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    adj3 = SparseTensor.from_edge_index(e3)
    s3 = '3'
    array3 = ['6', '7', '8', '9']

    data1 = Data(x=x1, y=y1, x_sp=x1_sp, edge_index=e1, adj=adj1, s=s1,
                 array=array1, num_nodes=3)
    data2 = Data(x=x2, y=y2, x_sp=x2_sp, edge_index=e2, adj=adj2, s=s2,
                 array=array2, num_nodes=2)
    data3 = Data(x=x3, y=y3, x_sp=x3_sp, edge_index=e3, adj=adj3, s=s3,
                 array=array3, num_nodes=4)

    batch = Batch.from_data_list([data1])
    assert str(batch) == ('DataBatch(x=[3], edge_index=[2, 4], y=[1], '
                          'x_sp=[3, 1, nnz=3], adj=[3, 3, nnz=4], s=[1], '
                          'array=[1], num_nodes=3, batch=[3], ptr=[2])')
    assert batch.num_graphs == len(batch) == 1
    assert batch.x.tolist() == [1, 2, 3]
    assert batch.y.tolist() == [1]
    assert batch.x_sp.to_dense().view(-1).tolist() == batch.x.tolist()
    assert batch.edge_index.tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
    edge_index = torch.stack(batch.adj.coo()[:2], dim=0)
    assert edge_index.tolist() == batch.edge_index.tolist()
    assert batch.s == ['1']
    assert batch.array == [['1', '2']]
    assert batch.num_nodes == 3
    assert batch.batch.tolist() == [0, 0, 0]
    assert batch.ptr.tolist() == [0, 3]

    batch = Batch.from_data_list([data1, data2, data3], follow_batch=['s'])

    assert str(batch) == ('DataBatch(x=[9], edge_index=[2, 12], y=[3], '
                          'x_sp=[9, 1, nnz=9], adj=[9, 9, nnz=12], s=[3], '
                          's_batch=[3], s_ptr=[4], array=[3], num_nodes=9, '
                          'batch=[9], ptr=[4])')
    assert batch.num_graphs == len(batch) == 3
    assert batch.x.tolist() == [1, 2, 3, 1, 2, 1, 2, 3, 4]
    assert batch.y.tolist() == [1, 2, 3]
    assert batch.x_sp.to_dense().view(-1).tolist() == batch.x.tolist()
    assert batch.edge_index.tolist() == [[0, 1, 1, 2, 3, 4, 5, 6, 6, 7, 7, 8],
                                         [1, 0, 2, 1, 4, 3, 6, 5, 7, 6, 8, 7]]
    edge_index = torch.stack(batch.adj.coo()[:2], dim=0)
    assert edge_index.tolist() == batch.edge_index.tolist()
    assert batch.s == ['1', '2', '3']
    assert batch.s_batch.tolist() == [0, 1, 2]
    assert batch.s_ptr.tolist() == [0, 1, 2, 3]
    assert batch.array == [['1', '2'], ['3', '4', '5'], ['6', '7', '8', '9']]
    assert batch.num_nodes == 9
    assert batch.batch.tolist() == [0, 0, 0, 1, 1, 2, 2, 2, 2]
    assert batch.ptr.tolist() == [0, 3, 5, 9]

    data = batch[0]
    assert str(data) == ("Data(x=[3], edge_index=[2, 4], y=[1], "
                         "x_sp=[3, 1, nnz=3], adj=[3, 3, nnz=4], s='1', "
                         "array=[2], num_nodes=3)")
    data = batch[1]
    assert str(data) == ("Data(x=[2], edge_index=[2, 2], y=[1], "
                         "x_sp=[2, 1, nnz=2], adj=[2, 2, nnz=2], s='2', "
                         "array=[3], num_nodes=2)")

    data = batch[2]
    assert str(data) == ("Data(x=[4], edge_index=[2, 6], y=[1], "
                         "x_sp=[4, 1, nnz=4], adj=[4, 4, nnz=6], s='3', "
                         "array=[4], num_nodes=4)")

    assert len(batch.index_select([1, 0])) == 2
    assert len(batch.index_select(torch.tensor([1, 0]))) == 2
    assert len(batch.index_select(torch.tensor([True, False]))) == 1
    assert len(batch.index_select(np.array([1, 0], dtype=np.int64))) == 2
    assert len(batch.index_select(np.array([True, False]))) == 1
    assert len(batch[:2]) == 2

    data_list = batch.to_data_list()
    assert len(data_list) == 3

    assert len(data_list[0]) == 8
    assert data_list[0].x.tolist() == [1, 2, 3]
    assert data_list[0].y.tolist() == [1]
    assert data_list[0].x_sp.to_dense().view(-1).tolist() == [1, 2, 3]
    assert data_list[0].edge_index.tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
    edge_index = torch.stack(data_list[0].adj.coo()[:2], dim=0)
    assert edge_index.tolist() == data_list[0].edge_index.tolist()
    assert data_list[0].s == '1'
    assert data_list[0].array == ['1', '2']
    assert data_list[0].num_nodes == 3

    assert len(data_list[1]) == 8
    assert data_list[1].x.tolist() == [1, 2]
    assert data_list[1].y.tolist() == [2]
    assert data_list[1].x_sp.to_dense().view(-1).tolist() == [1, 2]
    assert data_list[1].edge_index.tolist() == [[0, 1], [1, 0]]
    edge_index = torch.stack(data_list[1].adj.coo()[:2], dim=0)
    assert edge_index.tolist() == data_list[1].edge_index.tolist()
    assert data_list[1].s == '2'
    assert data_list[1].array == ['3', '4', '5']
    assert data_list[1].num_nodes == 2

    assert len(data_list[2]) == 8
    assert data_list[2].x.tolist() == [1, 2, 3, 4]
    assert data_list[2].y.tolist() == [3]
    assert data_list[2].x_sp.to_dense().view(-1).tolist() == [1, 2, 3, 4]
    assert data_list[2].edge_index.tolist() == [[0, 1, 1, 2, 2, 3],
                                                [1, 0, 2, 1, 3, 2]]
    edge_index = torch.stack(data_list[2].adj.coo()[:2], dim=0)
    assert edge_index.tolist() == data_list[2].edge_index.tolist()
    assert data_list[2].s == '3'
    assert data_list[2].array == ['6', '7', '8', '9']
    assert data_list[2].num_nodes == 4

    torch_geometric.set_debug(True)


def test_batching_with_new_dimension():
    torch_geometric.set_debug(True)

    class MyData(Data):
        def __cat_dim__(self, key, value, *args, **kwargs):
            if key == 'foo':
                return None
            else:
                return super().__cat_dim__(key, value, *args, **kwargs)

    x1 = torch.tensor([1, 2, 3], dtype=torch.float)
    foo1 = torch.randn(4)
    y1 = torch.tensor(1)

    x2 = torch.tensor([1, 2], dtype=torch.float)
    foo2 = torch.randn(4)
    y2 = torch.tensor(2)

    batch = Batch.from_data_list(
        [MyData(x=x1, foo=foo1, y=y1),
         MyData(x=x2, foo=foo2, y=y2)])

    assert str(batch) == ('MyDataBatch(x=[5], y=[2], foo=[2, 4], batch=[5], '
                          'ptr=[3])')
    assert batch.num_graphs == len(batch) == 2
    assert batch.x.tolist() == [1, 2, 3, 1, 2]
    assert batch.foo.size() == (2, 4)
    assert batch.foo[0].tolist() == foo1.tolist()
    assert batch.foo[1].tolist() == foo2.tolist()
    assert batch.y.tolist() == [1, 2]
    assert batch.batch.tolist() == [0, 0, 0, 1, 1]
    assert batch.ptr.tolist() == [0, 3, 5]
    assert batch.num_graphs == 2

    data = batch[0]
    assert str(data) == ('MyData(x=[3], y=[1], foo=[4])')
    data = batch[1]
    assert str(data) == ('MyData(x=[2], y=[1], foo=[4])')

    torch_geometric.set_debug(True)


def test_pickling():
    data = Data(x=torch.randn(5, 16))
    batch = Batch.from_data_list([data, data, data, data])
    assert id(batch._store._parent()) == id(batch)
    assert batch.num_nodes == 20

    path = os.path.join(os.sep, 'tmp', f'{random.randrange(sys.maxsize)}.pt')
    torch.save(batch, path)
    assert id(batch._store._parent()) == id(batch)
    assert batch.num_nodes == 20

    batch = torch.load(path)
    assert id(batch._store._parent()) == id(batch)
    assert batch.num_nodes == 20

    assert batch.__class__.__name__ == 'DataBatch'
    assert batch.num_graphs == len(batch) == 4

    os.remove(path)


def test_recursive_batch():
    data1 = Data(
        x={
            '1': torch.randn(10, 32),
            '2': torch.randn(20, 48)
        }, edge_index=[get_edge_index(30, 30, 50),
                       get_edge_index(30, 30, 70)], num_nodes=30)

    data2 = Data(
        x={
            '1': torch.randn(20, 32),
            '2': torch.randn(40, 48)
        }, edge_index=[get_edge_index(60, 60, 80),
                       get_edge_index(60, 60, 90)], num_nodes=60)

    batch = Batch.from_data_list([data1, data2])

    assert batch.num_graphs == len(batch) == 2
    assert batch.num_nodes == 90

    assert torch.allclose(batch.x['1'],
                          torch.cat([data1.x['1'], data2.x['1']], dim=0))
    assert torch.allclose(batch.x['2'],
                          torch.cat([data1.x['2'], data2.x['2']], dim=0))
    assert (batch.edge_index[0].tolist() == torch.cat(
        [data1.edge_index[0], data2.edge_index[0] + 30], dim=1).tolist())
    assert (batch.edge_index[1].tolist() == torch.cat(
        [data1.edge_index[1], data2.edge_index[1] + 30], dim=1).tolist())
    assert batch.batch.size() == (90, )
    assert batch.ptr.size() == (3, )

    out1 = batch[0]
    assert len(out1) == 3
    assert out1.num_nodes == 30
    assert torch.allclose(out1.x['1'], data1.x['1'])
    assert torch.allclose(out1.x['2'], data1.x['2'])
    assert out1.edge_index[0].tolist(), data1.edge_index[0].tolist()
    assert out1.edge_index[1].tolist(), data1.edge_index[1].tolist()

    out2 = batch[1]
    assert len(out2) == 3
    assert out2.num_nodes == 60
    assert torch.allclose(out2.x['1'], data2.x['1'])
    assert torch.allclose(out2.x['2'], data2.x['2'])
    assert out2.edge_index[0].tolist(), data2.edge_index[0].tolist()
    assert out2.edge_index[1].tolist(), data2.edge_index[1].tolist()


def test_batching_of_batches():
    data = Data(x=torch.randn(2, 16))
    batch = Batch.from_data_list([data, data])

    batch = Batch.from_data_list([batch, batch])
    assert batch.num_graphs == len(batch) == 2
    assert batch.x[0:2].tolist() == data.x.tolist()
    assert batch.x[2:4].tolist() == data.x.tolist()
    assert batch.x[4:6].tolist() == data.x.tolist()
    assert batch.x[6:8].tolist() == data.x.tolist()
    assert batch.batch.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


def test_hetero_batch():
    e1 = ('p', 'a')
    e2 = ('a', 'p')
    data1 = HeteroData()
    data1['p'].x = torch.randn(100, 128)
    data1['a'].x = torch.randn(200, 128)
    data1[e1].edge_index = get_edge_index(100, 200, 500)
    data1[e1].edge_attr = torch.randn(500, 32)
    data1[e2].edge_index = get_edge_index(200, 100, 400)
    data1[e2].edge_attr = torch.randn(400, 32)

    data2 = HeteroData()
    data2['p'].x = torch.randn(50, 128)
    data2['a'].x = torch.randn(100, 128)
    data2[e1].edge_index = get_edge_index(50, 100, 300)
    data2[e1].edge_attr = torch.randn(300, 32)
    data2[e2].edge_index = get_edge_index(100, 50, 200)
    data2[e2].edge_attr = torch.randn(200, 32)

    batch = Batch.from_data_list([data1, data2])

    assert batch.num_graphs == len(batch) == 2
    assert batch.num_nodes == 450

    assert torch.allclose(batch['p'].x[:100], data1['p'].x)
    assert torch.allclose(batch['a'].x[:200], data1['a'].x)
    assert torch.allclose(batch['p'].x[100:], data2['p'].x)
    assert torch.allclose(batch['a'].x[200:], data2['a'].x)
    assert (batch[e1].edge_index.tolist() == torch.cat([
        data1[e1].edge_index,
        data2[e1].edge_index + torch.tensor([[100], [200]])
    ], 1).tolist())
    assert torch.allclose(
        batch[e1].edge_attr,
        torch.cat([data1[e1].edge_attr, data2[e1].edge_attr], 0))
    assert (batch[e2].edge_index.tolist() == torch.cat([
        data1[e2].edge_index,
        data2[e2].edge_index + torch.tensor([[200], [100]])
    ], 1).tolist())
    assert torch.allclose(
        batch[e2].edge_attr,
        torch.cat([data1[e2].edge_attr, data2[e2].edge_attr], 0))
    assert batch['p'].batch.size() == (150, )
    assert batch['p'].ptr.size() == (3, )
    assert batch['a'].batch.size() == (300, )
    assert batch['a'].ptr.size() == (3, )

    out1 = batch[0]
    assert len(out1) == 3
    assert out1.num_nodes == 300
    assert torch.allclose(out1['p'].x, data1['p'].x)
    assert torch.allclose(out1['a'].x, data1['a'].x)
    assert out1[e1].edge_index.tolist() == data1[e1].edge_index.tolist()
    assert torch.allclose(out1[e1].edge_attr, data1[e1].edge_attr)
    assert out1[e2].edge_index.tolist() == data1[e2].edge_index.tolist()
    assert torch.allclose(out1[e2].edge_attr, data1[e2].edge_attr)

    out2 = batch[1]
    assert len(out2) == 3
    assert out2.num_nodes == 150
    assert torch.allclose(out2['p'].x, data2['p'].x)
    assert torch.allclose(out2['a'].x, data2['a'].x)
    assert out2[e1].edge_index.tolist() == data2[e1].edge_index.tolist()
    assert torch.allclose(out2[e1].edge_attr, data2[e1].edge_attr)
    assert out2[e2].edge_index.tolist() == data2[e2].edge_index.tolist()
    assert torch.allclose(out2[e2].edge_attr, data2[e2].edge_attr)


def test_pair_data_batching():
    class PairData(Data):
        def __inc__(self, key, value, *args, **kwargs):
            if key == 'edge_index_s':
                return self.x_s.size(0)
            if key == 'edge_index_t':
                return self.x_t.size(0)
            else:
                return super().__inc__(key, value, *args, **kwargs)

    x_s = torch.randn(5, 16)
    edge_index_s = torch.tensor([
        [0, 0, 0, 0],
        [1, 2, 3, 4],
    ])
    x_t = torch.randn(4, 16)
    edge_index_t = torch.tensor([
        [0, 0, 0],
        [1, 2, 3],
    ])

    data = PairData(x_s=x_s, edge_index_s=edge_index_s, x_t=x_t,
                    edge_index_t=edge_index_t)
    batch = Batch.from_data_list([data, data])

    assert torch.allclose(batch.x_s, torch.cat([x_s, x_s], dim=0))
    assert batch.edge_index_s.tolist() == [[0, 0, 0, 0, 5, 5, 5, 5],
                                           [1, 2, 3, 4, 6, 7, 8, 9]]

    assert torch.allclose(batch.x_t, torch.cat([x_t, x_t], dim=0))
    assert batch.edge_index_t.tolist() == [[0, 0, 0, 4, 4, 4],
                                           [1, 2, 3, 5, 6, 7]]


def test_batch_with_empty_list():
    x = torch.randn(4, 1)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    data = Data(x=x, edge_index=edge_index, nontensor=[])

    batch = Batch.from_data_list([data, data])
    assert batch.nontensor == [[], []]
    assert batch[0].nontensor == []
    assert batch[1].nontensor == []


def test_nested_follow_batch():
    def tr(n, m):
        return torch.rand((n, m))

    d1 = Data(xs=[tr(4, 3), tr(11, 4), tr(1, 2)], a={"aa": tr(11, 3)},
              x=tr(10, 5))
    d2 = Data(xs=[tr(5, 3), tr(14, 4), tr(3, 2)], a={"aa": tr(2, 3)},
              x=tr(11, 5))
    d3 = Data(xs=[tr(6, 3), tr(15, 4), tr(2, 2)], a={"aa": tr(4, 3)},
              x=tr(9, 5))
    d4 = Data(xs=[tr(4, 3), tr(16, 4), tr(1, 2)], a={"aa": tr(8, 3)},
              x=tr(8, 5))

    # Dataset
    data_list = [d1, d2, d3, d4]

    batch = Batch.from_data_list(data_list, follow_batch=['xs', 'a'])

    # assert shapes
    assert batch.xs[0].shape == (19, 3)
    assert batch.xs[1].shape == (56, 4)
    assert batch.xs[2].shape == (7, 2)
    assert batch.a['aa'].shape == (25, 3)

    assert len(batch.xs_batch) == 3
    assert len(batch.a_batch) == 1

    # assert _batch
    assert batch.xs_batch[0].tolist() == \
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3]
    assert batch.xs_batch[1].tolist() == \
           [0] * 11 + [1] * 14 + [2] * 15 + [3] * 16
    assert batch.xs_batch[2].tolist() == \
           [0] * 1 + [1] * 3 + [2] * 2 + [3] * 1

    assert batch.a_batch['aa'].tolist() == \
           [0] * 11 + [1] * 2 + [2] * 4 + [3] * 8
