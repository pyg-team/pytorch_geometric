import torch
from torch_geometric.data.data2 import Data


def test_data():
    tmp = '_'

    data = Data({
        'author': {
            'x': tmp
        },
        ('paper', 'author'): {
            'edge_attr': tmp
        }
    }, author={'y': tmp}, paper__author={'edge_index': tmp}, x=tmp)
    print("DRIN")
    print(data)

    print(data._to_canonical('author'))
    print(data._to_canonical(('author', )))
    print(data._to_canonical('author', 'paper'))
    print(data._to_canonical(('author', 'paper')))

    print(data['author'])
    print(data['paper', 'author'])
    print(data['paper', '_', 'author'])
    print(data[('paper', '_', 'author')])
    print(data[('paper', 'author')])

    print(data.metadata)
