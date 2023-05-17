import torch

from torch_geometric.loader import NeighborLoader, PrefetchLoader
from torch_geometric.nn import GraphSAGE
from torch_geometric.testing import withCUDA


@withCUDA
def test_prefetch_loader(device):
    data = [torch.randn(5, 5) for _ in range(10)]

    loader = PrefetchLoader(data, device=device)
    assert str(loader).startswith('PrefetchLoader')
    assert len(loader) == 10

    for i, batch in enumerate(loader):
        assert batch.device == device
        assert torch.equal(batch.cpu(), data[i])


if __name__ == '__main__':
    import argparse

    from ogb.nodeproppred import PygNodePropPredDataset
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    data = PygNodePropPredDataset('ogbn-products', root='/tmp/ogb')[0]

    model = GraphSAGE(
        in_channels=data.x.size(-1),
        hidden_channels=64,
        num_layers=2,
    ).cuda()

    loader = NeighborLoader(
        data,
        input_nodes=torch.arange(1024 * 200),
        batch_size=1024,
        num_neighbors=[10, 10],
        num_workers=args.num_workers,
        filter_per_worker=True,
        persistent_workers=args.num_workers > 0,
    )

    print('Forward pass without prefetching...')
    for batch in tqdm(loader):
        with torch.no_grad():
            batch = batch.cuda()
            model(batch.x, batch.edge_index)

    print('Forward pass with prefetching...')
    for batch in tqdm(PrefetchLoader(loader)):
        with torch.no_grad():
            model(batch.x, batch.edge_index)
