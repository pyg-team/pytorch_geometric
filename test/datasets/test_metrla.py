import shutil

from torch_geometric.datasets import MetrLa


def test_metrla() -> None:
    root = '/tmp/metr_la'
    n_previous_steps = 6
    n_future_steps = 6

    dataset = MetrLa(root=root, n_previous_steps=n_previous_steps,
                     n_future_steps=n_future_steps)

    expected_dataset_len = 34260  # (hard-coded for 6-prev and 6-next)

    # Sanity checks
    assert len(dataset.gdrive_ids) == 4

    # Path assertions
    assert dataset.raw_dir == f'{root}/raw'
    assert dataset.processed_dir == f'{root}/processed'

    assert dataset.io.dataset_len == expected_dataset_len

    # Pick a data point
    data = dataset[0]
    assert len(data) == 4

    # Assert data shapes
    assert data.x.size() == (6, 207, 1)
    assert data.y.size() == (6, 207, 1)

    # Assert COO adjacency matrix shapes
    assert list(data.edge_index.size()) == [2, 1722]
    assert list(data.edge_attr.size()) == [1722]

    shutil.rmtree(root)
