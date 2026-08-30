import os
from typing import Callable, List, Optional

import numpy as np
import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)


class SynthFinDataset(InMemoryDataset):
    r"""The SynthFin-AML dataset, an anti-money laundering (AML) benchmark
    graph. It contains 100k nodes and 1.27M edges, representing a 10-day
    snapshot of financial transactions with injected laundering topologies
    (e.g., Structuring).

    Nodes represent bank accounts and edges represent transactions.
    The task is transductive node classification, where the goal is to classify
    nodes as either clean (0) or fraud (1).

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-download and re-process
            the dataset. (default: :obj:`False`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10
        :header-rows: 1

        * - #nodes
          - #edges
          - #features
          - #classes
        * - 100,000
          - 1,273,403
          - 10
          - 2
    """
    url = ('https://huggingface.co/datasets/ovvaliyev/'
           'synthfin-aml/resolve/main/raw.zip')

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['nodes.csv', 'edges.csv']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self) -> None:
        import networkx as nx
        import pandas as pd

        nodes_df = pd.read_csv(self.raw_paths[0])
        edges_df = pd.read_csv(self.raw_paths[1])

        # Compute explicit topological features required for the benchmark
        od = edges_df.groupby('source_id').size().rename('out_degree')
        id_ = edges_df.groupby('target_id').size().rename('in_degree')
        ov = edges_df.groupby('source_id')['amount'].sum().rename('out_volume')
        iv = edges_df.groupby('target_id')['amount'].sum().rename('in_volume')
        om = edges_df.groupby('source_id')['amount'].max().rename(
            'out_max_amt')
        im = edges_df.groupby('target_id')['amount'].max().rename('in_max_amt')

        ev = edges_df.merge(iv, left_on='target_id', right_index=True,
                            how='left')
        ev = ev.merge(ov, left_on='source_id', right_index=True, how='left')
        ni = ev.groupby('source_id')['in_volume'].mean().rename(
            'nbr_in_volume')
        no_ = ev.groupby('target_id')['out_volume'].mean().rename(
            'nbr_out_volume')

        G = nx.from_pandas_edgelist(edges_df, 'source_id', 'target_id',
                                    edge_attr='amount',
                                    create_using=nx.DiGraph())
        pr = pd.Series(nx.pagerank(G, weight='amount'), name='pagerank')

        feat = nodes_df.set_index('agent_id').copy()
        feat = feat.join([od, id_, ov, iv, om, im, ni, no_, pr]).fillna(0)

        for c in [
                'initial_balance', 'out_volume', 'in_volume', 'out_max_amt',
                'in_max_amt', 'nbr_in_volume', 'nbr_out_volume', 'pagerank'
        ]:
            feat[c] = np.log1p(feat[c])

        y_val = feat['is_fraud'].values
        X_df = feat.drop(columns=['profile', 'is_fraud'])
        X_val = X_df.values

        # Standardize features
        mu, sd = X_val.mean(0), X_val.std(0)
        X_s = (X_val - mu) / (sd + 1e-5)

        # Mapping node IDs to continuous range 0..N-1
        a2i = {a: i for i, a in enumerate(feat.index)}

        edges_mapped = edges_df.copy()
        edges_mapped['src'] = edges_mapped['source_id'].map(a2i)
        edges_mapped['dst'] = edges_mapped['target_id'].map(a2i)
        edges_mapped = edges_mapped.dropna(subset=['src', 'dst'])

        src = edges_mapped['src'].astype(int).values
        dst = edges_mapped['dst'].astype(int).values
        amounts = edges_mapped['amount'].values

        edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
        edge_attr = torch.tensor(np.log1p(amounts), dtype=torch.float32)

        x = torch.tensor(X_s, dtype=torch.float32)
        y = torch.tensor(y_val, dtype=torch.long)

        # Create standard transductive masks (80% train / 10% val / 10% test, stratified)
        rng = np.random.default_rng(42)
        fraud_idx = np.where(y_val == 1)[0]
        clean_idx = np.where(y_val == 0)[0]

        rng.shuffle(fraud_idx)
        rng.shuffle(clean_idx)

        f_train = int(0.8 * len(fraud_idx))
        c_train = int(0.8 * len(clean_idx))
        f_val = int(0.9 * len(fraud_idx))
        c_val = int(0.9 * len(clean_idx))

        train_idx = np.concatenate([fraud_idx[:f_train], clean_idx[:c_train]])
        val_idx = np.concatenate(
            [fraud_idx[f_train:f_val], clean_idx[c_train:c_val]])
        test_idx = np.concatenate([fraud_idx[f_val:], clean_idx[c_val:]])

        train_mask = torch.zeros(len(y), dtype=torch.bool)
        val_mask = torch.zeros(len(y), dtype=torch.bool)
        test_mask = torch.zeros(len(y), dtype=torch.bool)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                    train_mask=train_mask, val_mask=val_mask,
                    test_mask=test_mask)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
