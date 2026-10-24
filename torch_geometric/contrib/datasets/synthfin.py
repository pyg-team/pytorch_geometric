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
    The task is inductive node classification across temporal snapshots, where
    the goal is to classify nodes as either clean (0) or fraud (1).

    To prevent temporal leakage (where future transactions leak into past node
    features or message passing), the dataset is split into three strictly
    isolated inductive graphs:
    - **dataset[0] (Train):** Edges Day <=7. Features on Days 1-7.
    - **dataset[1] (Val):** Edges Day <=8. Features on Days 1-8.
    - **dataset[2] (Test):** Edges Day <=10. Features on Days 1-10.

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
          - #edges (Test)
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

        # Extract temporal information from transactions
        edges_df['timestamp'] = pd.to_datetime(edges_df['timestamp'])
        min_ts = edges_df['timestamp'].min().floor('D')
        edges_df['edge_time'] = (edges_df['timestamp'] - min_ts).dt.days + 1

        y_val = nodes_df.set_index('agent_id')['is_fraud'].values

        # Mapping node IDs to continuous range 0..N-1
        a2i = {a: i for i, a in enumerate(nodes_df['agent_id'])}

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

        data_list = []
        for i, (day_limit, mask_idx) in enumerate([(7, train_idx),
                                                   (8, val_idx),
                                                   (10, test_idx)]):
            snap_edges = edges_df[edges_df['edge_time'] <= day_limit]

            # Compute explicit topological features required for the benchmark
            od = snap_edges.groupby('source_id').size().rename('out_degree')
            id_ = snap_edges.groupby('target_id').size().rename('in_degree')
            ov = snap_edges.groupby('source_id')['amount'].sum().rename(
                'out_volume')
            iv = snap_edges.groupby('target_id')['amount'].sum().rename(
                'in_volume')
            om = snap_edges.groupby('source_id')['amount'].max().rename(
                'out_max_amt')
            im = snap_edges.groupby('target_id')['amount'].max().rename(
                'in_max_amt')

            ev = snap_edges.merge(iv, left_on='target_id', right_index=True,
                                  how='left')
            ev = ev.merge(ov, left_on='source_id', right_index=True,
                          how='left')
            ni = ev.groupby('source_id')['in_volume'].mean().rename(
                'nbr_in_volume')
            no_ = ev.groupby('target_id')['out_volume'].mean().rename(
                'nbr_out_volume')

            G = nx.from_pandas_edgelist(snap_edges, 'source_id', 'target_id',
                                        edge_attr='amount',
                                        create_using=nx.DiGraph())
            pr = pd.Series(nx.pagerank(G, weight='amount'), name='pagerank')

            feat = nodes_df.set_index('agent_id').copy()
            feat = feat.join([od, id_, ov, iv, om, im, ni, no_, pr]).fillna(0)

            for c in [
                    'initial_balance', 'out_volume', 'in_volume',
                    'out_max_amt', 'in_max_amt', 'nbr_in_volume',
                    'nbr_out_volume', 'pagerank'
            ]:
                feat[c] = np.log1p(feat[c])

            X_df = feat.drop(columns=['profile', 'is_fraud'])
            X_val = X_df.values

            # Standardize features (fit on Train (Graph 0) to prevent leakage)
            if i == 0:
                mu, sd = X_val.mean(0), X_val.std(0)
            X_s = (X_val - mu) / (sd + 1e-5)

            edges_mapped = snap_edges.copy()
            edges_mapped['src'] = edges_mapped['source_id'].map(a2i)
            edges_mapped['dst'] = edges_mapped['target_id'].map(a2i)
            edges_mapped = edges_mapped.dropna(subset=['src', 'dst'])

            src = edges_mapped['src'].astype(int).values
            dst = edges_mapped['dst'].astype(int).values
            amounts = edges_mapped['amount'].values
            edge_times = edges_mapped['edge_time'].astype(int).values

            edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
            edge_attr = torch.tensor(np.log1p(amounts), dtype=torch.float32)
            edge_time = torch.tensor(edge_times, dtype=torch.long)

            x = torch.tensor(X_s, dtype=torch.float32)
            y = torch.tensor(y_val, dtype=torch.long)

            mask = torch.zeros(len(y), dtype=torch.bool)
            mask[mask_idx] = True

            # Return different masks depending on the graph index
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                        edge_time=edge_time,
                        train_mask=mask if i == 0 else torch.zeros_like(mask),
                        val_mask=mask if i == 1 else torch.zeros_like(mask),
                        test_mask=mask if i == 2 else torch.zeros_like(mask))

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])
