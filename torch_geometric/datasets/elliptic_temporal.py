from typing import Any, Callable, Optional, Tuple

from torch_geometric.datasets import EllipticBitcoinDataset


class EllipticBitcoinTemporalDataset(EllipticBitcoinDataset):
    r"""The time-step aware Elliptic Bitcoin dataset of Bitcoin transactions
    from the `"Anti-Money Laundering in Bitcoin: Experimenting with Graph
    Convolutional Networks for Financial Forensics"
    <https://arxiv.org/abs/1908.02591>`_ paper.

    :class:`EllipticBitcoinTemporalDataset` maps Bitcoin transactions to real
    entities belonging to licit categories (exchanges, wallet providers,
    miners, licit services, etc.) versus illicit ones (scams, malware,
    terrorist organizations, ransomware, Ponzi schemes, etc.)

    There exists 203,769 node transactions and 234,355 directed edge payments
    flows, with two percent of nodes (4,545) labelled as illicit, and
    twenty-one percent of nodes (42,019) labelled as licit.
    The remaining transactions are unknown.

    .. note::

        In contrast to :class:`EllipticBitcoinDataset`, this dataset returns
        Bitcoin transactions only for a given timestamp :obj:`t`.

    Args:
        root (str): Root directory where the dataset should be saved.
        t (int): The Timestep for which nodes should be selected (from :obj:`1`
            to :obj:`49`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10
        :header-rows: 1

        * - #nodes
          - #edges
          - #features
          - #classes
        * - 203,769
          - 234,355
          - 165
          - 2
    """
    def __init__(
        self,
        root: str,
        t: int,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        if t < 1 or t > 49:
            raise ValueError("'t' needs to be between 1 and 49")

        self.t = t
        super().__init__(root, transform, pre_transform)

    @property
    def processed_file_names(self) -> str:
        return f'data_t_{self.t}.pt'

    def _process_df(self, feat_df: Any, edge_df: Any,
                    class_df: Any) -> Tuple[Any, Any, Any]:

        feat_df = feat_df[feat_df['time_step'] == self.t]

        mask = edge_df['txId1'].isin(feat_df['txId'].values)
        edge_df = edge_df[mask]

        class_df = class_df.merge(feat_df[['txId']], how='right',
                                  left_on='txId', right_on='txId')

        return feat_df, edge_df, class_df
