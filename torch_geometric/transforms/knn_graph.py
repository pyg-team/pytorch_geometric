import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_undirected


@functional_transform('knn_graph')
class KNNGraph(BaseTransform):
    r"""Creates a k-NN graph based on node positions :obj:`pos`
    (functional name: :obj:`knn_graph`).

    Args:
        k (int, optional): The number of neighbors. (default: :obj:`6`)
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
        force_undirected (bool, optional): If set to :obj:`True`, new edges
            will be undirected. (default: :obj:`False`)
        flow (string, optional): The flow direction when used in combination
            with message passing (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`).
            If set to :obj:`"source_to_target"`, every target node will have
            exactly :math:`k` source nodes pointing to it.
            (default: :obj:`"source_to_target"`)
        cosine (boolean, optional): If :obj:`True`, will use the cosine
            distance instead of euclidean distance to find nearest neighbors.
            (default: :obj:`False`)
        num_workers (int): Number of workers to use for computation. Has no
            effect in case :obj:`batch` is not :obj:`None`, or the input lies
            on the GPU. (default: :obj:`1`)
    """
    def __init__(
        self,
        k: int = 6,
        loop: bool = False,
        force_undirected: bool = False,
        flow: str = 'source_to_target',
        cosine: bool = False,
        num_workers: int = 1,
    ):
        self.k = k
        self.loop = loop
        self.force_undirected = force_undirected
        self.flow = flow
        self.cosine = cosine
        self.num_workers = num_workers

    def __call__(self, data: Data) -> Data:
        data.edge_attr = None
        batch = data.batch if 'batch' in data else None

        edge_index = torch_geometric.nn.knn_graph(
            data.pos,
            self.k,
            batch,
            loop=self.loop,
            flow=self.flow,
            cosine=self.cosine,
            num_workers=self.num_workers,
        )

        if self.force_undirected:
            edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)

        data.edge_index = edge_index

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(k={self.k})'
