from torch_geometric.distributed import LocalGraphStore
from typing import Type, Protocol, Tuple, Union
from torch_geometric.sampler import BaseSampler, NeighborSampler
from torch_geometric.data import FeatureStore, GraphStore, Data, HeteroData
from torch_geometric.loader import NodeLoader
from abc import abstractmethod
from torch_geometric.typing import InputNodes, InputEdges
from torch import Tensor
from torch_geometric.sampler.neighbor_sampler import NumNeighborsType
import torch

class DataSampler(Protocol, BaseSampler):

    @abstractmethod
    def __init__(self, data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]]) -> None:
        ...

class NeighborSamplingRAGGraphStore(LocalGraphStore):
    def __init__(self, feature_store: FeatureStore, num_neighbors: NumNeighborsType = [10], **kwargs):
        self.feature_store = feature_store
        self.num_neighbors = num_neighbors
        self.sample_kwargs = kwargs
        self._sampler_is_initialized = False
        super().__init__() 
    
    def _init_sampler(self):
        self.sampler = NeighborSampler(data=(self.feature_store, self), num_neighbors=self.num_neighbors, **self.sample_kwargs)
        self._sampler_is_initialized = True
     
    def put_edge_id(self, edge_id: Tensor, *args, **kwargs) -> bool:
        ret = super().put_edge_id(edge_id, *args, **kwargs)
        self._sampler_is_initialized = False
        return ret
    
    def put_edge_index(self, edge_index: Tuple[Tensor], *args, **kwargs) -> bool:        
        ret = super().put_edge_index(edge_index, *args, **kwargs)
        self._sampler_is_initialized = False
        return ret

    
    def retrieve_subgraph(self, seed_nodes: InputNodes, seed_edges: InputEdges) -> Union[Data, HeteroData]:
        if not self._sampler_is_initialized:
            self._init_sampler()
        
        # Right now, only input nodes as tensors will be supported
        if not isinstance(seed_nodes, Tensor):
            raise NotImplementedError

        if seed_edges is not None:
            if isinstance(seed_edges, Tensor):
                seed_edges = seed_edges.reshape((-1))
                seed_nodes = torch.cat((seed_nodes, seed_edges), dim=0)
            else:
                raise NotImplementedError

        seed_nodes = seed_nodes.unique()
        num_nodes = len(seed_nodes)
        loader = NodeLoader(data=(self.feature_store, self), node_sampler=self.sampler, input_nodes=seed_nodes, batch_size=num_nodes)

        return next(iter(loader))
        