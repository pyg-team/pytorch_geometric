from typing import Any
from torch_geometric.typing import ProxyTensor
from torch_geometric.data.storage import GlobalStorage
from gaas_client.client import GaasClient

import torch

class TorchTensorGaasGraphDataProxy(ProxyTensor):
    """
    Implements a partial Torch Tensor interface that forwards requests to a
    GaaS server maintaining the actual data in a graph instance.
    The interface supported consists of only the APIs specific DGL workflows
    need - anything else will raise AttributeError.
    """
    _data_categories = ["vertex", "edge"]

    def __init__(self, gaas_client: GaasClient, gaas_graph_id: int, data_category: str):
        if data_category not in self._data_categories:
            raise ValueError("data_category must be one of "
                             f"{self._data_categories}, got {data_category}")
        self.__client = gaas_client
        self.__graph_id = gaas_graph_id
        self.__category = data_category

    def __getitem__(self, index: int):
        """
        Returns a torch.Tensor containing the edge or vertex data (based on the
        instance's data_category) for index, retrieved from graph data on the
        instance's GaaS server.
        """
        # tensor is a transposed dataframe (tensor[0] is df.iloc[0])
        if isinstance(index, torch.Tensor):
            index = [int(i) for i in index]

        if self.__category == "edge":
            if index > 1:
                raise IndexError(index)
            # FIXME find a more efficient way to do this that doesn't transfer so much data
            data = self.__client.get_graph_edge_dataframe_rows(
                index_or_indices=-1, graph_id=self.__graph_id)
        else:
            data = self.__client.get_graph_vertex_dataframe_rows(
                index_or_indices=index, graph_id=self.__graph_id)

        torch_data = torch.from_numpy(data.T)[index]
        if self.__category == 'vertex':
            return torch_data.to(torch.float32)
        return torch_data.to(torch.long)

    @property
    def shape(self) -> torch.Size:
        if self.__category == "edge":
            shape = self.__client.get_graph_edge_dataframe_shape(
                graph_id=self.__graph_id)
            return torch.Size([shape[1] - 1, shape[0]])
        else:
            shape = self.__client.get_graph_vertex_dataframe_shape(
                graph_id=self.__graph_id)
            return torch.Size(shape)
    
    @property
    def dtype(self) -> Any:
        if self.__category == 'edge':
            return torch.long
        else:
            return torch.float32
    
    def dim(self) -> int:
        return self.shape[0]
    
    def size(self, idx=None) -> Any:
        if idx is None:
            return self.shape
        else:
            return self.shape[idx]


class CuGraphStorage(GlobalStorage):
    def __init__(self, gaas_client: GaasClient, gaas_graph_id: int):
        super().__init__()
        self.gaas_client = gaas_client
        self.gaas_graph_id = gaas_graph_id
        self.node_index = TorchTensorGaasGraphDataProxy(gaas_client, gaas_graph_id, 'vertex')
        self.edge_index = TorchTensorGaasGraphDataProxy(gaas_client, gaas_graph_id, 'edge')
    
    @property
    def num_nodes(self) -> int:
        return self.gaas_client.get_num_vertices(self.gaas_graph_id)
    
    @property
    def num_edges(self) -> int:
        return self.gaas_client.get_num_edges(self.gaas_graph_id)
    
    