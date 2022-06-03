from torch import device as TorchDevice
from torch_geometric.data import Data
from torch_geometric.data.cugraph.cugraph_storage import CuGraphStorage

from gaas_client.client import GaasClient
from gaas_client.defaults import graph_id as DEFAULT_GRAPH_ID

class CuGraphData(Data):
    def __init__(self, gaas_client: GaasClient, graph_id: int=DEFAULT_GRAPH_ID, device=TorchDevice('cpu')):
        super().__init__()

        self.__dict__['_store'] = CuGraphStorage(gaas_client, graph_id, device=device)
        self.__dict__['device'] = device
    
    def to(self, to_device: TorchDevice) -> Data:
        return CuGraphData(
            self.__dict__['_store'].gaas_client,
            self.__dict__['_store'].gaas_graph_id,
            to_device
        )