from torch_geometric.data import Data
from torch_geometric.data.cugraph.cugraph_storage import CuGraphStorage

from gaas_client.client import GaasClient
from gaas_client.defaults import graph_id as DEFAULT_GRAPH_ID

class CuGraphData(Data):
    def __init__(self, gaas_client: GaasClient, graph_id: int=DEFAULT_GRAPH_ID):
        super().__init__()

        self.__dict__['_store'] = CuGraphStorage(gaas_client, graph_id)