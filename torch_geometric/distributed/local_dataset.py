
import torch

from typing import Dict, List, Optional, Union

from torch_geometric.data import TensorAttr, EdgeAttr
from torch_geometric.distributed.local_graph_store import LocalGraphStore as Graph
from torch_geometric.distributed.local_feature_store import LocalFeatureStore as Feature

from torch_geometric.typing import NodeType, EdgeType, TensorDataType
from torch_geometric.utils import convert_to_tensor, squeeze


class LocalDataset(object):
    r""" Local data manager to initialize the graph topology and feature data 
    from each local partition by using the LocalGraphStore/LocalFeatureStore
    """

    def __init__(
        self,
        graph: Union[Graph, Dict[EdgeType, Graph]] = None,
        node_features: Union[Feature, Dict[NodeType, Feature]] = None,
        edge_features: Union[Feature, Dict[EdgeType, Feature]] = None,
        node_labels: Union[TensorDataType, Dict[NodeType, TensorDataType]] = None
        ):
        self.graph = graph
        self.node_features = node_features
        self.edge_features = edge_features
        self.node_labels = squeeze(convert_to_tensor(node_labels))

    def init_graph(
        self,
        edge_index: Union[TensorDataType, Dict[EdgeType, TensorDataType]] = None,
        edge_ids: Union[TensorDataType, Dict[EdgeType, TensorDataType]] = None,
        layout: Union[str, Dict[EdgeType, str]] = 'COO',
        directed: bool = False
        ):
        r""" Initialize the graph data storage and build the Graph.
        
        Args:
        edge_index: Edge index for graph topo,
        edge_ids:   Edge ids for graph edges, 
        layout (str): The edge layout representation for the input edge index,
        directed (bool):indicate graph topology is directed or not.(default: ``False``)
        """
        edge_index = convert_to_tensor(edge_index, dtype=torch.int64)
        edge_ids = convert_to_tensor(edge_ids, dtype=torch.int64)
        self._directed = directed
        
        if edge_index is not None:
            if isinstance(edge_index, dict):
                # heterogeneous.
                if edge_ids is not None:
                    assert isinstance(edge_ids, dict)
                else:
                    edge_ids = {}
                if not isinstance(layout, dict):
                    layout = {etype: layout for etype in edge_index.keys()}

                graph_store = Graph() 
                for etype, e_idx in edge_index.items():
                    node_num = e_idx[0].size()
                    graph_store.put_edge_index(
                        edge_index=e_idx,
                        edge_type=etype, layout='coo',
                        size=(node_num, node_num))            
                    graph_store.put_edge_id(
                        edge_id=edge_ids[etype],
                        edge_type=etype, layout='coo',
                        size=(node_num, node_num))
                    self.graph = graph_store
            else:
                # homogeneous.
                graph_store = Graph()
                node_num = edge_index[0].size()
                graph_store.put_edge_index(
                    edge_index=edge_index,
                    edge_type=None, layout='coo', size=(node_num, node_num))
                graph_store.put_edge_id(edge_id=edge_ids,
    				    edge_type=None, layout='coo', size=(node_num, node_num))
                self.graph = graph_store


    def init_node_features(
        self,
        node_feature_data: Union[TensorDataType, Dict[NodeType, TensorDataType]] = None,
        ids: Union[TensorDataType, Dict[EdgeType, TensorDataType]] = None,
        partition_idx: int = 1,
        dtype: Optional[torch.dtype] = None
        ):
        r""" Initialize the node feature data storage.
        Args:
        node_feature_data: raw node feature data,
        ids:  global node ids
        id2idx: mapping between node id and local feature index
        """
        
        if node_feature_data is not None:
            self.node_features = create_features(
                convert_to_tensor(node_feature_data), convert_to_tensor(ids), partition_idx, "node_feat"
                )
    
    def init_edge_features(
        self,
        edge_feature_data: Union[TensorDataType, Dict[EdgeType, TensorDataType]] = None,
        ids: Union[TensorDataType, Dict[EdgeType, TensorDataType]] = None,
        partition_idx: int = 1,
        dtype: Optional[torch.dtype] = None
        ):
        r""" Initialize the edge feature data by LocalFeatureStore. 
        Args:
        edge_feature_data: raw edge feature data, 
        ids:  global edge ids
        id2idx: mapping between edge id to index.
        """    
        
        if edge_feature_data is not None:
            self.edge_features = create_features(
            convert_to_tensor(edge_feature_data, dtype), convert_to_tensor(ids), partition_idx, "edge_feat"
            )
    
    def init_node_labels(
        self,
        node_label_data: Union[TensorDataType, Dict[NodeType, TensorDataType]] = None
        ):
        #Initialize the node labels.
        if node_label_data is not None:
    	    self.node_labels = squeeze(convert_to_tensor(node_label_data))
     
    
    
def create_features(feature_data, ids, partition_idx, attr_name):
    # Initialize the node/edge feature by FeatureStore.
    if feature_data is not None:    	    
        if isinstance(feature_data, dict):
            # heterogeneous.    
            features = Feature()
            for graph_type, feat in feature_data.items():
                features.put_tensor(feat, group_name=f'part_{partition_idx}', attr_name=graph_type, index=None)    
                if ids is not None:
                    features.put_global_id(ids[graph_type], group_name=f'part_{partition_idx}', attr_name=graph_type)
        else:
            # homogeneous.    
                features = Feature()
                features.put_tensor(feature_data, group_name=f'part_{partition_idx}', attr_name=None, index=None)
                if ids is not None:
                    features.put_global_id(ids, group_name=f'part_{partition_idx}', attr_name=None)
    else:
        features = None

    return features
