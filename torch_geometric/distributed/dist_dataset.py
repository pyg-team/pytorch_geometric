
import torch

from typing import Dict, List, Optional, Union
from torch_geometric.data import Data, HeteroData
from torch_geometric.distributed.local_graph_store import LocalGraphStore as Graph
from torch_geometric.distributed.local_feature_store import LocalFeatureStore as Feature

from torch_geometric.distributed.partition import load_partition
from torch_geometric.typing import (
  NodeType, EdgeType, TensorDataType,
)

from torch_geometric.testing import get_random_edge_index


class DistDataset():
    r""" load the distributed graph/node_feats/edge_feats/label/partition books from partition files
    and if load from other partition format it will be initialized by graphstore/featurestore format.
    """
    def __init__(
        self,
        num_partitions: int = 1,
        partition_idx: int = 0,
        graph_partition: Union[Graph, Dict[EdgeType, Graph]] = None,
        node_feat_partition: Union[Feature, Dict[NodeType, Feature]] = None,
        edge_feat_partition: Union[Feature, Dict[EdgeType, Feature]] = None,
        node_labels: Union[TensorDataType, Dict[NodeType, TensorDataType]] = None,
        node_pb: Union[torch.Tensor, Dict[NodeType, torch.Tensor]] = None,
        edge_pb: Union[torch.Tensor, Dict[EdgeType, torch.Tensor]] = None,
        node_feat_pb: Union[torch.Tensor, Dict[NodeType, torch.Tensor]] = None,
        edge_feat_pb: Union[torch.Tensor, Dict[EdgeType, torch.Tensor]] = None,
    ):
        self.meta = None
        self.num_partitions = num_partitions
        self.partition_idx = partition_idx
       
        self.graph = graph_partition
        self.node_features = node_feat_partition
        self.edge_features = edge_feat_partition
        self.node_labels = node_labels

        self.node_pb = node_pb
        self.edge_pb = edge_pb
        
        self._node_feat_pb = node_feat_pb
        self._edge_feat_pb = edge_feat_pb
        #self.data = None
        
    
    def load(
        self,
        root_dir: str,
        partition_idx: int,
        node_label_file: Union[str, Dict[NodeType, str]] = None,
        partition_format:  str = "pyg",
        keep_pyg_data:  bool = True
    ):
        r""" Load one dataset partition from partitioned files.
        
        Args:
            root_dir (str): The file path to load the partition data.
            partition_idx (int): Current partition idx.
            node_label_file (str): The path to the node labels
            partition_format:  pyg/dgl/glt
            keep_pyg_data:  keep the original pyg data besides graphstore/featurestore.
        """
        if partition_format=="pyg":
            (
                self.meta,
                self.num_partitions,
                self.partition_idx,
                graph_data,
                node_feat_data,
                edge_feat_data,
                self.node_pb,
                self.edge_pb
            ) = load_partition(root_dir, partition_idx)
  
           
            # init graph/node feature/edge feature by graphstore/featurestore
            if(self.meta["is_hetero"]):
                # hetero ..
                
                # convert partition data into dict.
                edge_id_dict, edge_index_dict, num_nodes_dict, edge_attr_dict = {}, {}, {}, {}
                for etype in self.meta['edge_types']:
                    edge_id_dict[tuple(etype)] = graph_data[tuple(etype)]['edge_id']
                    edge_index_dict[tuple(etype)] = torch.tensor([graph_data[tuple(etype)]['row'].tolist(), graph_data[tuple(etype)]['col'].tolist()])
                    num_nodes_dict[etype[0]] =  graph_data[tuple(etype)]['size'][0]
                    num_nodes_dict[etype[2]] =  graph_data[tuple(etype)]['size'][1]

                    if edge_feat_data[tuple(etype)]['feats']['edge_attr'] is not None:
                        edge_attr_dict[tuple(etype)] = edge_feat_data[tuple(etype)]['feats']['edge_attr'] 

                node_id_dict, x_dict = {}, {}
                for ntype in self.meta['node_types']:
                    node_id_dict[ntype] = node_feat_data[ntype]['global_id']
                    
                    if node_feat_data[ntype]['feats']['x'] is not None:
                        x_dict[ntype] = node_feat_data[ntype]['feats']['x']
                
                # initialize graph
                self.graph = Graph.from_hetero_data(edge_id_dict, edge_index_dict, num_nodes_dict)
                
                # initialize edge features
                if len(edge_attr_dict)>0:
                    self.edge_features = Feature.from_hetero_data(node_id_dict=node_id_dict, edge_id_dict=edge_id_dict, edge_attr_dict=edge_attr_dict)
                    self._edge_feat_pb = self.edge_pb

                # initialize node features
                if len(x_dict)>0:
                    self.node_features = Feature.from_hetero_data(node_id_dict=node_id_dict, x_dict=x_dict)
                    self._node_feat_pb = self.node_pb

            else:
                # homo ..

                # initialize graph
                edge_index = torch.tensor([graph_data['row'].tolist(), graph_data['col'].tolist()])
                self.graph = Graph.from_data(graph_data['edge_id'], edge_index, num_nodes=graph_data['size'][0])

                # initialize node features
                if node_feat_data['feats']['x'] is not None:
                    self._node_feat_pb = self.node_pb
                    self.node_features = Feature.from_data(node_id=node_feat_data['global_id'], x=node_feat_data['feats']['x'])

                # initialize edge features
                if edge_feat_data['feats']['edge_attr'] is not None:
                    self._edge_feat_pb = self.edge_pb
                    self.edge_features = Feature.from_data(node_id=node_feat_data['global_id'], 
                            edge_id=edge_feat_data['global_id'], edge_attr=edge_feat_data['feats']['edge_attr'])
       

            if keep_pyg_data:
                # This will also generate the PyG Data format from Store format for back compatibility
                # besides the graphstore/featurestore format.
                
                if(self.meta["is_hetero"]):
                    # heterogeneous.
                    data = HeteroData()
                    
                    edge_attrs=self.graph.get_all_edge_attrs()
                    edge_index={}
                    edge_ids={}
                    for item in edge_attrs:
                        edge_index[item.edge_type] = self.graph.get_edge_index(item)
                        edge_ids[item.edge_type] = self.graph.get_edge_id(item)
                        data[item.edge_type].edge_index = edge_index[item.edge_type]

                    if len(x_dict)>0:
                        tensor_attrs = self.node_features.get_all_tensor_attrs()
                        node_feat={}
                        node_ids={}
                        for item in tensor_attrs:
                            node_feat[item.group_name] = self.node_features.get_tensor(item.fully_specify())
                            node_ids[item.group_name] = self.node_features.get_global_id(item.group_name) 
                            data[item.group_name].x = node_feat[item.group_name]
        
                    if len(edge_attr_dict)>0:
                        edge_attrs=self.edge_features.get_all_edge_attrs()
                        edge_feat={}
                        edge_ids={}
                        for item in edge_attrs:
                            edge_feat[item.edge_type] = self.edge_features.get_tensor(item.fully_specify())
                            edge_ids[item.edge_type] = self.edge_features.get_global_id(item.group_name) 
                            data[item.edge_type].edge_attr = edge_feat[item.edge_type]
                    self.data = data       
        
                else:
                    # homogeneous.
                    self.data = Data(x=node_feat_data['feats']['x'], edge_index=edge_index, edge_attr=edge_feat_data['feats']['edge_attr'])

        else:  
            # including other partition format, like dgl/glt ..
            # use LocalGraphStore.from_data() and LocalFeatureStore.from_data() api for initialization ..
            pass
    
        # init for labels
        if node_label_file is not None:
            if isinstance(node_label_file, dict):
                whole_node_labels = {}
                for ntype, file in node_label_file.items():
                    whole_node_labels[ntype] = torch.load(file)
            else:
                whole_node_labels = torch.load(node_label_file)
            self.node_labels = whole_node_labels
    
    @property
    def node_feat_pb(self):
      if self._node_feat_pb is None:
        return self.node_pb
      return self._node_feat_pb
    
    @property
    def edge_feat_pb(self):
      if self._edge_feat_pb is None:
        return self.edge_pb
      return self._edge_feat_pb

    def get_node_types(self):
        if(self.meta["is_hetero"]):
            edge_attrs=self.graph.get_all_edge_attrs()
            ntypes = set()
            for attrs in edge_attrs:
                ntypes.add(attrs.edge_type[0])
                ntypes.add(attrs.edge_type[2])
            self._node_types = list(ntypes)
            return self._node_types
        return None

    def get_edge_types(self):
        if(self.meta["is_hetero"]):
            edge_attrs=self.graph.get_all_edge_attrs()
            etypes = []
            for attrs in edge_attrs:
                etypes.append(attrs.edge_type)
            self._edge_types = etypes 
            return self._edge_types
        return None

    def get_node_label(self, ntype: Optional[NodeType] = None):
        if isinstance(self.node_labels, torch.Tensor):
            return self.node_labels
        if isinstance(self.node_labels, dict):
            assert ntype is not None
            return self.node_labels.get(ntype, None)
        return None

