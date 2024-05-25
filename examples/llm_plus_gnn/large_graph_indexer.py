from typing import Tuple, Hashable, Dict, Iterable, Set, Optional, Any
import pickle as pkl

# Is there a multiprocessing-friendly implementation of this?

TripletLike = Tuple[Hashable, Hashable, Hashable]

class LargeGraphIndexer:
    """ For a dataset that consists of mulitiple subgraphs that are assumed to
        be part of a much larger graph, collate the values into a large graph store
        to save resources
    """

    # TODO: Add support for Hetero graphs

    def __init__(self, nodes: Dict[Hashable, int], edges: Dict[TripletLike, int], node_attr: Optional[Dict[int, Dict[str, Any]]] = None, edge_attr: Optional[Dict[int, Dict[str, Any]]] = None) -> None:
        self.nodes = nodes
        self.edges = edges

        self.node_attr = node_attr or {v: dict(pid=k) for k, v in nodes.items()}
        self.edge_attr = edge_attr or {v: dict(h=k[0], r=k[1], t=k[2]) for k, v in edges.items()}

    @classmethod
    def from_triplets(cls, triplets: Iterable[TripletLike]) -> 'LargeGraphIndexer':
        nodes = dict()
        edges = dict()


        for h, r, t in triplets:
            nodes[h] = nodes.get(h) if h in nodes else len(nodes)
            nodes[t] = nodes.get(t) if t in nodes else len(nodes)
            edge_idx = (h, r, t)
            edges[edge_idx] = edges.get(edge_idx) if edge_idx in edges else len(edges)
        
        return cls(nodes, edges)
    
    @classmethod
    def collate(cls, graphs: Iterable['LargeGraphIndexer'], skip_shared_check=False) -> 'LargeGraphIndexer':
        indexer = cls(dict(), dict())

        for other_indexer in graphs:
            if not skip_shared_check:
                # figure out which ids to reindex
                shared_nodes = other_indexer.nodes.keys() & indexer.nodes.keys()

                # This should be parallelizable
                for node in shared_nodes:
                    indexer_node_attr = indexer.node_attr[indexer.nodes[node]]
                    other_indexer_node_attr = other_indexer.node_attr[other_indexer.nodes[node]].copy()

                    # Assert that these are from the same graph
                    joint_keys = indexer_node_attr.keys() & other_indexer_node_attr.keys()
                    
                    for k, v in other_indexer_node_attr.items():
                        if k in joint_keys and indexer_node_attr[k] != other_indexer_node_attr[k]:
                            raise AttributeError(f"Node features do not match! Only subgraphs from the same graph can be collated.")
                        indexer.node_attr[indexer.nodes[node]][k] = v
            
            new_nodes = other_indexer.nodes.keys() - indexer.nodes.keys()

            for node in new_nodes:
                indexer.nodes[node] = len(indexer.nodes)
                indexer.node_attr[indexer.nodes[node]] = other_indexer.node_attr[other_indexer.nodes[node]].copy()

            if not skip_shared_check:
            
                shared_edges = other_indexer.edges.keys() & indexer.edges.keys()
                
                # This should also be parallelizable
                for edge in shared_edges:
                    indexer_edge_attr = indexer.edge_attr[indexer.edges[edge]]
                    other_indexer_edge_attr = other_indexer.edge_attr[other_indexer.edges[edge]] 

                    # Assert that these are from the same graph
                    joint_keys = indexer_edge_attr.keys() & other_indexer_edge_attr.keys()

                    for k, v in other_indexer_edge_attr.items():
                        if k in joint_keys and indexer_edge_attr[k] != other_indexer_edge_attr[k]:
                            raise AttributeError(f"Edge features do not match! Only subgraphs from the same graph can be collated.")
                        indexer.edge_attr[indexer.edges[edge]][k] = v
            
            new_edges = other_indexer.edges.keys() - indexer.edges.keys()

            for edge in new_edges:
                indexer.edges[edge] = len(indexer.edges)
                indexer.edge_attr[indexer.edges[edge]] = other_indexer.edge_attr[other_indexer.edges[edge]].copy()
            
        return indexer
    
    def get_unique_node_features(self, node_type: str) -> Set[Hashable]:
        try:
            return set([self.node_attr[i][node_type] for i in self.nodes.values()])
        except KeyError:
            raise AttributeError(f"Nodes do not have a feature called {node_type}")
    
    def get_unique_edge_features(self, edge_type: str) -> Set[Hashable]:
        try:
            return set([self.edge_attr[i][edge_type] for i in self.edges.values()])
        except KeyError:
            raise AttributeError(f"Edges do not have a feature called {edge_type}")
    
    def save(self, path: str) -> None:
        with open(path, "w") as f:
            pkl.dump(self, f)
        
    @classmethod
    def from_disk(cls, path: str) -> 'LargeGraphIndexer':
        with open(path) as f:
            return pkl.load(f)