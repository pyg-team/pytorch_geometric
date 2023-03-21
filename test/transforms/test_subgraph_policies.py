import torch
from torch_geometric.data import Data, Batch
from torch_geometric.transform import SubgraphData, SubgraphPolicy, NodeDeletionPolicy, EdgeDeletionPolicy, EgoNetPolicy

def test_subgraph_data():
    x = torch.randn(5, 10)
    edge_index = torch.tensor([[0, 1, 1, 2, 3], [1, 0, 2, 1, 4]])
    edge_attr = torch.randn(5, 2)
    y = torch.tensor([0, 1, 0, 1, 0])
    pos = torch.randn(5, 3)
    
    subgraph_id = torch.tensor([0, 0, 1, 1, 1])
    subgraph_batch = torch.tensor([0, 1])
    subgraph_n_id = torch.tensor([0, 1, 2, 3])
    orig_edge_index = torch.tensor([[0, 1, 1, 2, 3], [1, 0, 2, 1, 4]])
    orig_edge_attr = torch.randn(5, 2)
    num_subgraphs = 2
    num_nodes_per_subgraph = 4
    
    subgraph_data = SubgraphData(x=x, edge_index=edge_index, edge_attr=edge_attr,
                                 y=y, pos=pos, subgraph_id=subgraph_id,
                                 subgraph_batch=subgraph_batch, subgraph_n_id=subgraph_n_id,
                                 orig_edge_index=orig_edge_index, orig_edge_attr=orig_edge_attr,
                                 num_subgraphs=num_subgraphs,
                                 num_nodes_per_subgraph=num_nodes_per_subgraph)
    
    assert subgraph_data.__inc__('orig_edge_index', None) == num_nodes_per_subgraph
    assert subgraph_data.__inc__('subgraph_batch', None) == 0
    assert subgraph_data.__inc__('invalid_key', None) == 1
    
    assert torch.all(torch.eq(subgraph_data.subgraph_id, subgraph_id))
    assert torch.all(torch.eq(subgraph_data.subgraph_batch, subgraph_batch))
    assert torch.all(torch.eq(subgraph_data.subgraph_n_id, subgraph_n_id))
    assert torch.all(torch.eq(subgraph_data.orig_edge_index, orig_edge_index))
    assert torch.all(torch.eq(subgraph_data.orig_edge_attr, orig_edge_attr))
    assert subgraph_data.num_subgraphs == num_subgraphs
    assert subgraph_data.num_nodes_per_subgraph == num_nodes_per_subgraph
 

def test_node_deletion_policy():
    # Create an input graph with 4 nodes and 4 edges
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    edge_attr = torch.randn(edge_index.size(1), 16)
    x = torch.randn(4, 32)
    y = torch.tensor([0, 1, 0, 1])

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    # Apply node deletion policy
    policy = NodeDeletionPolicy()
    subgraphs_data = policy(data)

    # Check the output
    assert subgraphs_data.num_subgraphs == 4

    for i in range(4):
        subgraph_data = subgraphs_data[i]

        # Check the subgraph data
        assert subgraph_data.num_nodes == 3
        assert subgraph_data.subgraph_id == i

        # Check that the deleted node is not in the subgraph
        assert i not in subgraph_data.subgraph_n_id.tolist()
        assert subgraph_data.edge_index.shape[1] == 2
        
def test_edge_deletion_policy():
    # Define a toy input graph
    x = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1, 2, 2, 3], [1, 2, 0, 3, 4]], dtype=torch.long)
    edge_attr = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    # Instantiate an EdgeDeletionPolicy object
    policy = EdgeDeletionPolicy()
    
    # Apply the policy to the input graph
    subgraph_data = policy(data)
    
    # Check that the number of resulting subgraphs is correct
    assert len(subgraph_data) == data.num_edges
    
    # Check that each subgraph is undirected
    for i in range(len(subgraph_data)):
        assert subgraph_data[i].is_undirected()
    
    # Check that each subgraph has the correct number of nodes
    for i in range(len(subgraph_data)):
        assert subgraph_data[i].num_nodes == data.num_nodes
    
    # Check that each subgraph has the correct number of edges
    for i in range(len(subgraph_data)):
        assert subgraph_data[i].num_edges == data.num_edges - 1
    
    # Check that each subgraph has the correct edge set
    for i in range(len(subgraph_data)):
        head, tail = subgraph_data[i].edge_index
        edge_mask = (head != 2) & (tail != 2)
        assert torch.all(head[edge_mask] == data.edge_index[0][edge_mask])
        assert torch.all(tail[edge_mask] == data.edge_index[1][edge_mask])
        assert torch.all(subgraph_data[i].edge_attr[edge_mask] == data.edge_attr[edge_mask])     
        
def test_ego_net_policy():
    # Define some input data
    x = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_attr = torch.tensor([[0.1], [0.2]], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # Create an instance of the EgoNetPolicy with 1 hop and ego_plus=False
    policy = EgoNetPolicy(num_hops=1, ego_plus=False)

    # Apply the policy to the input data
    subgraphs = policy(data)

    # Check that the output is a list of Data objects
    assert isinstance(subgraphs, list)
    assert all(isinstance(subgraph, Data) for subgraph in subgraphs)

    # Check that the output contains the expected number of subgraphs
    expected_num_subgraphs = 3
    assert len(subgraphs) == expected_num_subgraphs

    # Check that each subgraph has the expected attributes
    expected_subgraph_ids = torch.arange(expected_num_subgraphs)
    expected_num_nodes = data.num_nodes
    for i, subgraph in enumerate(subgraphs):
        assert subgraph.x.shape == (expected_num_nodes, 2)
        assert subgraph.edge_index.shape == (2, 2)
        assert subgraph.edge_attr.shape == (1,)
        assert torch.all(subgraph.subgraph_id == expected_subgraph_ids[i])
        assert torch.all(subgraph.subgraph_n_id == torch.arange(expected_num_nodes))
        assert subgraph.num_nodes == expected_num_nodes

    # Create an instance of the EgoNetPolicy with 1 hop and ego_plus=True
    policy = EgoNetPolicy(num_hops=1, ego_plus=True)

    # Apply the policy to the input data
    subgraphs = policy(data)

    # Check that the output contains the expected number of subgraphs
    expected_num_subgraphs = 3
    assert len(subgraphs) == expected_num_subgraphs

    # Check that each subgraph has the expected attributes
    expected_subgraph_ids = torch.arange(expected_num_subgraphs)
    expected_num_nodes = data.num_nodes
    for i, subgraph in enumerate(subgraphs):
        assert subgraph.x.shape == (expected_num_nodes, 4)
        assert subgraph.edge_index.shape == (2, 2)
        assert subgraph.edge_attr.shape == (1,)
        assert torch.all(subgraph.subgraph_id == expected_subgraph_ids[i])
        assert torch.all(subgraph.subgraph_n_id == torch.arange(expected_num_nodes))
        assert subgraph.num_nodes == expected_num_nodes
