Hierarchical Graph Adjacency Matrix (HGAM) 
==========================================

HGAM it is a technique available to :pyg:`PyG` users which progressively trims the adjacency matrix of the graph of the batch being fed into a GNN. It works seamlessly across several models, basically reducing the amount of compute necessary to generate the representations for the seed node of the given batch.
The seed nodes are indeed the nodes selected initially by the dataloader, and for which the sampling procedure builds a subgraph starting from those seed nodes. 
The batch graph thus obtained is typically fed as it is into the GNN training process, which would naively run message passing at all depths and build representations at all depths for all nodes included in the initial batch graph.

Crucially, HGAM recognizes that all this computation is not necessary to generate the final representations for the seed nodes (which are the real target of the batch computation); HGAM allows for every layer of the GNN to compute only the representations of the nodes that are necessary for that layer, leading to a reduction of the computation and a speed up of the training process that grows with the depth of the GNN being considered.
This is achieved in practice by trimming the adjacency matrix and the various features matrix as the computation proceeds throughout the GNN layers, in line with the fact that in order to compute the representation for the seed nodes (also called target nodes), from which the batch subgraph was build via sampling methods, the depth of the relevant neighborhood shrinks as we proceed through the layers of the GNN.
The trimming applied by HGAM is possible as the nodes of the subgraph built via sampling are ordered according to a BFS (Breadth First Search) strategy, meaning that the adjacency matrix rows and columns refer to an node ordering that starts with the seed nodes (in any order) followed by the 1-hop neighbors of the first seed node, followed by the 1-hop sampled neighbors of the second seed node and so on.
After the 1-hop neighbors have been all listed, then the 1-hop neighbors of the first 1-hop neighbors of the first seed node will be added, and so on with the same logic for all required depths, according to the number of layers of the GNN.

To support this trimming and implement it effectively, the neighbor loader has been modified in such a way that returns a set of metadata on the sampled graph that help the trimming procedure.
HGAM is currently an option that :pyg:`PyG` users are free to switch on, or leave it off (current default).

.. note::   
    To sum up, Hierarchical Graph Adjacency Matrix(HGAM) is special data structure that enables efficient message passing computation.

// Thanks :intel:`null` and u.

HGAM feature is implemented in :pyg:`PyG` and can be used by utilizing  :class:`~torch_geometric.loader.NeighborLoader` and special `trim_to_layer <https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/utils/trim_to_layer.py>`__ utils functionalities.


How the BFS batch ordering allows to trim efficiently the adj matrix 
--------------------------------------------------------------------

BFS ordering of nodes in a batch allows for incremental trimming (reduction) of the adjacency matrix that describe the batch subgraph. 
This progressive trimming is done in a computational convenient manner thanks to the BFS ordering, that causes the nodes more distant from the seed nodes to be appear farther away in the list of ordered nodes.
Thus creating an adjacency matrix that can be reduced by simply dropping rows and columns corresponding to the nodes that are no longer needed as the computation goes to the next GNN layer.


New metadata retuned by pyg-lib neighbor sampler to support HGAM
-----------------------------------------------------------------

The additional metadata that the new version of neighbor loader returns how many new nodes and new edges have been added to the graph representing the batch at each layer during the batch graph creation process done by the neighbor loader. 
This information allows for fast manipulation of the adjacency matrix, which in turns lead to great computation reduction.


HGAM usage
----------

NeighborLoader prepares needed meta-data structure for HGAM by default.
It can be accessed from batch object returned by NeighborLoader for both homogenous and heterogenous data.


Homogenous data example:

.. code-block:: python

    from torch_geometric.datasets import Planetoid
    from torch_geometric.loader import NeighborLoader

    data = Planetoid(path, name='Cora')[0]

    loader = NeighborLoader(
        data,
        # Sample 10 neighbors for each node for 3 iterations
        num_neighbors=[10] * 3,
        # Use a batch size of 128 for sampling training nodes
        batch_size=128,
        input_nodes=data.train_mask,
    )

    sampled_data = next(iter(loader))
    print(sampled_data)
    >>> Data(x=[1883, 1433], edge_index=[2, 5441], y=[1883], train_mask=[1883], val_mask=[1883], test_mask=[1883], n_id=[1883], e_id=[5441], num_sampled_nodes=[4], num_sampled_edges=[3], input_id=[128], batch_size=128)

    print(sampled_data.num_sampled_nodes)
    >>> [128, 425, 702, 628] # Number of sampled nodes per iteration/layer
    print(sampled_data.num_sampled_edges)
    >>> [520, 2036, 2885] # Number of sampled edges per iteration/layer


Heterogeneous data example:

.. code-block:: python

    from torch_geometric.datasets import OGB_MAG
    from torch_geometric.loader import NeighborLoader

    hetero_data = OGB_MAG(root='../data')[0]

    loader = NeighborLoader(
        hetero_data,
        # Sample 0 neighbors for each node and edge type for 2 iterations
        num_neighbors={key: [10] * 2 for key in hetero_data.edge_types},
        # Use a batch size of 128 for sampling training nodes of type paper
        batch_size=128,
        input_nodes=('paper', hetero_data['paper'].train_mask),
    )

    sampled_hetero_data = next(iter(loader))
    print(sampled_hetero_data)
    >>> HeteroData(
        paper={
            x=[2275, 128],
            year=[2275],
            y=[2275],
            train_mask=[2275],
            val_mask=[2275],
            test_mask=[2275],
            n_id=[2275],
            num_sampled_nodes=[3],
            input_id=[128],
            batch_size=128,
        },
        author={
            num_nodes=2541,
            n_id=[2541],
            num_sampled_nodes=[3],
        },
        institution={
            num_nodes=0,
            n_id=[0],
            num_sampled_nodes=[3],
        },
        field_of_study={
            num_nodes=0,
            n_id=[0],
            num_sampled_nodes=[3],
        },
        (author, affiliated_with, institution)={
            edge_index=[2, 0],
            e_id=[0],
            num_sampled_edges=[2],
        },
        (author, writes, paper)={
            edge_index=[2, 3255],
            e_id=[3255],
            num_sampled_edges=[2],
        },
        (paper, cites, paper)={
            edge_index=[2, 2691],
            e_id=[2691],
            num_sampled_edges=[2],
        },
        (paper, has_topic, field_of_study)={
            edge_index=[2, 0],
            e_id=[0],
            num_sampled_edges=[2],
        }
        )
    print(sampled_hetero_data['paper'].num_sampled_nodes)
    >>> [128, 508, 1598] # Number of sampled nodes per iteration/layer for 'paper' node type

    print(sampled_hetero_data['author', 'writes', 'paper'].num_sampled_edges)
    >>>> [629, 2621] # Number of sampled edges per iteration/layer for 'author_writes_paper' edge type


Returned by NeighborLoader :obj:`num_sampled_nodes` and :obj:`num_sampled_edges` fields can be used by :obj:`trim_to_layer` utils function.

Example of using :obj:`trim_to_layer` feature:
Basic Training Example: leveraging new metadata returned by neighbor loader 

The neighbor loader returns the metadata useful for the HGAM trimming action. 
To train with HGAM is just enough to pass those metadata to any of the native :pyg:`PyG` models (which extend from BasicGNN class), to activate the HGAM computation reduction:


.. code-block:: python

    model = GraphSAGE(
        dataset.num_features,
        hidden_channels=64,
        out_channels=dataset.num_classes,
        num_layers=3,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def train(trim=False):
        for batch in tqdm(loader):
            optimizer.zero_grad()
            batch = batch.to(device)

            if not trim:
                out = model(batch.x, batch.edge_index)
            else:
                out = model(
                    batch.x,
                    batch.edge_index,
                    num_sampled_nodes_per_hop=batch.num_sampled_nodes,
                    num_sampled_edges_per_hop=batch.num_sampled_edges,
                )

            out = out[:batch.batch_size]
            y = batch.y[:batch.batch_size]

            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()

    print('One epoch training without Hierarchical Graph Sampling:')
    train(trim=False)

    print('One epoch training with Hierarchical Graph Sampling:')
    train(trim=True)

    cd examples
    python hierarchical_sampling.py
    >>> One epoch training without Hierarchical Graph Sampling:
    >>> 100%|██████████| 150/150 [01:45<00:00,  1.43it/s]

    >>> One epoch training with Hierarchical Graph Sampling:
    >>> 100%|██████████| 150/150 [01:05<00:00,  2.29it/s]




