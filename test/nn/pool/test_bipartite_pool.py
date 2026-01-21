import torch

from torch_geometric import nn


def test_bipartite_pooling():
    num_nodes = 100
    ratio = 10
    in_channels = 5
    out_channels = 8
    num_graphs = 4
    kw = dict(in_channels=in_channels, out_channels=out_channels)

    gnnlist = [
        nn.GINConv(torch.nn.Linear(in_channels, out_channels)),
        # nn.GENConv(**kw), # gradient test breaks
        nn.GeneralConv(**kw),
        nn.GraphConv(**kw),
        nn.MFConv(**kw),
        # nn.SimpleConv(), # gradient test breaks
        nn.SAGEConv(**kw),
        nn.WLConvContinuous(),
        nn.GATv2Conv(add_self_loops=False, **kw),
        nn.GATConv(add_self_loops=False, **kw),
    ]
    batch = torch.arange(num_graphs).repeat_interleave(num_nodes)
    for gnn in gnnlist:

        pool = nn.BipartitePooling(in_channels, ratio=ratio, gnn=gnn)
        # make sure pool.seed_nodes is != 0
        # otherwise the grad is sometimes too close to 0
        pool.seed_nodes.data.zero_()

        x = torch.randn((num_graphs * num_nodes, in_channels)).requires_grad_()
        x.retain_grad()
        out, new_batchidx = pool(x, batch)

        if isinstance(gnn, (nn.SimpleConv, nn.WLConvContinuous)):
            assert out.shape == torch.Size([num_graphs * ratio, in_channels])
        else:
            assert out.shape == torch.Size([num_graphs * ratio, out_channels])

        for grad_graph in range(num_graphs):

            out[new_batchidx == grad_graph].sum().backward(retain_graph=True)
            # only graph igraph gets a gradient
            for check_graph in range(num_graphs):
                grad_grap_i = x.grad[batch == check_graph].abs().sum(1)
                if grad_graph == check_graph:
                    assert (grad_grap_i > 0).all()
                else:
                    assert (grad_grap_i == 0).all()

            x.grad.zero_()
            # all seed nodes get a gradient
            assert (pool.seed_nodes.grad.abs().sum(1) > 0).all()
            pool.seed_nodes.grad.zero_()
