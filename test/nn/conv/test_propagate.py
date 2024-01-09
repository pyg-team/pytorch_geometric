from torch_geometric.nn.conv.propagate import module_from_template


def test_propagate1():
    script = module_from_template('torch_geometric.nn.conv.gat_conv_propagate',
                                  'torch_geometric/nn/conv/propagate.jinja')
    print(script.Propagate)


def test_propagate2():
    script = module_from_template('torch_geometric.nn.conv.gat_conv_propagate',
                                  'torch_geometric/nn/conv/propagate.jinja')
    print(script.Propagate)
