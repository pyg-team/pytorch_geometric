import torch


def init_weights(m):
    r"""
    Performs weight initialization

    Args:
        m (nn.Module): PyTorch module

    """
    if (isinstance(m, torch.nn.BatchNorm2d)
            or isinstance(m, torch.nn.BatchNorm1d)):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, torch.nn.Linear):
        m.weight.data = torch.nn.init.xavier_uniform_(
            m.weight.data, gain=torch.nn.init.calculate_gain('relu'))
        if m.bias is not None:
            m.bias.data.zero_()
