import torch


class HandleNodeAttention(object):
    r"""Reads node attention from node features for certain datasets, such as
    COLORS and TRIANGLES.

    .. note::
        Node attention should be stored as the first feature
        in the data.x object.

    """

    def __call__(self, data):
        if data.x.dim() == 1:
            data.x = data.x.unsqueeze(-1)
        data.node_attention = torch.softmax(data.x[:, 0], dim=0)
        if data.x.shape[1] > 1:
            data.x = data.x[:, 1:]
        else:
            # not supposed to use node attention as node features,
            # because it is typically not available in the val/test set
            data.x = None

        return data
