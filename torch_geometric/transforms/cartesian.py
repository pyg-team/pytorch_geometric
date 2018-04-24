import torch


class Cartesian(object):
    def __init__(self, cat=True):
        self.cat = cat

    def __call__(self, data):
        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr

        cartesian = pos[col] - pos[row]
        cartesian /= 2 * cartesian.abs().max()
        cartesian += 0.5

        if pseudo is not None and self.cat:
            pseudo = pseudo.unsqueeze(-1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, cartesian], dim=-1)
        else:
            data.edge_attr = cartesian

        return data

    def __repr__(self):
        return '{}(cat={})'.format(self.__class__.__name__, self.cat)
