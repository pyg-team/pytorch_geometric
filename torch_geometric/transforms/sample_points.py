import torch


class SamplePoints(object):
    def __init__(self, num):
        self.num = num

    def __call__(self, data):
        pos, face = data.pos, data.face
        assert not pos.is_cuda and not face.is_cuda
        assert pos.size(1) == 3 and face.size(0) == 3

        area = (pos[face[1]] - pos[face[0]]).cross(pos[face[2]] - pos[face[0]])
        area = torch.sqrt((area**2).sum(dim=-1)) / 2

        prob = area / area.sum()
        sample = torch.multinomial(prob, self.num, replacement=True)
        face = face[:, sample]

        frac = torch.rand(self.num, 2)
        mask = frac.sum(dim=-1) > 1
        frac[mask] = 1 - frac[mask]

        pos_sampled = pos[face[0]]
        pos_sampled += frac[:, :1] * (pos[face[1]] - pos[face[0]])
        pos_sampled += frac[:, 1:] * (pos[face[2]] - pos[face[0]])

        data.pos = pos_sampled
        data.face = None
        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.num)
