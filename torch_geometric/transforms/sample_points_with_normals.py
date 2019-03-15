import torch
import math

class SamplePointsWithNormals(object):
    def __init__(self, num, normalLength=1):
        self.num = num
        self.targetLen = normalLength

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
        
        vec1 = pos[face[1]] - pos[face[0]]
        vec2 = pos[face[2]] - pos[face[0]]
        norm = vec1.cross(vec2)

        sumLen = torch.div(torch.sqrt(torch.sum(norm**2, dim=1)), self.targetLen)
        
        norm_x = torch.div(norm[:, :1].reshape(self.num), sumLen)
        norm_y = torch.div(norm[:, 1:2].reshape(self.num), sumLen)
        norm_z = torch.div(norm[:, 2:3].reshape(self.num), sumLen)       

        norm[:, :1] = norm_x.reshape(self.num, 1)
        norm[:, 1:2] = norm_y.reshape(self.num, 1)
        norm[:, 2:3] = norm_z.reshape(self.num, 1)               
        data.n = norm

        pos_sampled += frac[:, :1] * (pos[face[1]] - pos[face[0]])
        pos_sampled += frac[:, 1:] * (pos[face[2]] - pos[face[0]])

        data.pos = pos_sampled
        data.face = None        
        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.num)
