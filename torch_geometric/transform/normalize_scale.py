from __future__ import division


class NormalizeScale(object):
    def __call__(self, data):
        pos = data.pos
        data.offset = -pos.min(dim=0,keepdim=True)[0]
        pos += data.offset
        data.scale = 1/pos.max()
        pos *= data.scale
        to_mid = 0.5*(1-pos.max(0,keepdim=True)[0])
        pos += to_mid
        data.offset += to_mid/data.scale
        data.pos = pos
        return data
