from __future__ import division


class NormalizeScale(object):
    def __call__(self, data):
        pos = data.pos
        pos = pos - pos.min(dim=0,keepdim=True)[0]
        pos /= pos.max()
        pos += 0.5*(1-pos.max(0,keepdim=True)[0])
        data.pos = pos*0.9999
        return data
