import torch


class FlatPosition(object):
    def __call__(self, data):
        input, position = data.input, data.position

        data.position = position[:, :-1].contiguous()
        rest = position[:, -1:].contiguous()

        if input is None:
            data.input = rest
        else:
            data.input = torch.cat([input, rest], dim=1)

        return data
