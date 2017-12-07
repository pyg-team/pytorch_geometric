import torch


class FlatPosition(object):
    def __call__(self, data):
        input, position = data.input, data.position

        data.position = position[:, :-1]

        if input is None:
            data.input = position[:, -1:]
        else:
            data.input = torch.cat([input, position[:, -1:]], dim=1)

        return data
