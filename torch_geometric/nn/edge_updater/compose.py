from inspect import Parameter, Signature
from typing import Iterable, Dict

import torch

from torch_geometric.nn.conv.utils.inspector import Inspector
from torch_geometric.nn.edge_updater import EdgeUpdater


def merge_signatures(srcs):
    parameters = {}
    for src in srcs:
        signature = Signature.from_callable(src, follow_wrapped=True)

        assert signature.return_annotation in [dict, Dict]
        parameters.update(signature.parameters)

    parameters = sorted(parameters.values(), key=lambda x: x.default != Parameter.empty)
    return Signature(parameters, return_annotation=Dict)


def copy_signature(signature, to):
    def wrapper(*args, **kwargs):
        return to(*args, **kwargs)
    wrapper.__signature__ = signature
    return wrapper


class Compose(EdgeUpdater):
    r"""TODO
    """

    def __init__(self, updaters: Iterable[EdgeUpdater]):
        r"""TODO
        """
        super().__init__()

        self.updaters = torch.nn.ModuleList(updaters)
        self.inspectors = [Inspector(u) for u in self.updaters]
        for module, inspector in zip(self.updaters, self.inspectors):
            inspector.inspect(module.forward)

        self._signature = merge_signatures(
            [u.forward for u in self.updaters]
        )

    def reset_parameters(self) -> None:
        for module in self.updaters:
            module.reset_parameters()

    def forward(self, **kwargs) -> Dict:
        output = {}

        for updater, inspector in zip(self.updaters, self.inspectors):
            updater_kwargs = inspector.distribute('forward', kwargs)
            updater_ouptut = updater(**updater_kwargs)
            output.update(updater_ouptut)
            kwargs.update(updater_ouptut)

        return output

    def __getattribute__(self, item):
        value = object.__getattribute__(self, item)
        if item == 'forward':
            value = copy_signature(signature=self._signature, to=value)
        return value

    def __repr__(self):
        upd_names = ', '.join([x.__repr__() for x in self.updaters])
        return f'{self.__class__.__name__}([{upd_names}])'


if __name__ == '__main__':  # TODO REMOVE
    from torch_geometric.nn.edge_updater import Fill, GCNNorm
    import inspect

    edge_weight = torch.randn(5)
    print(edge_weight)
    updater = Compose([
        Fill(2.), Fill(3), GCNNorm()
    ])
    print(inspect.signature(updater.forward))

    # x = updater(edge_weight=edge_weight)
    # print(x)
