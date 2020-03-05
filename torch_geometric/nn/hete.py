import copy

import torch
from torch._six import container_abcs

from torch_geometric.nn.inits import reset


class HeteConv(torch.nn.Module):
    def __init__(self, convs, aggr='add', parallelize=False):
        super(HeteConv, self).__init__()

        assert isinstance(convs, container_abcs.Mapping)
        self.convs = convs
        self.modules = torch.nn.ModuleList(convs.values())

        assert aggr in ['add', 'mean', 'max', 'mul', 'concat', None]
        self.aggr = aggr

        if parallelize and torch.cuda.is_available():
            self.streams = {key: torch.cuda.Stream() for key in convs.keys()}
        else:
            self.streams = None

    def reset_parameters(self):
        for conv in self.convs.values():
            reset(conv)

    def forward(self, node_dict, edge_dict):
        out_edge_dict = {}

        for edge_key, edge_info in edge_dict.items():
            head, _, tail = edge_key
            conv = self.convs[edge_key]

            # Gather related edge information into `input_dict`.
            input_dict = copy.copy(edge_info)

            # Distinguish between inter- and inter-message passing and gather
            # related node information into `input_dict`.
            if head == tail:
                input_dict.update(node_dict[head])
            else:
                head_info = node_dict[head]
                tail_info = node_dict[tail]

                node_keys = list(head_info.keys()) + list(tail_info.keys())
                input_dict.update({
                    key: (head_info.get(key), tail_info.get(key))
                    for key in set(node_keys)
                })

                head_key = list(head_info.keys())[0]
                tail_key = list(tail_info.keys())[0]

                input_dict['size'] = (head_info[head_key].size(conv.node_dim),
                                      tail_info[tail_key].size(conv.node_dim))

            is_cuda = False
            for item in input_dict.values():
                if hasattr(item, 'is_cuda'):
                    is_cuda = item.is_cuda
                    break

            # Perform message passing.
            if self.streams is not None and is_cuda:
                with torch.cuda.stream(self.streams[edge_key]):
                    out_edge_dict[edge_key] = conv(**input_dict)
            else:
                out_edge_dict[edge_key] = conv(**input_dict)

        if self.streams is not None and is_cuda:
            torch.cuda.synchronize()

        # Group `out_edge_dict` to output node embeddings based on `tail`.
        out_node_dict = {tail: [] for _, _, tail in edge_dict.keys()}

        for (_, _, tail), item in out_edge_dict.items():
            out_node_dict[tail] += [item]

        # Aggregate multiple embeddings that share the same tail.
        for key in sorted(out_node_dict.keys()):
            if len(item) == 1:
                out_node_dict[key] = out_node_dict[key][0]
            else:
                out_node_dict[key] = self.aggregate(key, out_node_dict[key])

        return out_node_dict

    def aggregate(self, node_type, xs):
        if self.aggr == 'concat':
            return torch.cat(xs, dim=-1)

        x = torch.stack(xs, dim=-1)
        if self.aggr == 'add':
            return x.sum(dim=-1)
        elif self.aggr == 'mean':
            return x.mean(dim=-1)
        elif self.aggr == 'max':
            return x.max(dim=-1)[0]
        elif self.aggr == 'mul':
            return x.prod(dim=-1)[0]
