import torch


class LineGraph(object):
    r"""Creates a line graph for a given graph

    """

    def __call__(self, data):
        data.x = data.edge_attr
        data.edge_attr = None

        if data.is_directed():
            beg_index = 0
            new_edge_index = torch.tensor([], dtype=data.edge_index.dtype)
            for node_beg in data.edge_index.t():
                end_array = (data.edge_index[0, :] ==
                             node_beg[1]).nonzero().squeeze(dim=1)
                if(end_array.size()[0]):
                    for node_end_index in end_array:
                        to_cat = torch.tensor([[beg_index], [node_end_index]])
                        new_edge_index = torch.cat((new_edge_index, to_cat), 1)
                beg_index += 1
            data.edge_index = new_edge_index

        else:
            beg_index = 0
            new_edge_index = torch.tensor([], dtype=data.edge_index.dtype)
            for i in range(data.edge_index.shape[1]):
                if i % 2:
                    continue
                beg_index = i
                for j in range(data.edge_index[:, i+2:].shape[1]):
                    if j % 2:
                        continue
                    end_index = i+2+j
                    data_beg = data.edge_index[:, beg_index]
                    data_end = data.edge_index[:, end_index]
                    if(data_beg[0] == data_end[0] or
                       data_beg[0] == data_end[1] or
                       data_beg[1] == data_end[0] or
                       data_beg[1] == data_end[1]):
                        to_cat = torch.tensor([[beg_index/2], [end_index/2]],
                                              dtype=data.edge_index.dtype)
                        new_edge_index = torch.cat((new_edge_index, to_cat), 1)
                        to_cat = to_cat.flip(0)
                        new_edge_index = torch.cat((new_edge_index, to_cat), 1)

            data.edge_index = new_edge_index
        return data

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)
