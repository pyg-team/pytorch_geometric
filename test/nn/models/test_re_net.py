import pickle

import torch

from torch_geometric.datasets.icews import EventDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RENet
from torch_geometric.testing import is_full_test


class MyTestEventDataset(EventDataset):
    def __init__(self, root, seq_len):
        super().__init__(root, pre_transform=RENet.pre_transform(seq_len))
        self.load(self.processed_paths[0])

    @property
    def num_nodes(self):
        return 16

    @property
    def num_rels(self):
        return 8

    @property
    def processed_file_names(self):
        return 'data.pt'

    def _download(self):
        pass

    def process_events(self):
        sub = torch.randint(self.num_nodes, (64, ), dtype=torch.long)
        rel = torch.randint(self.num_rels, (64, ), dtype=torch.long)
        obj = torch.randint(self.num_nodes, (64, ), dtype=torch.long)
        t = torch.arange(8, dtype=torch.long).view(-1, 1).repeat(1, 8).view(-1)
        return torch.stack([sub, rel, obj, t], dim=1)

    def process(self):
        data_list = self._process_data_list()
        self.save(data_list, self.processed_paths[0])


def test_re_net(tmp_path):
    dataset = MyTestEventDataset(tmp_path, seq_len=4)
    loader = DataLoader(dataset, 2, follow_batch=['h_sub', 'h_obj'])

    model = RENet(dataset.num_nodes, dataset.num_rels, hidden_channels=16,
                  seq_len=4)

    if is_full_test():
        jit = torch.jit.export(model)

    logits = torch.randn(6, 6)
    y = torch.tensor([0, 1, 2, 3, 4, 5])

    mrr, hits1, hits3, hits10 = model.test(logits, y)
    assert 0.15 < mrr <= 1
    assert hits1 <= hits3 and hits3 <= hits10 and hits10 == 1

    for data in loader:
        log_prob_obj, log_prob_sub = model(data)
        if is_full_test():
            log_prob_obj_jit, log_prob_sub_jit = jit(data)
            assert torch.allclose(log_prob_obj_jit, log_prob_obj)
            assert torch.allclose(log_prob_sub_jit, log_prob_sub)
        model.test(log_prob_obj, data.obj)
        model.test(log_prob_sub, data.sub)


def test_renet_pre_transform_pickle():
    from torch_geometric.data import Data

    transform = RENet.pre_transform(seq_len=2)
    transform(Data(sub=0, rel=0, obj=1, t=0))
    restored = pickle.loads(pickle.dumps(transform))
    assert str(restored) == str(transform) == 'PreTransform(seq_len=2)'
    for sub, obj, t in [(0, 1, 1), (1, 0, 2)]:
        expected = transform(Data(sub=sub, rel=0, obj=obj, t=t))
        actual = restored(Data(sub=sub, rel=0, obj=obj, t=t))
        for key in ['h_sub', 'h_sub_t', 'h_obj', 'h_obj_t']:
            assert torch.equal(expected[key], actual[key])


def test_re_net_worker(tmp_path):
    import multiprocessing

    dataset = MyTestEventDataset(tmp_path, seq_len=4)
    expected = next(
        iter(DataLoader(dataset, 2, follow_batch=['h_sub', 'h_obj'])))
    for context in ['spawn', 'forkserver']:
        if context not in multiprocessing.get_all_start_methods():
            continue
        loader = DataLoader(dataset, 2, follow_batch=['h_sub', 'h_obj'],
                            num_workers=1, multiprocessing_context=context)
        actual = next(iter(loader))
        for key in expected.keys():
            assert torch.equal(expected[key], actual[key])
