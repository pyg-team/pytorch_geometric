import torch


class AddTrainValTestMask(object):
    r"""Adds train_mask, val_mask, test_mask for node classification.

    Args:
        split (string): The type of dataset split (:obj:`"train-rest"`,
            :obj:`"test-rest"`, :obj:`"random"`).
            If set to :obj:`"train-rest"`, all nodes except those in the
            validation and test sets will be used for training (as in the
            `"FastGCN: Fast Learning with Graph Convolutional Networks via
            Importance Sampling" <https://arxiv.org/abs/1801.10247>`_ paper).
            If set to :obj:`"test-rest"`, all nodes except those in the
            training and validation sets will be used for test (as in the
            `"Pitfalls of Graph Neural Network Evaluation"
            <https://arxiv.org/abs/1811.05868>`_ paper).
            If set to :obj:`"random"`, train, validation, and test sets will be
            randomly generated, according to :obj:`num_train_per_class`,
            :obj:`num_val` and :obj:`num_test`.
        num_train_per_class (int, optional): The number of training samples
            per class in case of :obj:`"test-rest"` and :obj:`"random"` split.
            (default: :obj:`20`)
        num_val (int or float, optional): The number of validation samples.
            If float, it represents the ratio of samples to include in the
            validation set. (default: :obj:`500`)
        num_test (int or float, optional): The number of test samples in case
            of :obj:`"train-rest"` and :obj:`"random"` split. If float, it
            represents the ratio of samples to include in the test set.
            (default: :obj:`1000`)
    """
    def __init__(self, split: str, num_train_per_class: int = 20,
                 num_val: int or float = 500,
                 num_test: int or float = 1000):
        assert split in ['train-rest', 'test-rest', 'random']
        self.split = split
        self.num_train_per_class = num_train_per_class
        self.num_val = num_val
        self.num_test = num_test

    def __call__(self, data):

        N = data.x.size(0)
        data.train_mask = torch.zeros(N, dtype=torch.bool)
        data.val_mask = torch.zeros(N, dtype=torch.bool)
        data.test_mask = torch.zeros(N, dtype=torch.bool)

        if isinstance(self.num_val, float):
            num_val = int(N * self.num_val)
        else:
            num_val = self.num_val

        if isinstance(self.num_test, float):
            num_test = int(N * self.num_test)
        else:
            num_test = self.num_test

        if self.split == 'train-rest':
            idx = torch.randperm(N)
            data.val_mask[idx[:num_val]] = True
            data.test_mask[idx[num_val:num_val + num_test]] = True
            data.train_mask.fill_(True)
            data.train_mask[data.val_mask | data.test_mask] = False

        else:
            num_classes = data.y.max().item() + 1
            for c in range(num_classes):
                idx = (data.y == c).nonzero(as_tuple=False).view(-1)
                idx = idx[torch.randperm(idx.size(0))]
                data.train_mask[idx[:self.num_train_per_class]] = True

            remaining = (~data.train_mask).nonzero(as_tuple=False).view(-1)
            remaining = remaining[torch.randperm(remaining.size(0))]

            data.val_mask[remaining[:num_val]] = True
            if self.split == 'test-rest':
                data.test_mask[remaining[num_val:]] = True
            elif self.split == 'random':
                data.test_mask[remaining[num_val:num_val + num_test]] = True

        return data

    def __repr__(self):
        return '{}(split={})'.format(self.__class__.__name__, self.split)
