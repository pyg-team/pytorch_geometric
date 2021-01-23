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
        num_val (int, optional): The number of validation samples.
            (default: :obj:`500`)
        num_test (int, optional): The number of test samples in case of
            :obj:`"train-rest"` and :obj:`"random"` split.
            (default: :obj:`1000`)
        num_classes (int, optional): The number of classes. This parameter
            should be set in case of :obj:`"test-rest"` and :obj:`"random"`
            split. (default: :obj:`None`)
    """
    def __init__(self, split, num_train_per_class=20, num_val=500,
                 num_test=1000, num_classes=None):
        assert split in ['train-rest', 'test-rest', 'random']
        assert split == 'train-rest' or num_classes is not None, \
            'num_classes should be set in case of test-rest and random split'
        self.split = split
        self.num_classes = num_classes
        self.num_train_per_class = num_train_per_class
        self.num_val = num_val
        self.num_test = num_test

    def __call__(self, data):

        N = data.x.size(0)
        data.train_mask = torch.zeros(N, dtype=torch.bool)
        data.val_mask = torch.zeros(N, dtype=torch.bool)
        data.test_mask = torch.zeros(N, dtype=torch.bool)

        num_val_and_test = self.num_val + self.num_test

        if self.split == 'train-rest':
            idx = torch.randperm(N)
            data.val_mask[idx[:self.num_val]] = True
            data.test_mask[idx[self.num_val:num_val_and_test]] = True
            data.train_mask.fill_(True)
            data.train_mask[data.val_mask | data.test_mask] = False

        else:
            for c in range(self.num_classes):
                idx = (data.y == c).nonzero(as_tuple=False).view(-1)
                idx = idx[torch.randperm(idx.size(0))]
                data.train_mask[idx[:self.num_train_per_class]] = True

            remaining = (~data.train_mask).nonzero(as_tuple=False).view(-1)
            remaining = remaining[torch.randperm(remaining.size(0))]

            data.val_mask[remaining[:self.num_val]] = True
            if self.split == 'test-rest':
                data.test_mask[remaining[self.num_val:]] = True
            elif self.split == 'random':
                data.test_mask[remaining[self.num_val:num_val_and_test]] = True

        return data

    def __repr__(self):
        return '{}(split={})'.format(self.__class__.__name__, self.split)
