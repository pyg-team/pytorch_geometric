from torch_geometric.transforms import BaseTransform


class GetzsTrainMask(BaseTransform):
    r"""Remove the unseen classes from training data
    to get the zero-shot (i.e., completely-imbalanced) label setting

    Args:
        removed_classes (set): The unseen classes. (default: :obj:`{}`)
    """
    def __init__(self, removed_classes={}):
        self.removed_classes = removed_classes

    def __call__(self, data):
        train_mask_zs = data.train_mask.clone()
        for i in range(train_mask_zs.numel()):
            if train_mask_zs[i] == 1:
                if data.y[i].item() in self.removed_classes:
                    train_mask_zs[i] = 0
        data.train_mask = train_mask_zs
        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.removed_classes)
