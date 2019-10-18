from torch.nn import BatchNorm1d


class BatchNorm(BatchNorm1d):
    def __init__(self, channels, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(BatchNorm, self).__init__(channels, eps, momentum, affine,
                                        track_running_stats)

    def forward(self, x):
        return super(BatchNorm, self).forward(x)

    def __repr__(self):
        return ('{}({}, eps={}, momentum={}, affine={}, '
                'track_running_stats={})').format(self.__class__.__name__,
                                                  self.num_features, self.eps,
                                                  self.momentum, self.affine,
                                                  self.track_running_stats)
