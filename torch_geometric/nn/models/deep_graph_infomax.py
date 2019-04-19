import torch
from torch.nn import Parameter
from sklearn.linear_model import LogisticRegression

from ..inits import reset, uniform

EPS = 1e-15


class DeepGraphInfomax(torch.nn.Module):
    def __init__(self, hidden_channels, encoder, summary, corruption):
        super(DeepGraphInfomax, self).__init__()
        self.hidden_channels = hidden_channels
        self.encoder = encoder
        self.summary = summary
        self.corruption = corruption

        self.weight = Parameter(torch.Tensor(hidden_channels, hidden_channels))

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.summary)
        uniform(self.hidden_channels, self.weight)

    def forward(self, *args, **kwargs):
        pos_z = self.encoder(*args, **kwargs)
        cor = self.corruption(*args, **kwargs)
        cor = cor if isinstance(cor, tuple) else (cor, )
        neg_z = self.encoder(*cor)
        summary = self.summary(pos_z, *args, **kwargs)
        return pos_z, neg_z, summary

    def discriminate(self, z, summary, sigmoid=True):
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        return torch.sigmoid(value) if sigmoid else value

    def loss(self, pos_z, neg_z, summary):
        pos_loss = -torch.log(
            self.discriminate(pos_z, summary, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(
            1 - self.discriminate(neg_z, summary, sigmoid=True) + EPS).mean()

        return pos_loss + neg_loss

    def test(self,
             train_z,
             train_y,
             test_z,
             test_y,
             solver='lbfgs',
             multi_class='auto',
             *args,
             **kwargs):
        clf = LogisticRegression(
            solver=solver, multi_class=multi_class, *args, **kwargs).fit(
                train_z.detach().cpu().numpy(),
                train_y.detach().cpu().numpy())
        return clf.score(test_z.detach().cpu().numpy(),
                         test_y.detach().cpu().numpy())

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.hidden_channels)
