import torch


class Set2Set(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 processing_steps,
                 num_layers=1):
        super(Set2Set, self).__init__()

        self.in_channels = in_channels
        self.out_channels = 2 * in_channels
        self.hidden_channels = in_channels
        self.processing_steps = processing_steps
        self.num_layers = num_layers

        self.lstm = torch.nn.LSTM(self.out_channels, hidden_channels,
                                  num_layers)

        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()

    def forward(self, x, batch):
        batch_size = batch.max().item() + 1

        # Bring x into shape [batch_size, max_nodes, in_channels].
        xs = x.split(torch.bincount(batch).tolist())
        max_nodes = max([t.size(0) for t in xs])
        xs = [[t, t.new_zeros(max_nodes - t.size(0), t.size(1))] for t in xs]
        xs = [torch.cat(t, dim=0) for t in xs]
        x = torch.stack(xs, dim=0)

        h = (x.new_zeros((self.num_layers, batch_size, self.hidden_channels)),
             x.new_zeros((self.num_layers, batch_size, self.hidden_channels)))
        q_star = x.new_zeros(1, batch_size, self.out_channels)

        for i in range(self.processing_steps):
            q, h = self.lstm(q_star, h)
            q = q.view(batch_size, 1, self.in_channels)
            e = (x * q).sum(dim=-1)  # Dot product.
            a = torch.softmax(e, dim=-1)
            a = a.view(batch_size, max_nodes, 1)
            r = (a * x).sum(dim=1, keepdim=True)
            q_star = torch.cat([q, r], dim=-1)
            q_star = q_star.view(1, batch_size, self.out_channels)

        q_star = q_star.view(batch_size, self.out_channels)
        return q_star

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
