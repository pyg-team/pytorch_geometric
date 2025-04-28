import torch
import torch.nn as nn
import torch.nn.functional as F


class BTCN(nn.Module):
    """Neural network block that applies a bidirectional temporal
    convolution to each node of a graph.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 activation='relu', device='cpu'):
        """Args:
        in_channels (int): Number of features per node.
        out_channels (int): Desired number of output features.
        kernel_size (int, optional): Size of the 1D temporal kernel.
            Default is 3.
        activation (str, optional): Activation function to use ('relu' or
            'sigmoid'). Default is 'relu'.
        device (str, optional): Device to run the computation on.
            Default is 'cpu'.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.activation = activation
        self.device = device

        # Define temporal convolution layers for forward
        # and backward directions
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

        self.conv1b = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2b = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3b = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """Forward pass of the bidirectional temporal convolution.

        Args:
            X (torch.Tensor): Input data of shape
                (batch_size, num_timesteps, num_nodes).

        Returns:
            torch.Tensor: Output data of shape
                (batch_size, num_timesteps, num_features).
        """
        # batch_size, seq_len, num_nodes = X.size()

        batch_size = X.shape[0]
        seq_len = X.shape[1]
        # Expand dimensions for convolution
        Xf = X.unsqueeze(1)  # (batch_size, 1, num_timesteps, num_nodes)

        # Reverse time for backward direction
        inv_idx = torch.arange(Xf.size(2) - 1, -1, -1, device=self.device)
        Xb = Xf.index_select(2, inv_idx)  # Reverse the time dimension

        # Permute dimensions for temporal convolution
        Xf = Xf.permute(0, 1, 3,
                        2)  # (batch_size, num_nodes, 1, num_timesteps)
        Xb = Xb.permute(0, 1, 3,
                        2)  # (batch_size, num_nodes, 1, num_timesteps)

        # Apply forward direction convolutions
        tempf = self.conv1(Xf) * torch.sigmoid(self.conv2(Xf))
        outf = tempf + self.conv3(Xf)
        outf = outf.reshape(batch_size, seq_len - self.kernel_size + 1,
                            self.out_channels)

        # Apply backward direction convolutions
        tempb = self.conv1b(Xb) * torch.sigmoid(self.conv2b(Xb))
        outb = tempb + self.conv3b(Xb)
        outb = outb.reshape(batch_size, seq_len - self.kernel_size + 1,
                            self.out_channels)

        # Reconstruct time padding
        rec = torch.zeros(batch_size, self.kernel_size - 1, self.out_channels,
                          device=self.device)
        outf = torch.cat((outf, rec), dim=1)
        outb = torch.cat((outb, rec), dim=1)

        # Reverse backward output to align with forward
        inv_idx = torch.arange(outb.size(1) - 1, -1, -1, device=self.device)
        outb = outb.index_select(1, inv_idx)

        # Combine forward and backward outputs
        out = outf + outb
        if self.activation == 'relu':
            out = F.relu(outf) + F.relu(outb)
        elif self.activation == 'sigmoid':
            out = F.sigmoid(outf) + F.sigmoid(outb)

        return out
