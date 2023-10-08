#reference from https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/torch_geometric_temporal/nn/attention/gman.py

import math
from typing import Union, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2D(nn.Module):
    r"""An implementation of the 2D-convolution block.
    For details see this paper: `"GMAN: A Graph Multi-Attention Network for Traffic Prediction."
    <https://arxiv.org/pdf/1911.08415.pdf>`_

    Args:
        input_dims (int): Dimension of input.
        output_dims (int): Dimension of output.
        kernel_size (tuple or list): Size of the convolution kernel.
        stride (tuple or list, optional): Convolution strides, default (1,1).
        use_bias (bool, optional): Whether to use bias, default is True.
        activation (Callable, optional): Activation function, default is torch.nn.functional.relu.
        bn_decay (float, optional): Batch normalization momentum, default is None.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        kernel_size: Union[tuple, list],
        stride: Union[tuple, list] = (1, 1),
        use_bias: bool = True,
        activation: Optional[Callable[[torch.FloatTensor], torch.FloatTensor]] = F.relu,
        bn_decay: Optional[float] = None,
    ):
        super(Conv2D, self).__init__()
        self._activation = activation
        self._conv2d = nn.Conv2d(
            input_dims,
            output_dims,
            kernel_size,
            stride=stride,
            padding=0,
            bias=use_bias,
        )
        self._batch_norm = nn.BatchNorm2d(output_dims, momentum=bn_decay)
        torch.nn.init.xavier_uniform_(self._conv2d.weight)

        if use_bias:
            torch.nn.init.zeros_(self._conv2d.bias)

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass of the 2D-convolution block.

        Arg types:
            * **X** (PyTorch Float Tensor) - Input tensor, with shape (batch_size, num_his, num_nodes, input_dims).

        Return types:
            * **X** (PyTorch Float Tensor) - Output tensor, with shape (batch_size, num_his, num_nodes, output_dims).
        """
        X = X.permute(0, 3, 2, 1)
        X = self._conv2d(X)
        X = self._batch_norm(X)
        if self._activation is not None:
            X = self._activation(X)
        return X.permute(0, 3, 2, 1)


class FullyConnected(nn.Module):
    r"""An implementation of the fully-connected layer.
    For details see this paper: `"GMAN: A Graph Multi-Attention Network for Traffic Prediction."
    <https://arxiv.org/pdf/1911.08415.pdf>`_

    Args:
        input_dims (int or list): Dimension(s) of input.
        units (int or list): Dimension(s) of outputs in each 2D convolution block.
        activations (Callable or list): Activation function(s).
        bn_decay (float, optional): Batch normalization momentum, default is None.
        use_bias (bool, optional): Whether to use bias, default is True.
    """

    def __init__(
        self,
        input_dims: Union[int, list],
        units: Union[int, list],
        activations: Union[Callable[[torch.FloatTensor], torch.FloatTensor], list],
        bn_decay: float,
        use_bias: bool = True,
    ):
        super(FullyConnected, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        assert type(units) == list
        self._conv2ds = nn.ModuleList(
            [
                Conv2D(
                    input_dims=input_dim,
                    output_dims=num_unit,
                    kernel_size=[1, 1],
                    stride=[1, 1],
                    use_bias=use_bias,
                    activation=activation,
                    bn_decay=bn_decay,
                )
                for input_dim, num_unit, activation in zip(
                    input_dims, units, activations
                )
            ]
        )

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass of the fully-connected layer.

        Arg types:
            * **X** (PyTorch Float Tensor) - Input tensor, with shape (batch_size, num_his, num_nodes, 1).

        Return types:
            * **X** (PyTorch Float Tensor) - Output tensor, with shape (batch_size, num_his, num_nodes, units[-1]).
        """
        for conv in self._conv2ds:
            X = conv(X)
        return X


class SpatioTemporalEmbedding(nn.Module):
    r"""An implementation of the spatial-temporal embedding block.
    For details see this paper: `"GMAN: A Graph Multi-Attention Network for Traffic Prediction."
    <https://arxiv.org/pdf/1911.08415.pdf>`_

    Args:
        D (int) : Dimension of output.
        bn_decay (float): Batch normalization momentum.
        steps_per_day (int): Steps to take for a day.
        use_bias (bool, optional): Whether to use bias in Fully Connected layers, default is True.
    """

    def __init__(
        self, D: int, bn_decay: float, steps_per_day: int, use_bias: bool = True
    ):
        super(SpatioTemporalEmbedding, self).__init__()
        self._fully_connected_se = FullyConnected(
            input_dims=[D, D],
            units=[D, D],
            activations=[F.relu, None],
            bn_decay=bn_decay,
            use_bias=use_bias,
        )

        self._fully_connected_te = FullyConnected(
            input_dims=[steps_per_day + 7, D],
            units=[D, D],
            activations=[F.relu, None],
            bn_decay=bn_decay,
            use_bias=use_bias,
        )

    def forward(
        self, SE: torch.FloatTensor, TE: torch.FloatTensor, T: int
    ) -> torch.FloatTensor:
        """
        Making a forward pass of the spatial-temporal embedding.

        Arg types:
            * **SE** (PyTorch Float Tensor) - Spatial embedding, with shape (num_nodes, D).
            * **TE** (Pytorch Float Tensor) - Temporal embedding, with shape (batch_size, num_his + num_pred, 2).(dayofweek, timeofday)
            * **T** (int) - Number of time steps in one day.

        Return types:
            * **output** (PyTorch Float Tensor) - Spatial-temporal embedding, with shape (batch_size, num_his + num_pred, num_nodes, D).
        """
        SE = SE.unsqueeze(0).unsqueeze(0)
        SE = self._fully_connected_se(SE)
        dayofweek = torch.empty(TE.shape[0], TE.shape[1], 7).to(SE.device)
        timeofday = torch.empty(TE.shape[0], TE.shape[1], T).to(SE.device)
        for i in range(TE.shape[0]):
            dayofweek[i] = F.one_hot(TE[..., 0][i].to(torch.int64) % 7, 7)
        for j in range(TE.shape[0]):
            timeofday[j] = F.one_hot(TE[..., 1][j].to(torch.int64) % T, T)
        TE = torch.cat((dayofweek, timeofday), dim=-1)
        TE = TE.unsqueeze(dim=2)
        TE = self._fully_connected_te(TE)
        del dayofweek, timeofday
        return SE + TE


class SpatialAttention(nn.Module):
    r"""An implementation of the spatial attention mechanism.
    For details see this paper: `"GMAN: A Graph Multi-Attention Network for Traffic Prediction."
    <https://arxiv.org/pdf/1911.08415.pdf>`_

    Args:
        K (int) : Number of attention heads.
        d (int) : Dimension of each attention head outputs.
        bn_decay (float): Batch normalization momentum.
    """

    def __init__(self, K: int, d: int, bn_decay: float):
        super(SpatialAttention, self).__init__()
        D = K * d
        self._d = d
        self._K = K
        self._fully_connected_q = FullyConnected(
            input_dims=2 * D, units=D, activations=F.relu, bn_decay=bn_decay
        )
        self._fully_connected_k = FullyConnected(
            input_dims=2 * D, units=D, activations=F.relu, bn_decay=bn_decay
        )
        self._fully_connected_v = FullyConnected(
            input_dims=2 * D, units=D, activations=F.relu, bn_decay=bn_decay
        )
        self._fully_connected = FullyConnected(
            input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay
        )

    def forward(
        self, X: torch.FloatTensor, STE: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Making a forward pass of the spatial attention mechanism.

        Arg types:
            * **X** (PyTorch Float Tensor) - Input sequence, with shape (batch_size, num_step, num_nodes, K*d).
            * **STE** (Pytorch Float Tensor) - Spatial-temporal embedding, with shape (batch_size, num_step, num_nodes, K*d).

        Return types:
            * **X** (PyTorch Float Tensor) - Spatial attention scores, with shape (batch_size, num_step, num_nodes, K*d).
        """
        batch_size = X.shape[0]
        X = torch.cat((X, STE), dim=-1)
        query = self._fully_connected_q(X)
        key = self._fully_connected_k(X)
        value = self._fully_connected_v(X)
        query = torch.cat(torch.split(query, self._K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self._K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self._K, dim=-1), dim=0)
        attention = torch.matmul(query, key.transpose(2, 3))
        attention /= self._d ** 0.5
        attention = F.softmax(attention, dim=-1)
        X = torch.matmul(attention, value)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)
        X = self._fully_connected(X)
        del query, key, value, attention
        return X


class TemporalAttention(nn.Module):
    r"""An implementation of the temporal attention mechanism.
    For details see this paper: `"GMAN: A Graph Multi-Attention Network for Traffic Prediction."
    <https://arxiv.org/pdf/1911.08415.pdf>`_

    Args:
        K (int) : Number of attention heads.
        d (int) : Dimension of each attention head outputs.
        bn_decay (float): Batch normalization momentum.
        mask (bool): Whether to mask attention score.
    """

    def __init__(self, K: int, d: int, bn_decay: float, mask: bool):
        super(TemporalAttention, self).__init__()
        D = K * d
        self._d = d
        self._K = K
        self._mask = mask
        self._fully_connected_q = FullyConnected(
            input_dims=2 * D, units=D, activations=F.relu, bn_decay=bn_decay
        )
        self._fully_connected_k = FullyConnected(
            input_dims=2 * D, units=D, activations=F.relu, bn_decay=bn_decay
        )
        self._fully_connected_v = FullyConnected(
            input_dims=2 * D, units=D, activations=F.relu, bn_decay=bn_decay
        )
        self._fully_connected = FullyConnected(
            input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay
        )

    def forward(
        self, X: torch.FloatTensor, STE: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Making a forward pass of the temporal attention mechanism.

        Arg types:
            * **X** (PyTorch Float Tensor) - Input sequence, with shape (batch_size, num_step, num_nodes, K*d).
            * **STE** (Pytorch Float Tensor) - Spatial-temporal embedding, with shape (batch_size, num_step, num_nodes, K*d).

        Return types:
            * **X** (PyTorch Float Tensor) - Temporal attention scores, with shape (batch_size, num_step, num_nodes, K*d).
        """
        batch_size = X.shape[0]
        X = torch.cat((X, STE), dim=-1)
        query = self._fully_connected_q(X)
        key = self._fully_connected_k(X)
        value = self._fully_connected_v(X)
        query = torch.cat(torch.split(query, self._K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self._K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self._K, dim=-1), dim=0)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        attention = torch.matmul(query, key)
        attention /= self._d ** 0.5
        if self._mask:
            batch_size = X.shape[0]
            num_step = X.shape[1]
            num_nodes = X.shape[2]
            mask = torch.ones(num_step, num_step).to(X.device)
            mask = torch.tril(mask)
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)
            mask = mask.repeat(self._K * batch_size, num_nodes, 1, 1)
            mask = mask.to(torch.bool)
            condition = torch.FloatTensor([-(2 ** 15) + 1]).to(X.device)
            attention = torch.where(mask, attention, condition)
        attention = F.softmax(attention, dim=-1)
        X = torch.matmul(attention, value)
        X = X.permute(0, 2, 1, 3)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)
        X = self._fully_connected(X)
        del query, key, value, attention
        return X


class GatedFusion(nn.Module):
    r"""An implementation of the gated fusion mechanism.
    For details see this paper: `"GMAN: A Graph Multi-Attention Network for Traffic Prediction."
    <https://arxiv.org/pdf/1911.08415.pdf>`_

    Args:
        D (int) : dimension of output.
        bn_decay (float): batch normalization momentum.
    """

    def __init__(self, D: int, bn_decay: float):
        super(GatedFusion, self).__init__()
        self._fully_connected_xs = FullyConnected(
            input_dims=D, units=D, activations=None, bn_decay=bn_decay, use_bias=False
        )
        self._fully_connected_xt = FullyConnected(
            input_dims=D, units=D, activations=None, bn_decay=bn_decay, use_bias=True
        )
        self._fully_connected_h = FullyConnected(
            input_dims=[D, D],
            units=[D, D],
            activations=[F.relu, None],
            bn_decay=bn_decay,
        )

    def forward(
        self, HS: torch.FloatTensor, HT: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Making a forward pass of the gated fusion mechanism.

        Arg types:
            * **HS** (PyTorch Float Tensor) - Spatial attention scores, with shape (batch_size, num_step, num_nodes, D).
            * **HT** (Pytorch Float Tensor) - Temporal attention scores, with shape (batch_size, num_step, num_nodes, D).

        Return types:
            * **H** (PyTorch Float Tensor) - Spatial-temporal attention scores, with shape (batch_size, num_step, num_nodes, D).
        """
        XS = self._fully_connected_xs(HS)
        XT = self._fully_connected_xt(HT)
        z = torch.sigmoid(torch.add(XS, XT))
        H = torch.add(torch.mul(z, HS), torch.mul(1 - z, HT))
        H = self._fully_connected_h(H)
        del XS, XT, z
        return H


class SpatioTemporalAttention(nn.Module):
    r"""An implementation of the spatial-temporal attention block, with spatial attention and temporal attention
    followed by gated fusion. For details see this paper: `"GMAN: A Graph Multi-Attention Network for Traffic Prediction."
    <https://arxiv.org/pdf/1911.08415.pdf>`_

    Args:
        K (int) : Number of attention heads.
        d (int) : Dimension of each attention head outputs.
        bn_decay (float): Batch normalization momentum.
        mask (bool): Whether to mask attention score in temporal attention.
    """

    def __init__(self, K: int, d: int, bn_decay: float, mask: bool):
        super(SpatioTemporalAttention, self).__init__()
        self._spatial_attention = SpatialAttention(K, d, bn_decay)
        self._temporal_attention = TemporalAttention(K, d, bn_decay, mask=mask)
        self._gated_fusion = GatedFusion(K * d, bn_decay)

    def forward(
        self, X: torch.FloatTensor, STE: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Making a forward pass of the spatial-temporal attention block.

        Arg types:
            * **X** (PyTorch Float Tensor) - Input sequence, with shape (batch_size, num_step, num_nodes, K*d).
            * **STE** (Pytorch Float Tensor) - Spatial-temporal embedding, with shape (batch_size, num_step, num_nodes, K*d).

        Return types:
            * **X** (PyTorch Float Tensor) - Attention scores, with shape (batch_size, num_step, num_nodes, K*d).
        """
        HS = self._spatial_attention(X, STE)
        HT = self._temporal_attention(X, STE)
        H = self._gated_fusion(HS, HT)
        del HS, HT
        X = torch.add(X, H)
        return X


class TransformAttention(nn.Module):
    r"""An implementation of the tranform attention mechanism.
    For details see this paper: `"GMAN: A Graph Multi-Attention Network for Traffic Prediction."
    <https://arxiv.org/pdf/1911.08415.pdf>`_

    Args:
        K (int) : Number of attention heads.
        d (int) : Dimension of each attention head outputs.
        bn_decay (float): Batch normalization momentum.
    """

    def __init__(self, K: int, d: int, bn_decay: float):
        super(TransformAttention, self).__init__()
        D = K * d
        self._K = K
        self._d = d
        self._fully_connected_q = FullyConnected(
            input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay
        )
        self._fully_connected_k = FullyConnected(
            input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay
        )
        self._fully_connected_v = FullyConnected(
            input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay
        )
        self._fully_connected = FullyConnected(
            input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay
        )

    def forward(
        self,
        X: torch.FloatTensor,
        STE_his: torch.FloatTensor,
        STE_pred: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Making a forward pass of the transform attention layer.

        Arg types:
            * **X** (PyTorch Float Tensor) - Input sequence, with shape (batch_size, num_his, num_nodes, K*d).
            * **STE_his** (Pytorch Float Tensor) - Spatial-temporal embedding for history,
            with shape (batch_size, num_his, num_nodes, K*d).
            * **STE_pred** (Pytorch Float Tensor) - Spatial-temporal embedding for prediction,
            with shape (batch_size, num_pred, num_nodes, K*d).

        Return types:
            * **X** (PyTorch Float Tensor) - Output sequence for prediction, with shape (batch_size, num_pred, num_nodes, K*d).
        """
        batch_size = X.shape[0]
        query = self._fully_connected_q(STE_pred)
        key = self._fully_connected_k(STE_his)
        value = self._fully_connected_v(X)
        query = torch.cat(torch.split(query, self._K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self._K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self._K, dim=-1), dim=0)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        attention = torch.matmul(query, key)
        attention /= self._d ** 0.5
        attention = F.softmax(attention, dim=-1)
        X = torch.matmul(attention, value)
        X = X.permute(0, 2, 1, 3)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)
        X = self._fully_connected(X)
        del query, key, value, attention
        return X


class GMAN(nn.Module):
    r"""An implementation of GMAN.
    For details see this paper: `"GMAN: A Graph Multi-Attention Network for Traffic Prediction."
    <https://arxiv.org/pdf/1911.08415.pdf>`_

    Args:
        L (int) : Number of STAtt blocks in the encoder/decoder.
        K (int) : Number of attention heads.
        d (int) : Dimension of each attention head outputs.
        num_his (int): Number of history steps.
        bn_decay (float): Batch normalization momentum.
        steps_per_day (int): Number of steps in a day.
        use_bias (bool): Whether to use bias in Fully Connected layers.
        mask (bool): Whether to mask attention score in temporal attention.
    """

    def __init__(
        self,
        L: int,
        K: int,
        d: int,
        num_his: int,
        bn_decay: float,
        steps_per_day: int,
        use_bias: bool,
        mask: bool,
    ):
        super(GMAN, self).__init__()
        D = K * d
        self._num_his = num_his
        self._steps_per_day = steps_per_day
        self._st_embedding = SpatioTemporalEmbedding(
            D, bn_decay, steps_per_day, use_bias
        )
        self._st_att_block1 = nn.ModuleList(
            [SpatioTemporalAttention(K, d, bn_decay, mask) for _ in range(L)]
        )
        self._st_att_block2 = nn.ModuleList(
            [SpatioTemporalAttention(K, d, bn_decay, mask) for _ in range(L)]
        )
        self._transform_attention = TransformAttention(K, d, bn_decay)
        self._fully_connected_1 = FullyConnected(
            input_dims=[1, D],
            units=[D, D],
            activations=[F.relu, None],
            bn_decay=bn_decay,
        )
        self._fully_connected_2 = FullyConnected(
            input_dims=[D, D],
            units=[D, 1],
            activations=[F.relu, None],
            bn_decay=bn_decay,
        )

    def forward(
        self, X: torch.FloatTensor, SE: torch.FloatTensor, TE: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Making a forward pass of GMAN.

        Arg types:
            * **X** (PyTorch Float Tensor) - Input sequence, with shape (batch_size, num_hist, num of nodes).
            * **SE** (Pytorch Float Tensor) - Spatial embedding, with shape (numbed of nodes, K * d).
            * **TE** (Pytorch Float Tensor) - Temporal embedding, with shape (batch_size, num_his + num_pred, 2).

        Return types:
            * **X** (PyTorch Float Tensor) - Output sequence for prediction, with shape (batch_size, num_pred, num of nodes).
        """
        X = torch.unsqueeze(X, -1)
        X = self._fully_connected_1(X)
        STE = self._st_embedding(SE, TE, self._steps_per_day)
        STE_his = STE[:, : self._num_his]
        STE_pred = STE[:, self._num_his :]
        for net in self._st_att_block1:
            X = net(X, STE_his)
        X = self._transform_attention(X, STE_his, STE_pred)
        for net in self._st_att_block2:
            X = net(X, STE_pred)
        X = torch.squeeze(self._fully_connected_2(X), 3)
        return X