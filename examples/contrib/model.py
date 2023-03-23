import torch
import torch.optim as optim
from torch_geometric.contrib.nn.conv.graphwavenet import GraphWaveNet
import util


class Model():

    def __init__(self, scaler, num_nodes, lrate, wdecay, device, adj_mx):
        self.gwnet = GraphWaveNet(num_nodes, 2, 1, 12)
        self.gwnet.to(device)
        self.optimizer = optim.Adam(self.gwnet.parameters(),
                                    lr=lrate,
                                    weight_decay=wdecay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5

        self.edge_index = [[], []]
        self.edge_weight = []

        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_mx.item((i, j)) != 0:
                    self.edge_index[0].append(i)
                    self.edge_index[1].append(j)
                    self.edge_weight.append(adj_mx.item((i, j)))

        self.edge_index = torch.tensor(self.edge_index)
        self.edge_weight = torch.tensor(self.edge_weight)

    def train(self, input, real_val):
        self.gwnet.train()
        self.optimizer.zero_grad()
        input = input.transpose(-3, -1)
        output = self.gwnet(input, self.edge_index, self.edge_weight)

        output = output.transpose(-3, -1)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)

        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.gwnet.parameters(), self.clip)
        self.optimizer.step()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mape, rmse

    def eval(self, input, real_val):
        self.gwnet.eval()
        input = input.transpose(-3, -1)
        output = self.gwnet(input, self.edge_index, self.edge_weight)
        output = output.transpose(-3, -1)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mape, rmse
