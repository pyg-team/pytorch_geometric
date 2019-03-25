import math

import numpy as np
import torch

from sklearn.metrics import roc_auc_score, average_precision_score


class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, z):
        adj = torch.matmul(z, z.t())
        return adj


class GAE(torch.nn.Module):
    def __init__(self, encoder):
        super(GAE, self).__init__()
        self.encoder = encoder
        self.decoder = Decoder()

    def forward(self, *args, **kwargs):
        z = self.encoder(*args, **kwargs)
        return self.decoder(z)

    def generate_edge_splits(self, data, val_ratio=0.05, test_ratio=0.1):
        assert 'batch' not in data  # No batch-mode.

        row, col = data.edge_index

        # Return upper triangular portion.
        mask = row < col
        row, col = row[mask], col[mask]

        n_v = math.floor(val_ratio * row.size(0))
        n_t = math.floor(test_ratio * row.size(0))

        # Positive edges.
        perm = torch.randperm(row.size(0))
        row, col = row[perm], col[perm]

        r, c = row[:n_v], col[:n_v]
        data.val_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
        data.test_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v + n_t:], col[n_v + n_t:]
        data.train_edge_index = torch.stack([r, c], dim=0)

        # Negative edges.
        num_nodes = data.num_nodes
        neg_adj_mask = torch.zeros(num_nodes, num_nodes, dtype=torch.uint8)
        neg_adj_mask[np.triu_indices(data.num_nodes, k=1)] = 1
        neg_adj_mask[row, col] = 0
        neg_row, neg_col = neg_adj_mask.nonzero().t()

        perm = torch.randperm(neg_row.size(0))
        perm = perm[:n_v + n_t]
        neg_row, neg_col = neg_row[perm], neg_col[perm]

        r, c = neg_row[:n_v], neg_col[:n_v]
        data.val_neg_edge_index = torch.stack([r, c], dim=0)
        neg_adj_mask[r, c] = 0
        r, c = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
        data.test_neg_edge_index = torch.stack([r, c], dim=0)
        neg_adj_mask[r, c] = 0
        data.train_neg_adj_mask = neg_adj_mask

        return data

    def reconstruction_loss(self, adj, edge_index, neg_adj_mask):
        row, col = edge_index
        loss = -torch.log(torch.sigmoid(adj[row, col])).mean()
        loss = loss - torch.log(1 - torch.sigmoid(adj[neg_adj_mask])).mean()
        return loss

    def evaluate(self, adj, edge_index, neg_edge_index):
        pos_y = adj.new_ones(edge_index.size(1))
        neg_y = adj.new_zeros(neg_edge_index.size(1))

        adj = torch.sigmoid(adj.detach())
        pos_pred = adj[edge_index[0], edge_index[1]]
        neg_pred = adj[neg_edge_index[0], neg_edge_index[1]]

        y = torch.cat([pos_y, neg_y], dim=0).cpu()
        pred = torch.cat([pos_pred, neg_pred], dim=0).cpu()

        roc_score = roc_auc_score(y, pred)
        ap_score = average_precision_score(y, pred)

        return roc_score, ap_score

class ARGA(GAE):
    def __init__(self, encoder, discriminator, n_latent):
        super(ARGA,self).__init__(encoder)
        self.discriminator = discriminator


    def reconstruction_loss(self, adj, edge_index, neg_adj_mask):
        print(adj)
        row, col = edge_index
        loss = -torch.log(torch.sigmoid(adj[row, col])).mean()
        print(loss)
        loss = loss - torch.log(1 - torch.sigmoid(adj[neg_adj_mask])).mean()
        print(loss)
        return loss

    def discriminate(self, z):
        z_real=torch.randn(z.size())
        d_real=self.discriminator(z_real)
        d_fake=self.discriminator(z)
        return d_real, d_fake

    def discriminator_loss(self, d_real, d_fake):
        dc_real_loss=nn.BCELoss(reduction='mean')(d_real,torch.ones(d_real.size()))
        dc_fake_loss=nn.BCELoss(reduction='mean')(d_fake,torch.zeros(d_fake.size()))
        dc_gen_loss=nn.BCELoss(reduction='mean')(d_fake,torch.ones(d_fake.size()))
        return dc_real_loss + dc_fake_loss + dc_gen_loss

    def loss(self, d_real, d_fake, *args):
        args=list(args)
        args[0] = self.decoder(args[0])
        recon_loss = self.reconstruction_loss(*args)
        d_loss = self.discriminator_loss(d_real, d_fake)
        total_loss = recon_loss+d_loss
        return total_loss

class VGAE(GAE):
    def __init__(self, encoder, out_channels, n_latent):
        super(VGAE,self).__init__(encoder)
        self.z_mean=nn.Linear(out_channels,n_latent)
        self.z_var=nn.Linear(out_channels,n_latent)
        torch.nn.init.xavier_uniform(self.z_mean.weight)
        torch.nn.init.xavier_uniform(self.z_var.weight)

    def kl_loss(self, mean, logvar):
        loss=torch.mean(0.5 * torch.sum(torch.exp(logvar) + mean**2 - 1. - logvar, 1))
        print(loss)
        return loss

    def reconstruction_loss(self, adj, edge_index, neg_adj_mask):
        row, col = edge_index
        #print(adj[row, col])
        loss = -torch.log(torch.sigmoid(adj[row, col])).mean()
        print(loss)
        loss = loss - torch.log(1 - torch.sigmoid(adj[neg_adj_mask])).mean()
        return loss

    def sample_z(self, mean, logvar):
        stddev = torch.exp(0.5 * logvar)
        noise = Variable(torch.randn(stddev.size()))
        if torch.cuda.is_available():
            noise=noise.cuda()
        return (noise * stddev) + mean

    def encode(self, x, edge_index):
        z=F.relu(self.encoder(x, edge_index))
        mean,logvar = self.z_mean(z), self.z_var(z)
        z=self.sample_z(mean,logvar)
        return z, mean, logvar

    def loss(self, z, mean, logvar, *args):
        args=list(args)
        args[0] = self.decoder(args[0])
        recon_loss = self.reconstruction_loss(*args)
        kl_loss = self.kl_loss(mean, logvar)
        total_loss = recon_loss+kl_loss
        return total_loss

class ARGVA(ARGA):
    def __init__(self, encoder, discriminator, out_channels):
        n_latent = out_channels
        super(ARGVA,self).__init__(encoder,discriminator,n_latent)
        self.discriminator = discriminator
        self.z_mean=nn.Linear(out_channels,n_latent)
        self.z_var=nn.Linear(out_channels,n_latent)
        torch.nn.init.xavier_uniform(self.z_mean.weight)
        torch.nn.init.xavier_uniform(self.z_var.weight)

    def kl_loss(self, mean, logvar):
        loss=torch.mean(0.5 * torch.sum(torch.exp(logvar) + mean**2 - 1. - logvar, 1))
        print(loss)
        return loss

    def sample_z(self, mean, logvar):
        stddev = torch.exp(0.5 * logvar)
        noise = Variable(torch.randn(stddev.size()))
        if torch.cuda.is_available():
            noise=noise.cuda()
        return (noise * stddev) + mean

    def encode(self, x, edge_index):
        z=F.relu(self.encoder(x, edge_index))
        mean,logvar = self.z_mean(z), self.z_var(z)
        z=self.sample_z(mean,logvar)
        return z, mean, logvar

####### tests

class Discriminator(nn.Module):
    def __init__(self, n_input, hidden_layers=[30,20]):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(n_input,hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0],hidden_layers[1])
        self.fc3 = nn.Linear(hidden_layers[1],1)
        torch.nn.init.xavier_uniform(self.fc1.weight)
        torch.nn.init.xavier_uniform(self.fc2.weight)
        torch.nn.init.xavier_uniform(self.fc3.weight)
        self.fc1 = nn.Sequential(self.fc1, nn.ReLU())
        self.fc2 = nn.Sequential(self.fc2, nn.ReLU())
        self.fc3 = nn.Sequential(self.fc3,nn.Sigmoid())

    def forward(self, z):
        z=self.fc1(z)
        z=self.fc2(z)
        out=self.fc3(z)
        return out

model=gae=GAE(Encoder(17,30))

data.train_mask = data.val_mask = data.test_mask = data.y = None
data = model.generate_edge_splits(data)
x, edge_index = data.x, data.edge_index
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


model=VGAE(Encoder(data.num_features,45),45,20)
optimizer=torch.optim.Adam(params=model.parameters(),lr=1e-3,weight_decay=1e-1)
model=ARGA(Encoder(data.num_features,45),Discriminator(45,[100,50]),45)
optimizer=torch.optim.Adam(params=model.parameters(),lr=1e-4,weight_decay=1e-2)
model=ARGVA(Encoder(data.num_features,10),Discriminator(10,[500,20]),10)
optimizer=torch.optim.Adam(params=model.parameters(),lr=1e-5,weight_decay=5e-1)

beta=1.
def train_vgae():
    model.train()
    optimizer.zero_grad()
    z, mean, logvar = model.encode(x, edge_index)
    loss = beta*model.kl_loss(mean, logvar)+model.reconstruction_loss(model.decoder(z), data.train_edge_index, data.train_neg_adj_mask) # model.
    print(loss.item())
    loss.backward()
    optimizer.step()

def train_arga():
    model.train()
    optimizer.zero_grad()
    z = model.encoder(x, edge_index)
    d_real, d_fake = model.discriminate(z)
    loss = model.discriminator_loss(d_real, d_fake)+model.reconstruction_loss(model.decoder(z), data.train_edge_index, data.train_neg_adj_mask) # model.
    print(loss.item())
    loss.backward()
    optimizer.step()

def train_argva():
    model.train()
    optimizer.zero_grad()
    z, mean, logvar = model.encode(x, edge_index)
    d_real, d_fake = model.discriminate(z)
    loss = beta*model.kl_loss(mean, logvar)+model.discriminator_loss(d_real, d_fake)+model.reconstruction_loss(model.decoder(z), data.train_edge_index, data.train_neg_adj_mask) # model.
    print(loss.item())
    loss.backward()
    optimizer.step()

def simulate():
    model.train(False)
    with torch.no_grad():
        return model.decoder(model.encode(x, edge_index)[0]).detach().numpy() # model.

def test(pos_edge_index, neg_edge_index):
    model.train(False)
    with torch.no_grad():
        z = model.encoder(x, edge_index)
        print(z.size())
    return model.eval(model.decoder(z), pos_edge_index, neg_edge_index) # model.

train = train_argva

from copy import deepcopy
top_model=None
top_score=0.
for i in range(1000): # 000
    train() # don't train for too long, else test performance will reduce due to overfitting, maybe need val set of edges, for now it is okay for demo purposes. That's why some are wrong no matter what
    val_score=np.mean(test(data.val_edge_index, data.val_neg_edge_index))
    print(val_score)
    print(np.mean(test(data.test_edge_index, data.test_neg_edge_index)))
    if val_score >= top_score:
        top_model = deepcopy(model)
        top_score = val_score

model = top_model
