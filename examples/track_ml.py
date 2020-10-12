import numpy as np
import math
import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.data import Data
from trackml.dataset import load_dataset


def get_train_data(event):
    event_id, hits, cells, particles, truth = event

    hit_cells = cells.groupby(['hit_id']).value.count().values
    hit_value = cells.groupby(['hit_id']).value.sum().values
    features = np.hstack((hits[['x', 'y', 'z']]/1000,
                          hit_cells.reshape(len(hit_cells), 1)/10,
                          hit_value.reshape(len(hit_cells), 1)))
    particle_ids = truth.particle_id.unique()
    particle_ids = particle_ids[np.where(particle_ids != 0)[0]]

    pair = []
    for particle_id in particle_ids:
        hit_ids = truth[truth.particle_id == particle_id].hit_id.values-1
        for i in hit_ids:
            for j in hit_ids:
                if i != j:
                    pair.append([i, j])
    pair = np.array(pair)

    Train1 = np.hstack((features[pair[:, 0]],
                        features[pair[:, 1]],
                        np.ones((len(pair), 1))))

    n = len(hits)
    size = len(Train1)*3
    p_id = truth.particle_id.values
    i = np.random.randint(n, size=size)
    j = np.random.randint(n, size=size)
    pair = np.hstack((i.reshape(size, 1), j.reshape(size, 1)))
    pair = pair[((p_id[i] == 0) | (p_id[i] != p_id[j]))]

    Train0 = np.hstack((features[pair[:, 0]],
                        features[pair[:, 1]],
                        np.zeros((len(pair), 1))))

    Train = np.vstack((Train1, Train0))
    del Train0, Train1
    np.random.shuffle(Train)

    x = torch.from_numpy(Train[:, :-1]).float()
    y = torch.from_numpy(Train[:, -1]).long()

    return Data(x=x, y=y)


def train(epoch, dataset, hard_neg_mining=False, batch_size=2000):
    model.train()
    losses = []
    for event in dataset:
        data = get_train_data(event)
        num_batches = math.ceil(data.x.shape[0]/batch_size)
        for i in range(num_batches):
            x_batch = data.x[i*batch_size:(i+1)*batch_size].to(device)
            y_batch = data.y[i*batch_size:(i+1)*batch_size].to(device)

            optimizer.zero_grad()
            out = model(x_batch).squeeze()
            loss = F.binary_cross_entropy_with_logits(out,
                                                      y_batch.to(out.dtype))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return np.mean(losses)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.seq = Seq(
            Lin(10, 800), ReLU(), BN(800),
            Lin(800, 400), ReLU(), BN(400),
            Lin(400, 400), ReLU(), BN(400),
            Lin(400, 400), ReLU(), BN(400),
            Lin(400, 200), ReLU(), BN(200),
            Lin(200, 1)
        )

    def forward(self, x):
        return self.seq(x)


PATH_DATASET = '/home/matthias.barth/trackml_dataset/train_sample.zip'
dataset = load_dataset(PATH_DATASET)

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)


for epoch in range(1, 50):
    loss = train(epoch, dataset)
    print(loss)
    raise
