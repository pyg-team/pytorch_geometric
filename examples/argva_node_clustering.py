import os.path as osp

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import GCNConv, ARGVA
import torch.nn.functional as F
import torch
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import v_measure_score, homogeneity_score, completeness_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


ENCODER_HIDDEN_CHANNEL = 32
ENCODER_OUTPUT_CHANNEL = 32
DISCRIMINATOR_HIDDEN_CHANNEL = 64

DISCRIMINATOR_TRAIN_COUNT = 5

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
cora = Planetoid(path, dataset, T.NormalizeFeatures())
data = cora.get(0)

data.train_mask = data.val_mask = data.test_mask = None
data = train_test_split_edges(data)


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden, cached=True)
        self.conv_mu = GCNConv(hidden, out_channels, cached=True)
        self.conv_logvar = GCNConv(hidden, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)


class Discriminator(torch.nn.Module):
    def __init__(self, in_channel, hidden, out_channel):
        super(Discriminator, self).__init__()
        self.dense1 = torch.nn.Linear(in_channel, hidden)
        self.dense2 = torch.nn.Linear(hidden, out_channel)
        self.dense3 = torch.nn.Linear(out_channel, out_channel)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x


encoder = Encoder(data.num_features, ENCODER_HIDDEN_CHANNEL,
                  ENCODER_OUTPUT_CHANNEL)
discriminator = Discriminator(
    ENCODER_OUTPUT_CHANNEL, DISCRIMINATOR_HIDDEN_CHANNEL, ENCODER_OUTPUT_CHANNEL)
model = ARGVA(encoder, discriminator)

discriminator_optimizer = torch.optim.Adam(
    model.discriminator.parameters(), lr=0.001)
model_optimizer = torch.optim.Adam(model.parameters(), lr=5*0.001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
discriminator.to(device)
model.to(device)
data = data.to(device)


def train():
    model.train()
    model_optimizer.zero_grad()
    z = model.encode(data.x, data.train_pos_edge_index)

    for i in range(DISCRIMINATOR_TRAIN_COUNT):
        discriminator.train()
        discriminator_optimizer.zero_grad()
        discriminator_loss = model.discriminator_loss(z)
        discriminator_loss.backward()
        discriminator_optimizer.step()

    loss = model.recon_loss(z, data.train_pos_edge_index)
    loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    model_optimizer.step()
    return loss


def test():
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.train_pos_edge_index)
        # Cluster embedded values using kmeans
        kmeans_input = z.cpu().data.numpy()
        kmeans = KMeans(n_clusters=7, random_state=0).fit(kmeans_input)
        pred = kmeans.predict(kmeans_input)

        labels = data.y.cpu().numpy()
        completeness = completeness_score(labels, pred)
        hm = homogeneity_score(labels, pred)
        nmi = v_measure_score(labels, pred)

    return model.test(z, data.test_pos_edge_index, data.test_neg_edge_index), completeness, hm, nmi


for epoch in range(150):
    loss = train()
    (auc, ap), completeness, hm, nmi = test()
    print('Epoch: {:05d}, '
          'Train Loss: {:.3f}, AUC: {:.3f}, AP: {:.3f}, Homogeneity: {:.3f}; Completeness: {:.3f}; NMI: {:.3f},  '.format(epoch, loss,
                                                                                                                          auc, ap, hm, completeness, nmi))


def plot(X, fig, col, size, true_labels):
    ax = fig.add_subplot(1, 1, 1)
    for i, point in enumerate(X):
        ax.scatter(point[0], point[1], s=size, c=col[true_labels[i]])


def plot_clusters(hidden_emb, true_labels):
    # Doing dimensionality reduction for plotting
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(hidden_emb)
    # Plot figure
    fig = plt.figure()
    plot(X_tsne, fig, ['red', 'green', 'blue', 'brown',
                       'purple', 'yellow', 'pink', 'orange'], 2, true_labels)
    plt.show()


with torch.no_grad():
    z = model.encode(data.x, data.train_pos_edge_index)
    plot_clusters(z.cpu().data, data.y.cpu().data)
