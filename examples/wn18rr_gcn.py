import torch
from torch_geometric.datasets import WordNet18RR
import model
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

writer = SummaryWriter()

dataset = WordNet18RR('.')
data=dataset[0]
NUM_NODE = 41105
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(9)
model = model.Net(num_node=NUM_NODE, num_node_ft=33, num_classes=dataset.num_classes())
optimizer = torch.optim.Adam(list(model.parameters()), lr=0.001)

reduced_edge_index = torch.tensor([[data.vertice_to_id[u] for u in data.edge_index[0,:].tolist()],
                                   [data.vertice_to_id[u] for u in data.edge_index[1,:].tolist()]], dtype=torch.long)
data.y = torch.tensor(data.y)

def train():
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    out = model(reduced_edge_index[:, data.train_mask])
    loss = F.nll_loss(out, data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    return total_loss

@torch.no_grad()
def test(mask):
    model.eval()
    out = model(reduced_edge_index[:, mask])
    loss = F.nll_loss(out, data.y[mask])
    return loss

for epoch in tqdm(range(1, 3000)):
    loss = train()
    writer.add_scalar("Loss/Train", loss, epoch)
    if epoch % 10 == 0:
        val_loss = test([_ or __ for _, __ in zip(data.val_mask, data.train_mask)])
        writer.add_scalar("Loss/Val", val_loss, epoch)
        model.train()

writer.close()
