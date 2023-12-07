from torch_geometric.data import Batch
from torch_geometric.datasets import TUDataset
import torch
import torch.optim as optim
import numpy as np
from tqdm import trange
import copy
from tqdm.auto import trange
import matplotlib.pyplot as plt

from xgnn_model import GCN_Graph

def create_single_batch(dataset):
    data_list = [data for data in dataset]
    batched_data = Batch.from_data_list(data_list)
    return batched_data

def test(test_dataset, model):
    model.eval()
    correct = 0
    total = test_dataset.num_graphs
    with torch.no_grad():
        pred = torch.round(model(test_dataset))
        pred = pred.squeeze()
        label = test_dataset.y.float()
        labels_prob = torch.stack([1 - label, label], dim=1)
        matches = (pred == labels_prob).float()

    return torch.mean(matches)

def train(dataset, args, train_indices, val_indices, test_indices):
    # Split dataset into training and testing (validation is not used here)
    train_dataset = create_single_batch([dataset[i] for i in train_indices]).to(device)
    test_dataset = create_single_batch([dataset[i] for i in test_indices]).to(device)

    # Model initialization
    model = GCN_Graph(args.input_dim, output_dim=2, dropout=args.dropout).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) # weight_decay=args.weight_decay

    # Training loop
    losses = []
    test_accs = []
    best_acc = 0
    best_model = None
    for epoch in trange(args.epochs, desc="Training", unit="Epoch"):
        model.train()
        opt.zero_grad()

        pred = model(train_dataset)
        label = train_dataset.y.float()
        labels_prob = torch.stack([1 - label, label], dim=1)
        loss = model.loss(pred, labels_prob)
        loss.backward()
        opt.step()
        total_loss = loss.item()
        losses.append(total_loss)

        # Test accuracy
        if epoch % 10 == 0:
            test_acc = test(test_dataset, model)
            test_accs.append(test_acc)
            if test_acc > best_acc:
                best_acc = test_acc
                best_model = copy.deepcopy(model)
        else:
            test_accs.append(test_accs[-1])

    return test_accs, losses, best_model, best_acc

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = {'device': device,
        'dropout': 0.1,
        'epochs': 1000,
        'input_dim' : 7,
        'opt': 'adam',
        'opt_scheduler': 'none',
        'opt_restart': 0,
        'weight_decay': 5e-5,
        'lr': 0.001}

args = objectview(args)

dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
num_graphs = len(dataset)

# Define split percentages
train_percentage = 0.7
val_percentage = 0.0

# Calculate split sizes
train_size = int(num_graphs * train_percentage)
val_size = int(num_graphs * val_percentage)
test_size = num_graphs - train_size - val_size

# Create shuffled indices
indices = np.random.permutation(num_graphs)
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

test_accs, losses, best_model, best_acc = train(dataset, args, train_indices, val_indices, test_indices)

try:
    torch.save(best_model.state_dict(), 'examples/explain/xgnn/mutag_model.pth')
    print("Model saved successfully.")
except Exception as e:
    print("Error saving model:", e)

# print("Maximum test set accuracy: {0}".format(max(test_accs)))
# print("Minimum loss: {0}".format(min(losses)))

# plt.title(dataset.name)
# plt.plot(losses, label="training loss")
# plt.plot(test_accs, label="test accuracy")
# plt.legend()
# plt.show()