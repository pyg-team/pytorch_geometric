import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange
from tqdm.auto import trange
from xgnn_model import GCN_Graph

from torch_geometric.data import Batch
from torch_geometric.datasets import TUDataset

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


def create_single_batch(dataset):
    data_list = [data for data in dataset]
    batched_data = Batch.from_data_list(data_list)
    return batched_data


def test(test_dataset, model):
    model.eval()
    with torch.no_grad():
        logits = model(test_dataset).squeeze()  # Logits for each graph
        probabilities = torch.sigmoid(
            logits)  # Convert logits to probabilities
        predictions = probabilities > 0.5  # Convert probabilities to binary predictions
        correct = (
            predictions == test_dataset.y).float()  # Assumes labels are 0 or 1
        accuracy = correct.mean()

    return accuracy


def train(dataset, args, train_indices, val_indices, test_indices):
    # Split dataset into training and testing (validation is not used here)
    train_dataset = create_single_batch([dataset[i]
                                         for i in train_indices]).to(device)
    test_dataset = create_single_batch([dataset[i]
                                        for i in test_indices]).to(device)

    # Model initialization
    model = GCN_Graph(args.input_dim, output_dim=1,
                      dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)  #

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
        loss = model.loss(pred.squeeze(), label)
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

args = {
    'device': device,
    'dropout': 0.1,
    'epochs': 5000,
    'input_dim': 7,
    'opt': 'adam',
    'opt_restart': 0,
    'weight_decay': 1e-4,
    'lr': 0.007
}

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

test_accs, losses, best_model, best_acc = train(dataset, args, train_indices,
                                                val_indices, test_indices)

try:
    torch.save(best_model.state_dict(),
               'examples/explain/xgnn/mutag_model.pth')
    print("Model saved successfully.")
except Exception as e:
    print("Error saving model:", e)

print("Maximum test set accuracy: {0}".format(max(test_accs)))
print("Minimum loss: {0}".format(min(losses)))

plt.title(dataset.name)
plt.plot(losses, label="training loss")
plt.plot(test_accs, label="test accuracy")
plt.legend()
plt.show()
