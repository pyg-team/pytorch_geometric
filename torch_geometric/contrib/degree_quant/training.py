# import time

# import numpy as np
# import torch
# import torch.nn.functional as F
# from sklearn.model_selection import StratifiedKFold
# from torch import tensor
# from torch.optim import Adam
# from tqdm import tqdm

# from torch_geometric.loader import DataLoader
# from torch_geometric.loader import DenseDataLoader as DenseLoader

# def cross_validation_with_val_set(dataset, model, folds, epochs, batch_size,
#                                   lr, lr_decay_factor, lr_decay_step_size,
#                                   weight_decay, use_tqdm=True, writer=None,
#                                   logger=None, save_path=None):

#     val_losses, accs, durations = [], [], []
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     for fold, (train_idx, test_idx,
#                val_idx) in enumerate(zip(*k_fold(dataset, folds))):

#         train_dataset = dataset[train_idx]
#         test_dataset = dataset[test_idx]
#         val_dataset = dataset[val_idx]

#         if "adj" in train_dataset[0]:
#             train_loader = DenseLoader(train_dataset, batch_size,
#             shuffle=True)
#             val_loader = DenseLoader(val_dataset, batch_size, shuffle=False)
#             test_loader = DenseLoader(test_dataset, batch_size,
#             shuffle=False)
#         else:
#             train_loader = DataLoader(train_dataset, batch_size,
#             shuffle=True)
#             val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
#             test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

#         model.to(device).reset_parameters()
#         optimizer = Adam(model.parameters(), lr=lr,
#         weight_decay=weight_decay)

#         if torch.cuda.is_available():
#             torch.cuda.synchronize()

#         t_start = time.perf_counter()

#         if use_tqdm:
#             t = tqdm(total=epochs, desc="Fold #" + str(fold), position=0,
#                      leave=True)

#         for epoch in range(1, epochs + 1):

#             train_loss = train(model, optimizer, train_loader)
#             val_loss = eval_loss(model, val_loader)
#             val_losses.append(val_loss)
#             accs.append(eval_acc(model, test_loader))

#             if epoch % lr_decay_step_size == 0:
#                 for param_group in optimizer.param_groups:
#                     param_group["lr"] = lr_decay_factor * param_group["lr"]

#             if use_tqdm:
#                 t.set_postfix({
#                     "Train_Loss": "{:05.3f}".format(train_loss),
#                     "Val_Loss": "{:05.3f}".format(val_loss),
#                     "Test Acc": "{:05.3f}".format(accs[-1])
#                 })
#                 t.update(1)

#         if torch.cuda.is_available():
#             torch.cuda.synchronize()
#         t_end = time.perf_counter()
#         durations.append(t_end - t_start)

#     loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)
#     loss, acc = loss.view(folds, epochs), acc.view(folds, epochs)
#     loss, argmin = loss.min(dim=1)
#     acc = acc[torch.arange(folds, dtype=torch.long), argmin]

#     loss_mean = loss.mean().item()
#     acc_mean = acc.mean().item()
#     acc_std = acc.std().item()
#     duration_mean = duration.mean().item()
#     print("Val Loss: {:.4f}, Test Accuracy: {:.3f} ± {:.3f}, Duration:
#           {:.3f}". format(loss_mean, acc_mean, acc_std, duration_mean))

#     full_dataloader = DataLoader(dataset, batch_size=batch_size,
#     shuffle=False)
#     # Compute the Inference on the Testing Part of the dataset
#     inference_timing(model, full_dataloader, batch_size)
#     # store the model
#     if save_path:
#         state_dict = model.state_dict()
#         torch.save(state_dict, save_path)

#     return loss_mean, acc_mean, acc_std

# # For Graph Dataset
# def k_fold(dataset, folds):
#     skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

#     test_indices, train_indices = [], []
#     for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
#         test_indices.append(torch.from_numpy(idx))

#     val_indices = [test_indices[i - 1] for i in range(folds)]

#     for i in range(folds):
#         train_mask = torch.ones(len(dataset), dtype=torch.bool)
#         train_mask[test_indices[i]] = 0
#         train_mask[val_indices[i]] = 0
#         train_indices.append(train_mask.nonzero().view(-1))

#     return train_indices, test_indices, val_indices

# def train(model, optimizer, loader):
#     model.train()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     total_loss = 0
#     for data in loader:
#         optimizer.zero_grad()
#         data = data.to(device)
#         out = model(data)
#         loss = F.nll_loss(out, data.y.view(-1))
#         loss.backward()
#         total_loss += loss.item() * num_graphs(data)
#         optimizer.step()
#     return total_loss / len(loader.dataset)

# def eval_acc(model, loader):
#     model.eval()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     correct = 0
#     for data in loader:
#         data = data.to(device)
#         with torch.no_grad():
#             pred = model(data).max(1)[1]
#         correct += pred.eq(data.y.view(-1)).sum().item()
#     return correct / len(loader.dataset)

# def eval_loss(model, loader):
#     model.eval()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     loss = 0
#     for data in loader:
#         data = data.to(device)

#         with torch.no_grad():
#             out = model(data)
#         loss += F.nll_loss(out, data.y.view(-1), reduction="sum").item()
#     return loss / len(loader.dataset)

# def inference_timing(model, loader, batch_size):
#     """
#     Computes the Inference Time on the dataset provided to the function

#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # batches = len(loader)
#     model.eval()
#     batch_time = []
#     print(f"Testing inference on {device}")
#     torch.cuda.synchronize(device)

#     for data in loader:

#         start = time.time()
#         model(data.to(device))

#         torch.cuda.synchronize(device)

#         end = time.time()
#         batch_time.append((end - start))

#     batch_time = np.array(batch_time) / batch_size
#     mean = np.mean(batch_time)
#     std = np.std(batch_time)
#     print(f'Inference time: {mean}±{std} seconds for batch size
#     {batch_size}')

# def num_graphs(data):
#     if data.batch is not None:
#         return data.num_graphs
#     else:
#         return data.x.size(0)
