# match same input of features
from AttFPfeaturing import datagenerator, getdataloader
#from Datas import dataloader
import pandas as pd
target_list = ['Result0']
import pickle

data = datagenerator(df, target_list)

df = pd.read_csv('esol.csv')

batch_size = 128

train_loader = getdataloader(data, batch_size, shuffle=True, drop_last=False)

for data in train_loader:
    print(data)
    print(data.x.shape)
    break

from AttFP import AttentiveFP
# generate the model architecture
model = AttentiveFP(40, 10, 200, R=2, T=2, dropout=0.2, debug=False)

# loop over data in a batch
import time
import torch

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_func = torch.nn.MSELoss()

# train the network
for t in range(200):
    start = time.time()
    epoch_loss = 0
    count_train = 0
    for data in train_loader:
        y = model(data)

        loss = loss_func(y.squeeze(),
                         data.y.squeeze())  # must be (1. nn output, 2. target)

        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        count_train += data.y.size(0)

        with torch.no_grad():
            epoch_loss += data.num_graphs * loss.item()

        stop = time.time()

        print('epoch', t + 1, ' - train loss:', epoch_loss / count_train,
              ' - runtime:', stop - start)
