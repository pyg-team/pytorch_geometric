import torch
from dataset import PemsBayDatasetLoader
from dcrnn import DCRNN


feature_dim = int(len(dataset.features[0][0]) * len(dataset.features[0][0][0]))
node_num = int(len(dataset.features[0]))

split_train_test_ratio = 0.7

#Load Dataset
loader = PemsBayDatasetLoader()
dataset = loader.get_dataset()

#Split Train and Test
train_dataset, test_dataset = dataset.features[:int(len(dataset.features) * split_train_test_ratio)] , dataset.features[int(len(dataset.features) * split_train_test_ratio):]

#Pre Processing node features : Change 2D node features to 1D in every time steps
Dynamic_node_Features_Train = torch.FloatTensor(train_dataset).view(-1,node_num,feature_dim)
Dynamic_node_Features_Test = torch.FloatTensor(test_dataset).view(-1,node_num,feature_dim)

#Graph Structure
Static_edge_index = torch.LongTensor(dataset.edge_index)
Static_edge_weight = torch.FloatTensor(dataset.edge_weight)

#Defining model
model = DCRNN(feature_dim,feature_dim,5)
predict_model = torch.nn.Linear(feature_dim*2,feature_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
  model.train()
  for epoch in range(100):
    cost = 0
    for time in range(Dynamic_node_Features_Train.shape[0]-1):
        y_hat_encoded = model(Dynamic_node_Features_Train[time], Static_edge_index, Static_edge_weight)
        y_hat = predict_model(y_hat_encoded)
        loss = torch.mean((y_hat-Dynamic_node_Features_Train[time+1])**2)
        cost = cost + loss
        if time > 100 and time % 100 == 0 :
          print(f'Mse Loss in Epoch :{epoch} and time : {time} : {loss}')
          cost = cost / 100
          cost.backward()
          optimizer.step()
          optimizer.zero_grad()
          cost = 0




@torch.no_grad()
def test():
  model.eval()  
  cost = 0
  for time in range(Dynamic_node_Features_Test.shape[0]-1):
    y_hat_encoded = model(Dynamic_node_Features_Test[time], Static_edge_index, Static_edge_weight)
    y_hat = predict_model(y_hat_encoded)
    loss = torch.mean((y_hat-Dynamic_node_Features_Test[time+1])**2)
    cost += loss
    print(f'Mse Loss in time : {time} : {loss}')


train()