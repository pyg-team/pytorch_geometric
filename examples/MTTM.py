# The MTTM method from the "A Multi-Type Transferable Method for Missing Link Prediction in Heterogeneous Social Networks" paper.
# IEEE: https://ieeexplore.ieee.org/abstract/document/10004751

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics
import re
import networkx as nx
import pandas as pd
import time, os
import numpy as np

num_epochs = 200
batch_size = 32
learning_rate = 0.001
hidden_dim = 16

class edgeFeatures(object):
    def __init__(self, label=None, type = None, embeddings = None):
        self.label = label
        self.type = type
        self.embeddings = embeddings
        return

def structuralGraph(realFileName, fakeFileName, dataset):
    dataReal = pd.read_csv(realFileName, sep=' ', skiprows=0)
    dataFake = pd.read_csv(fakeFileName, sep=' ', skiprows=0)

    train_Real_Graph = nx.Graph()
    train_Fake_Graph = nx.Graph()
    test_Real_Graph = nx.Graph()
    test_Fake_Graph = nx.Graph()

    real_edge_Attritube = np.array(dataReal.iloc[:, 0:3])
    fake_edge_Attritube = np.array(dataFake.iloc[:, 0:3])

    lenReal = len(real_edge_Attritube)
    lenFake = len(fake_edge_Attritube)

    # print(real_edge_Attritube)
    #new type id according to dataset
    if dataset.lower() == 'facebook':
        dataNewType = [9, 8, 7, 6, 5, 4]
    else:
        dataNewType = [2]

    for i in range(lenReal):
        relation = real_edge_Attritube[i][2]
        if relation in dataNewType:
            test_Real_Graph.add_edge(real_edge_Attritube[i][0], real_edge_Attritube[i][1], relationship=relation)
        else:
            train_Real_Graph.add_edge(real_edge_Attritube[i][0], real_edge_Attritube[i][1], relationship=relation)


    for i in range(lenFake):
        relation = fake_edge_Attritube[i][2]
        if relation in dataNewType:
            test_Fake_Graph.add_edge(fake_edge_Attritube[i][0], fake_edge_Attritube[i][1], relationship=relation)
        else:
            train_Fake_Graph.add_edge(fake_edge_Attritube[i][0], fake_edge_Attritube[i][1], relationship=relation)

    return train_Real_Graph, train_Fake_Graph, test_Real_Graph, test_Fake_Graph

def get_train_validate_test(dataset):
    realFileName = 'Datasets/' + dataset + '/realData.csv'
    fakeFileName = 'Datasets/' + dataset + '/fakeData.csv'
    train_Real_Graph, train_Fake_Graph, test_Real_Graph, test_Fake_Graph = structuralGraph(realFileName, fakeFileName, dataset)
    node2vecReFile = "Datasets/node2vecFeature/" + dataset + "Feature.txt"
    data = pd.read_csv(node2vecReFile, sep=' ', skiprows=1, header=None)
    edges = np.array(data.iloc[:, 0:1]) + np.array(data.iloc[:, 1:2])
    embeddings = np.array(data.iloc[:, 2:66])
    nodeL = np.array(data.iloc[:, 0:1])
    nodeR = np.array(data.iloc[:, 1:2])
    train_data = []
    test = []
    for i in range(len(edges)):
        edgeFeature = edgeFeatures(" ")
        nodel = int(re.sub("\D", "", nodeL[i][0]))
        noder = int(re.sub("\D", "", nodeR[i][0]))
        if train_Real_Graph.has_edge(nodel, noder) or train_Fake_Graph.has_edge(nodel, noder): # train set
            if train_Real_Graph.has_edge(nodel, noder):
                label = 1
                type = train_Real_Graph.get_edge_data(nodel, noder)['relationship']
            else:
                label = 0
                type = train_Fake_Graph.get_edge_data(nodel, noder)['relationship']
            edgeFeature.embeddings = embeddings[i]
            edgeFeature.label = label
            edgeFeature.type = type
            train_data.append(edgeFeature)
        elif test_Real_Graph.has_edge(nodel, noder) or test_Fake_Graph.has_edge(nodel, noder):  # test set
            if test_Real_Graph.has_edge(nodel, noder):
                label = 1
                type = test_Real_Graph.get_edge_data(nodel, noder)['relationship']
            else:
                label = 0
                type = test_Fake_Graph.get_edge_data(nodel, noder)['relationship']
            edgeFeature.embeddings = embeddings[i]
            edgeFeature.label = label
            edgeFeature.type = type
            test.append(edgeFeature)
        else:
            continue

    train, validate = train_test_split(train_data, test_size=0.2)  # train_test_split返回切分的数据集train/validate
    train_dataset = []
    validate_dataset = []
    test_dataset = []
    for index, element in enumerate(train):
        vectors = torch.tensor(element.embeddings, dtype=torch.float32)
        label = torch.tensor(element.label, dtype=torch.float32)
        type = torch.tensor(element.type, dtype=torch.float32)
        m = [vectors, label, type]
        train_dataset.append(m)
    for index, element in enumerate(validate):
        vectors = torch.tensor(element.embeddings, dtype=torch.float32)
        label = torch.tensor(element.label, dtype=torch.float32)
        type = torch.tensor(element.type, dtype=torch.float32)
        m = [vectors, label, type]
        validate_dataset.append(m)
    for index, element in enumerate(test):
        vectors = torch.tensor(element.embeddings, dtype=torch.float32)
        label = torch.tensor(element.label, dtype=torch.float32)
        type = torch.tensor(element.type, dtype=torch.float32)
        m = [vectors, label, type]
        test_dataset.append(m)
    print('train length', len(train_dataset))
    print('validate length', len(validate_dataset))
    print('test length', len(test_dataset))
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    validate_loader = DataLoader(dataset=validate_dataset, batch_size=batch_size,  shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, validate_loader, test_loader

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

class re_shape(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x.reshape(len(x),len(x[0][0])))

    @staticmethod
    def backward(ctx, grad_output):
        output =  grad_output.reshape(len(grad_output),1,len(grad_output[0]))
        return output,None

class GradReverse(Function):
    @ staticmethod
    def forward(ctx, x, lambd, **kwargs: None):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, *grad_output):
        return grad_output[0] * -ctx.lambd, None

    def backward(ctx, grad_output):
        return grad_output * -ctx.lambd, None

class adversarial_neural_networks(nn.Module):
    def __init__(self, predicted_Type):
        super(adversarial_neural_networks, self).__init__()
        self.predicted_Type = predicted_Type

        ##The generative predictor
        self.predictor = nn.Sequential()
        self.predictor.add_module('exta_Conv1',nn.Conv1d(in_channels=1, out_channels=1, kernel_size=10, stride=1, padding=0))
        self.predictor.add_module('fully_connected_layer1', nn.Linear(55, 32))

        self.predictor_classifier = nn.Sequential()
        self.predictor_classifier.add_module('c_fc1', nn.Linear(32,24))
        self.predictor_classifier.add_module('c_fc1_relu', nn.ReLU())
        self.predictor_classifier.add_module('c_fc2', nn.Linear(24, 16))
        self.predictor_classifier.add_module('c_fc2_relu', nn.ReLU())
        self.predictor_classifier.add_module('c_fc3', nn.Linear(16, 2))
        self.predictor_classifier.add_module('c_softmax', nn.Softmax(dim=1))  # 对每一行进行softmax

        #discriminative classifier learn shared feature
        self.discriminative_classifier = nn.Sequential()
        self.discriminative_classifier.add_module('d_fc1', nn.Linear(32, 16))
        self.discriminative_classifier.add_module('relu_f1', nn.ReLU())
        self.discriminative_classifier.add_module('d_fc2', nn.Linear(16, self.predicted_Type))
        self.discriminative_classifier.add_module('d_softmax',nn.Softmax(dim=1))

    def forward(self, embeddings):
        embeddings = self.predictor(embeddings)
        shared_embeddings = re_shape.apply(embeddings)
        link_output = self.predictor_classifier(shared_embeddings)
        reverse_embeddings = GradReverse.apply(shared_embeddings, 1.0)
        type_output = self.discriminative_classifier(reverse_embeddings)
        return link_output, type_output

def to_np(x):
    return x.data.cpu().numpy()

def train_adversarial_neural_networks(train_loader, validate_loader, model, output_file):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate)
    best_validate_acc = 0.000
    best_validate_dir = ''

    print('training model')
    # Train of Model
    for epoch in range(num_epochs):  # num_epochs is 50
        p = float(epoch) / 100
        lr = learning_rate / (1. + 10 * p) ** 0.75
        optimizer.lr = lr
        cost_vector = []
        prediction_cost_vector = []
        classification_cost_vector = []
        acc_vector = []
        valid_acc_vector = []
        vali_cost_vector = []
        train_score = []
        train_label = []
        for i, (train_data, train_labels, type_labels) in enumerate(train_loader):
            optimizer.zero_grad()
            train_data = to_var(train_data)
            train_labels = to_var(train_labels),
            type_labels = to_var(type_labels)
            link_outputs, type_outputs = model(train_data.unsqueeze(1))
            train_score += list(link_outputs[:, 1].cpu().detach().numpy())
            train_label += list(train_labels[0].numpy())
            train_labels = train_labels[0]
            train_labels = train_labels.long()
            type_labels = type_labels.long()
            prediction_loss = criterion(link_outputs, train_labels)
            classification_loss = criterion(type_outputs, type_labels)
            loss = prediction_loss + classification_loss
            loss.backward()
            optimizer.step()
            _, argmax = torch.max(link_outputs, 1)
            accuracy = (train_labels == argmax.squeeze()).float().mean()
            prediction_cost_vector.append(prediction_loss.item())
            classification_cost_vector.append(classification_loss.item())
            cost_vector.append(loss.item())
            acc_vector.append(accuracy.item())

        # validate process
        model.eval()
        validate_acc_vector_temp = []
        for i, (validate_data, validate_labels, type_labels) in enumerate(validate_loader):
            validate_data = to_var(validate_data)
            validate_labels = to_var(validate_labels)
            type_labels = to_var(type_labels)
            validate_outputs, type_outputs = model(validate_data.unsqueeze(1))
            _, validate_argmax = torch.max(validate_outputs, 1)
            validate_labels = validate_labels.long()
            vali_loss = criterion(validate_outputs, validate_labels)
            validate_accuracy = (validate_labels == validate_argmax.squeeze()).float().mean()
            vali_cost_vector.append(vali_loss.item())
            validate_acc_vector_temp.append(validate_accuracy.item())
        validate_acc = np.mean(validate_acc_vector_temp)
        valid_acc_vector.append(validate_acc)
        model.train()
        print('Epoch [%d/%d],  Loss: %.4f, Link Prediction Loss: %.4f, Type Classification Loss: %.4f, Train_Acc: %.4f,  Validate_Acc: %.4f.'
              % (epoch + 1, num_epochs, np.mean(cost_vector), np.mean(prediction_cost_vector), np.mean(classification_cost_vector),
                 np.mean(acc_vector), validate_acc))

        if validate_acc > best_validate_acc:
            best_validate_acc = validate_acc
            if not os.path.exists(output_file):
                os.mkdir(output_file)
            best_validate_dir = output_file + str(epoch + 1) + '.pkl'
            torch.save(model.state_dict(), best_validate_dir)
    return best_validate_dir

def test_adversarial_neural_networks(best_validate_dir, test_loader, model):
    # Test the Model
    print('testing model')
    model.load_state_dict(torch.load(best_validate_dir))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    test_score = []
    test_pred = []
    test_true = []
    tes_score = []
    tes_label = []

    for i, (test_data, test_labels, type_labels) in enumerate(test_loader):
        test_data = to_var(test_data)
        test_labels = to_var(test_labels)
        # type_labels = to_var(type_labels)
        test_outputs, type_outputs = model(test_data.unsqueeze(1))
        tes_score += list(test_outputs[:, 1].cpu().detach().numpy())
        tes_label += list(test_labels.numpy())
        _, test_argmax = torch.max(test_outputs, 1)
        if i == 0:
            test_score = to_np(test_outputs)
            test_pred = to_np(test_argmax)
            test_true = to_np(test_labels)
        else:
            test_score = np.concatenate((test_score, to_np(test_outputs)), axis=0)
            test_pred = np.concatenate((test_pred, to_np(test_argmax)), axis=0)
            test_true = np.concatenate((test_true, to_np(test_labels)), axis=0)

    test_accuracy = metrics.accuracy_score(test_true, test_pred)
    test_precision = metrics.precision_score(test_true, test_pred, average='macro')
    test_aucroc = metrics.roc_auc_score(tes_label, tes_score, average='macro')

    return test_aucroc, test_precision, test_accuracy

def main(predicted_Type, dataset, output_file):
    train_loader, validate_loader, test_loader = get_train_validate_test(dataset)
    model = adversarial_neural_networks(predicted_Type)
    best_validate_dir = train_adversarial_neural_networks(train_loader, validate_loader, model, output_file)
    auc, precision, accuracy = test_adversarial_neural_networks(best_validate_dir, test_loader, model)
    print("Final reault: AUC -- %.4f  " % (auc), "Precision -- %.4f  " % (precision), 'Accuracy -- %.4f  ' % (accuracy))

if __name__ == '__main__':
    datasets = ['Facebook', 'IMDB', 'YELP', 'DBLP']
    dataset = datasets[0]
    print('Input dataset is:', dataset)
    predicted_Type_datasets = {'Facebook': 4, 'IMDB': 2, 'YELP': 2, 'DBLP': 2}
    predicted_Type = predicted_Type_datasets[dataset]
    output_file = 'trainOutput/' + dataset.lower() + '/output.txt'
    main(predicted_Type, dataset, output_file)