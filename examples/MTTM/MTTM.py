# The MTTM method from the "A Multi-Type Transferable Method for Missing Link Prediction in Heterogeneous Social Networks" paper.
# IEEE: https://ieeexplore.ieee.org/abstract/document/10004751

import os
import re
import time

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch.autograd import Function, Variable
from torch.utils.data import DataLoader, Dataset

num_epochs = 200
batch_size = 32
learning_rate = 0.001
hidden_dim = 16


class edgeFeatures(object):
    def __init__(self, label=None, type=None, embeddings=None):
        self.label = label
        self.type = type
        self.embeddings = embeddings
        return


def structuralGraph(realFileName, fakeFileName, dataset):
    dataReal = pd.read_csv(realFileName, sep=' ', skiprows=0)
    dataFake = pd.read_csv(fakeFileName, sep=' ', skiprows=0)

    #初始化真实网路、虚假网络，测试集的真实网络和虚假网络
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
    # 选择Facebook中的4种作为预测边即类型为0，1，2，3的边作为测试集，类型为4，5，6，7，8，9的作为训练集
    if dataset.lower() == 'facebook':
        dataNewType = [9, 8, 7, 6, 5, 4]
    else:
        dataNewType = [2]

    #将数据<左节点，右节点，边类型>添加到初始化的训练集的真实网路、测试集的真实网络
    for i in range(lenReal):
        relation = real_edge_Attritube[i][2]
        if relation in dataNewType:
            test_Real_Graph.add_edge(real_edge_Attritube[i][0],
                                     real_edge_Attritube[i][1],
                                     relationship=relation)
        else:
            train_Real_Graph.add_edge(real_edge_Attritube[i][0],
                                      real_edge_Attritube[i][1],
                                      relationship=relation)

    #将数据<左节点，右节点，边类型>添加到训练集的虚假网络、测试集的虚假网络
    for i in range(lenFake):
        relation = fake_edge_Attritube[i][2]
        if relation in dataNewType:
            test_Fake_Graph.add_edge(fake_edge_Attritube[i][0],
                                     fake_edge_Attritube[i][1],
                                     relationship=relation)
        else:
            train_Fake_Graph.add_edge(fake_edge_Attritube[i][0],
                                      fake_edge_Attritube[i][1],
                                      relationship=relation)

    return train_Real_Graph, train_Fake_Graph, test_Real_Graph, test_Fake_Graph


def get_train_validate_test(dataset):
    realFileName = 'Datasets/' + dataset + '/realData.csv'
    fakeFileName = 'Datasets/' + dataset + '/fakeData.csv'

    #将数据集根据类型划分成训练集的真实网路、虚假网络，测试集的真实网络和虚假网络，以便于后续给网络的特征加上标签
    train_Real_Graph, train_Fake_Graph, test_Real_Graph, test_Fake_Graph = structuralGraph(
        realFileName, fakeFileName, dataset)

    node2vecReFile = "Datasets/node2vecFeature/" + dataset + "Feature.txt"
    data = pd.read_csv(node2vecReFile, sep=' ', skiprows=1, header=None)
    edges = np.array(data.iloc[:, 0:1]) + np.array(data.iloc[:, 1:2])
    embeddings = np.array(data.iloc[:, 2:66])
    nodeL = np.array(data.iloc[:, 0:1])
    nodeR = np.array(data.iloc[:, 1:2])
    train_data = []
    test = []

    #将训练集和测试集的数据准备好，每一条数据形式<label, type, embeddings>,也就是class： edgeFeatures
    #label：0或则1
    #type：训练集中是4，5，6，7，8，9测试集中是3，2，1，0
    #embeddings：64维的向量
    for i in range(len(edges)):
        edgeFeature = edgeFeatures(" ")
        nodel = int(re.sub("\D", "", nodeL[i][0]))
        noder = int(re.sub("\D", "", nodeR[i][0]))
        if train_Real_Graph.has_edge(nodel,
                                     noder) or train_Fake_Graph.has_edge(
                                         nodel, noder):  # train set
            if train_Real_Graph.has_edge(nodel, noder):
                label = 1
                type = train_Real_Graph.get_edge_data(nodel,
                                                      noder)['relationship']
            else:
                label = 0
                type = train_Fake_Graph.get_edge_data(nodel,
                                                      noder)['relationship']
            edgeFeature.embeddings = embeddings[i]
            edgeFeature.label = label
            edgeFeature.type = type
            train_data.append(edgeFeature)
        elif test_Real_Graph.has_edge(nodel,
                                      noder) or test_Fake_Graph.has_edge(
                                          nodel, noder):  # test set
            if test_Real_Graph.has_edge(nodel, noder):
                label = 1
                type = test_Real_Graph.get_edge_data(nodel,
                                                     noder)['relationship']
            else:
                label = 0
                type = test_Fake_Graph.get_edge_data(nodel,
                                                     noder)['relationship']
            edgeFeature.embeddings = embeddings[i]
            edgeFeature.label = label
            edgeFeature.type = type
            test.append(edgeFeature)
        else:
            continue

    #从训练集划分出一部分验证集
    train, validate = train_test_split(
        train_data, test_size=0.2)  # train_test_split返回切分的数据集train/validate
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
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              shuffle=True)
    validate_loader = DataLoader(dataset=validate_dataset,
                                 batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                             shuffle=False)

    #返回训练集、验证集以及测试集
    return train_loader, validate_loader, test_loader


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()

    return Variable(x)


class re_shape(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x.reshape(len(x), len(x[0][0])))

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.reshape(len(grad_output), 1, len(grad_output[0]))
        return output, None


class GradReverse(Function):
    @staticmethod
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
        #cnn+fc初步优化特征
        self.predictor = nn.Sequential()
        self.predictor.add_module(
            'exta_Conv1',
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=10, stride=1,
                      padding=0))
        self.predictor.add_module('fully_connected_layer1', nn.Linear(55, 32))

        # 三层fc进行链路预测的存在与不存在的二分类
        self.predictor_classifier = nn.Sequential()
        self.predictor_classifier.add_module('c_fc1', nn.Linear(32, 24))
        self.predictor_classifier.add_module('c_fc1_relu', nn.ReLU())
        self.predictor_classifier.add_module('c_fc2', nn.Linear(24, 16))
        self.predictor_classifier.add_module('c_fc2_relu', nn.ReLU())
        self.predictor_classifier.add_module('c_fc3', nn.Linear(16, 2))
        self.predictor_classifier.add_module(
            'c_softmax', nn.Softmax(dim=1))  # 对每一行进行softmax

        #discriminative classifier learn shared feature
        # 二层fc进行链路预测的类型分类
        self.discriminative_classifier = nn.Sequential()
        self.discriminative_classifier.add_module('d_fc1', nn.Linear(32, 16))
        self.discriminative_classifier.add_module('relu_f1', nn.ReLU())
        self.discriminative_classifier.add_module(
            'd_fc2', nn.Linear(16, self.predicted_Type))
        self.discriminative_classifier.add_module('d_softmax',
                                                  nn.Softmax(dim=1))

    def forward(self, embeddings):
        # print('embeddings:', embeddings)
        embeddings = self.predictor(embeddings)
        shared_embeddings = re_shape.apply(embeddings)
        link_output = self.predictor_classifier(shared_embeddings)

        # GradReverse->GRL梯度反转层
        reverse_embeddings = GradReverse.apply(shared_embeddings, 1.0)
        type_output = self.discriminative_classifier(reverse_embeddings)

        #输入链路预测的二分类结果和类型的多分类结果
        return link_output, type_output


def to_np(x):
    return x.data.cpu().numpy()


def train_adversarial_neural_networks(train_loader, validate_loader, model,
                                      output_file):

    if torch.cuda.is_available():
        model.cuda()
    #初始化损失函数：交叉熵函数
    criterion = nn.CrossEntropyLoss()
    #优化器
    optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate)
    #初始化精度
    best_validate_acc = 0.000
    best_validate_dir = ''

    print('training model')
    # Train of Model
    for epoch in range(num_epochs):  # num_epochs is 50
        p = float(epoch) / 100
        lr = learning_rate / (1. + 10 * p)**0.75
        optimizer.lr = lr
        cost_vector = []
        prediction_cost_vector = []
        classification_cost_vector = []
        acc_vector = []
        valid_acc_vector = []
        vali_cost_vector = []
        train_score = []
        train_label = []
        for i, (train_data, train_labels,
                type_labels) in enumerate(train_loader):
            optimizer.zero_grad()
            train_data = to_var(train_data)
            train_labels = to_var(train_labels)
            type_labels = to_var(type_labels)

            #模型返回一个链路预测结果，一个梯度反转后的类型预测结果
            link_outputs, type_outputs = model(train_data.unsqueeze(1))
            train_score += list(link_outputs[:, 1].cpu().detach().numpy())
            # train_label += list(train_labels[0].cpu().detach().numpy())
            # print('train_labels:', train_labels)
            # train_labels = train_labels[0]
            train_labels = train_labels.long()
            type_labels = type_labels.long()
            # print('link_outputs:', link_outputs, link_outputs.size())
            # print('train_labels:', train_labels, train_labels.size())
            #prediction_loss交叉熵，是链路预测部分的模型好坏程度判定，对应文章中的The generative predictor损失函数
            prediction_loss = criterion(link_outputs, train_labels)
            # classification_loss交叉熵，是类型分类部分的模型好坏程度判定，对应文章中的The discriminative classifier损失函数
            classification_loss = criterion(type_outputs, type_labels)
            #综合损失函数（Final loss）
            loss = prediction_loss + classification_loss
            loss.backward()
            optimizer.step()
            _, argmax = torch.max(link_outputs, 1)
            accuracy = (train_labels == argmax.squeeze()).float().mean()
            prediction_cost_vector.append(prediction_loss.item())
            classification_cost_vector.append(classification_loss.item())
            cost_vector.append(loss.item())
            acc_vector.append(accuracy.item())

        print('validating model')
        # validate process
        #矫正模型的偏差，跟训练流程一致，无需类型进行干扰
        model.eval()
        validate_acc_vector_temp = []
        for i, (validate_data, validate_labels,
                type_labels) in enumerate(validate_loader):
            validate_data = to_var(validate_data)
            validate_labels = to_var(validate_labels)
            type_labels = to_var(type_labels)
            validate_outputs, type_outputs = model(validate_data.unsqueeze(1))
            _, validate_argmax = torch.max(validate_outputs, 1)
            validate_labels = validate_labels.long()
            vali_loss = criterion(validate_outputs, validate_labels)
            validate_accuracy = (
                validate_labels == validate_argmax.squeeze()).float().mean()
            vali_cost_vector.append(vali_loss.item())
            validate_acc_vector_temp.append(validate_accuracy.item())
        validate_acc = np.mean(validate_acc_vector_temp)
        valid_acc_vector.append(validate_acc)
        model.train()
        print(
            'Epoch [%d/%d],  Loss: %.4f, Link Prediction Loss: %.4f, Type Classification Loss: %.4f, Train_Acc: %.4f,  Validate_Acc: %.4f.'
            % (epoch + 1, num_epochs, np.mean(cost_vector),
               np.mean(prediction_cost_vector),
               np.mean(classification_cost_vector), np.mean(acc_vector),
               validate_acc))

        #存储最好的训练精度
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
        #预测模型返回预测结果
        test_outputs, type_outputs = model(test_data.unsqueeze(1))
        #模型返回预测的数值
        tes_score += list(test_outputs[:, 1].cpu().detach().numpy())

        #真实标签
        tes_label += list(test_labels.cpu().detach().numpy())
        _, test_argmax = torch.max(test_outputs, 1)
        if i == 0:
            test_score = to_np(test_outputs)
            test_pred = to_np(test_argmax)
            test_true = to_np(test_labels)
        else:
            test_score = np.concatenate((test_score, to_np(test_outputs)),
                                        axis=0)
            test_pred = np.concatenate((test_pred, to_np(test_argmax)), axis=0)
            test_true = np.concatenate((test_true, to_np(test_labels)), axis=0)

    #三个指标，真实标签test_true，预测标签test_pred，预测得分tes_score
    test_accuracy = metrics.accuracy_score(test_true, test_pred)
    test_precision = metrics.precision_score(test_true, test_pred,
                                             average='macro')
    test_aucroc = metrics.roc_auc_score(tes_label, tes_score, average='macro')

    return test_aucroc, test_precision, test_accuracy


def main(predicted_Type, dataset, output_file):
    #拆分训练集、测试集和验证集
    train_loader, validate_loader, test_loader = get_train_validate_test(
        dataset)
    # 初始化对抗训练模型
    model = adversarial_neural_networks(predicted_Type)
    #训练模型，返回的结果是训练出来的最好的精度
    best_validate_dir = train_adversarial_neural_networks(
        train_loader, validate_loader, model, output_file)
    #测试模型，以链路预测中常用的三个指标 auc, precision, accuracy作为最终的测试结果
    auc, precision, accuracy = test_adversarial_neural_networks(
        best_validate_dir, test_loader, model)
    print("Final reault: AUC -- %.4f  " % (auc),
          "Precision -- %.4f  " % (precision),
          'Accuracy -- %.4f  ' % (accuracy))


if __name__ == '__main__':
    datasets = ['Facebook', 'IMDB', 'YELP', 'DBLP']
    dataset = datasets[0]
    print('Input dataset is:', dataset)
    predicted_Type_datasets = {'Facebook': 4, 'IMDB': 2, 'YELP': 2, 'DBLP': 2}
    predicted_Type = predicted_Type_datasets[dataset]
    print('predicted_Type:', predicted_Type)
    output_file = 'trainOutput/' + dataset.lower() + '/output.txt'
    main(predicted_Type, dataset, output_file)
