import numpy as np
import networkx as nx
from node2vec import Node2Vec
from node2vec.edges import HadamardEmbedder
import pandas as pd

def get_graph(filepath):
    dataGraph = pd.read_csv(filepath, sep=' ', skiprows=0)
    graph = nx.Graph()
    edges = np.array(dataGraph)
    for edge in edges:
        print('leftNode:', edge[0], 'rightNode:', edge[1])
        graph.add_edge(edge[0], edge[1])
    return graph

def get_edge_embeddings(graph, savepath):  # node2vec生成边向量表征(64维)
    node2vec = Node2Vec(graph, dimensions=64, walk_length=5, num_walks=10, workers=1)
    model = node2vec.fit(window=10, min_count=1, batch_words=1)
    edges_embs = HadamardEmbedder(keyed_vectors=model.wv)
    edges_kv = edges_embs.as_keyed_vectors()
    edges_kv.save_word2vec_format(savepath)

def pro_data_main(dataset):
    print("Input dataset:", dataset)
    pathReal = './' + dataset + '/realData.csv'
    #only real graph to be embedded
    #pathFake = 'Data/' + dataset + '/fakeData.csv'
    graph = get_graph(pathReal) #real graph
    savepath = './node2vecFeature/' + dataset.lower()+ 'Feature.txt'  # 最终数据
    get_edge_embeddings(graph, savepath)

if __name__ == '__main__':
    datasets = ['Facebook']
    for dataset in datasets:
        pro_data_main(dataset)