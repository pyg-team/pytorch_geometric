from scipy.io import loadmat
import numpy as np
import scipy.sparse as sp


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1)) + 0.01
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


# load data
data_name = 'YelpChi.mat'  # 'Amazon.mat' or 'YelpChi.mat'
mode = 'pos'  # if set to pos, it only compute two metrics for positive nodes
data = loadmat(data_name)

if data_name == 'YelpChi.mat':
    net_list = [data['net_rur'].nonzero(), data['net_rtr'].nonzero(),
                 data['net_rsr'].nonzero(), data['homo'].nonzero()]
else:  # amazon dataset
    net_list = [data['net_upu'].nonzero(), data['net_usu'].nonzero(),
                data['net_uvu'].nonzero(), data['homo'].nonzero()]

feature = normalize(data['features']).toarray()
label = data['label'][0]

# extract the edges of positive nodes in each relation graph
pos_nodes = set(label.nonzero()[0].tolist())
node_list = [set(net[0].tolist()) for net in net_list]
pos_node_list = [list(net_nodes.intersection(pos_nodes)) for net_nodes in node_list]
pos_idx_list = []
for net, pos_node in zip(net_list, pos_node_list):
    pos_idx_list.append(np.in1d(net[0], np.array(pos_node)).nonzero()[0])


feature_simi_list = []
label_simi_list = []
print('compute two metrics')
for net, pos_idx in zip(net_list, pos_idx_list):
    feature_simi = 0
    label_simi = 0
    if mode == 'pos':  # compute two metrics for positive nodes
        for idx in pos_idx:
            u, v = net[0][idx], net[1][idx]
            feature_simi += np.exp(-1 * np.square(np.linalg.norm(feature[u] - feature[v])))
            label_simi += label[u] == label[v]

        feature_simi = feature_simi / pos_idx.size
        label_simi = label_simi / pos_idx.size

    else:  # compute two metrics for all nodes
        for u, v in zip(net[0].tolist(), net[1].tolist()):
            feature_simi += np.exp(-1 * np.square(np.linalg.norm(feature[u] - feature[v])))
            label_simi += label[u] == label[v]

        feature_simi = feature_simi / net[0].size
        label_simi = label_simi / net[0].size

    feature_simi_list.append(feature_simi)
    label_simi_list.append(label_simi)

print(f'feature_simi: {feature_simi_list}')
print(f'label_simi: {label_simi_list}')
