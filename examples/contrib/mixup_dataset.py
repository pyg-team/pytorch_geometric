import logging
import os.path as osp
import random
import warnings

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from pyg_graphons import Graphon
from pyg_mixup_transform import MixupDataset
from utils import split_class_graphs, stat_graph

from torch_geometric.datasets import TUDataset

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore", module="torch_geometric")
"""
Testing adapted from https://github.com/eomeragic1/g-mixup-reproducibility to be compatible with our implementation.
"""


def prepare_dataset_onehot_y(dataset):
    y_set = set()
    for data in dataset:
        y_set.add(int(data.y))

    num_classes = len(y_set)
    for data in dataset:
        data.y = F.one_hot(data.y, num_classes=num_classes).to(torch.float)[0]
    return dataset


def benchmark(dataset_names: list, graphon_sizes: list, data_path='./',
              align_max_size=1000):

    for dataset_name, graphon_size in zip(dataset_names, graphon_sizes):
        logging.info(f"Looking at {dataset_name}, {graphon_size}")

        # Load and prepare the original dataset
        path = osp.join(data_path, dataset_name)
        dataset = TUDataset(path, name=dataset_name)
        dataset = list(dataset)  # Convert dataset to a list

        for graph in dataset:
            graph.y = graph.y.view(-1)

        # Apply one-hot encoding to the dataset labels
        dataset = prepare_dataset_onehot_y(dataset)
        class_graphs = split_class_graphs(dataset)

        random.seed(1314)
        random.shuffle(dataset)

        # Calculate and print graph statistics
        avg_num_nodes, avg_num_edges, avg_density, median_num_nodes, median_num_edges, median_density = stat_graph(
            dataset)
        print(' - (Orig) Median num nodes: ', int(median_num_nodes))
        print(' - (Orig) Avg num nodes: ', int(avg_num_nodes))
        print(' - (Orig) Median num edges: ', int(median_num_edges))
        print(' - (Orig) Avg num edges: ', int(avg_num_edges))
        print(' - (Orig) Median density: ', median_density)
        print(' - (Orig) Avg density: ', avg_density)

        # Prepare a figure for visualizing the graphons for each class
        fig, ax = plt.subplots(1, len(class_graphs),
                               figsize=(2 * len(class_graphs), 2),
                               facecolor='w')

        if dataset_name == 'IMDB-MULTI':
            fig.subplots_adjust(right=0.9)
            cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])

        # Test pyg_graphons by estimating graphons for each class
        est_graphons = []
        for label, graphs in class_graphs:
            g = Graphon(
                graphs=graphs,
                padding=True,
                r=graphon_size,
                label=label,
                align_max_size=align_max_size,
            )
            g._estimate()  # Estimate the graphon
            est_graphons.append((label, g.get_graphon()))
        """ Now, let's test the Mixup Dataset usage. """

        # Prepare for Mixup Dataset testing
        dataset_test = TUDataset(path, name=dataset_name)
        mixup_dataset = MixupDataset(
            dataset=dataset_test,
            specs=[(0, 1, 0.5, graphon_size, 1000, 20, 30)])

        mixup_dataset_list = list(mixup_dataset)

        # Apply one-hot encoding to the mixup dataset labels
        for graph in mixup_dataset_list:
            graph.y = graph.y.view(-1)

        mixup_dataset_list = prepare_dataset_onehot_y(mixup_dataset_list)
        class_graphs = split_class_graphs(mixup_dataset_list)

        # Shuffle the mixup dataset randomly
        random.seed(1314)
        random.shuffle(mixup_dataset_list)

        # Calculate and print mixup dataset graph statistics
        avg_num_nodes, avg_num_edges, avg_density, median_num_nodes, median_num_edges, median_density = stat_graph(
            mixup_dataset_list)
        print(' - (GMix) Median num nodes: ', int(median_num_nodes))
        print(' - (GMix) Avg num nodes: ', int(avg_num_nodes))
        print(' - (GMix) Median num edges: ', int(median_num_edges))
        print(' - (GMix) Avg num edges: ', int(avg_num_edges))
        print(' - (GMix) Median density: ', median_density)
        print(' - (GMix) Avg density: ', avg_density)

        # Plot the graphon for each class in the mixup dataset
        fig, ax = plt.subplots(1, len(class_graphs),
                               figsize=(3 * len(class_graphs), 3),
                               facecolor='w')

        for label, axis, i in zip(mixup_dataset._graphons, ax,
                                  range(len(mixup_dataset._graphons))):
            graphon = mixup_dataset._graphons[label]._graphon
            print(
                f" - Graphon info: label:{label}; mean: {graphon.mean()}, shape: {graphon.shape}"
            )
            im = axis.imshow(graphon, vmin=0, vmax=1, cmap=plt.cm.plasma)
            axis.set_title(f"Class {i}", weight="bold")
            axis.axis('off')

        if dataset_name == 'IMDB-MULTI':
            fig.colorbar(im, cax=cbar_ax, orientation='vertical')

        fig.suptitle(dataset_name, y=0.1, weight="bold")
        plt.savefig(f'../fig/{dataset_name}.png', facecolor='white',
                    bbox_inches='tight')


if __name__ == '__main__':
    d_names = ['IMDB-BINARY', 'REDDIT-BINARY', 'IMDB-MULTI']
    g_sizes = [19, 15, 12]

    benchmark(d_names, g_sizes)
