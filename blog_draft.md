# CS 224W Final Project: G-Mixup in PyG

## Motivation & explanation of research method

Data augmentation is a technique in machine learning that aims to increase the quantity and diversity of training data without collecting more samples. It is widely studied in image classification tasks, where it is common to apply transformations such as rotations, flips, adding small noise, or other perturbations to gathered training data. This effectively expands the training data which helps models generalize better while avoiding the effort of data collection. A more advanced data augmentation technique for image data taht goes beyond simple transformations is **mixup**, which involves adding new training examples which are convex combinations of existing training examples. The labels of new training examples are combined in the same way. This method was introduced by Zhang et al \cite{https://arxiv.org/pdf/1710.09412} in 2018 and has been shown to improve generalization of image classification models and stabilize the training of GANs.

The success of **mixup** in the image domain leads to the question of whether a similar technique can be applied to graph data in order to help graph neural networks generalize better. However, this is not a straight-forward task: graph data is fundamentally different from image data. Image data is well-structured: pixels are well ordered and follow a regular grid structure. Images can be represented in Euclidean space, allowing easy convex combinations of different points through simply interpolating 2 images. However, graphs are generally irregular objects and there is no natural ordering of nodes. Graphs have divergent topologies and cannot be represented in Euclidean space, meaning there is no way to "line up" two graphs and interpolate between them.

In the 2022 paper **G-Mixup: Graph Data Augmentation for Graph Classification** by Xiaotian Han, Zhimeng Jiang, Ninghao Liu, and Xia Hu, the authors introduce a technique that interpolates between different graph classes instead of individual examples of graphs by learning graphons that represent each class and interpolating those. Graphons are mathematical objects that can be thought of as generators for random graphs, which can be approximated by a matrix-like step function. Since matrices are well-aligned, one can interpolate between the approximated graphons to create a synthetic, mixed graphon which can be used to generate "mixed-label" graphs.

**G-Mixup** has three steps:
1. Estimate a graphon for each class of graphs.
2. Mix up the graphons of different graph classes.
3. Generate synthetic mixed label graphs based on the mixed graphons.

When used on large datasets, GNNs frequently suffer from overfitting and difficulty generalizing. G-Mixup provides a data augmentation method to generate new training data graphs, which can help to tackle this problem. It is also important to note that G-Mixup generates completely new, synthetic graphs rather than just modifying existing input graphs like many other graph data augmentation techniques. It has the potential to greatly improve the generalizability of GNNs.


## Implementation in PyG

We implement G-Mixup via the class "GMixupDataset," a wrapper class that takes in a PyG Dataset and applies the G-Mixup data augmentation technique to it. Via an override of the __getattr__ method, the GMixupDataset class can behave like and be used the same way as a PyG Dataset.

On initialization, GMixupDataset takes in:
* base_dataset: the PyG Dataset to be augmented
* log: (optional, default True): whether to print any console output while processing the dataset
* align_graphs: (optional, default True): whether to align the graphs by node degree before generating graphons. This is generally recommended to ensure graphon invariance to node ordering, but can be turned off for speed if it is known that the input graphs are already aligned.
* threshold: (optional, default 2.02): the threshold to use for singular value thresholding when generating graphons. This is a hyperparameter that typically ranges from 2 to 3, and can be tuned for better performance.
* generate_graphons: (optional, default True): if True, graphons for every class of graphs in the dataset will be generated and stored in the GMixupDataset object during initialization. If false, graphons will not be generated until a synthetic graph involving that class is requested, or a manual call to generate_graphons() is made. This can be useful if it is known that only a subset of the graphons will be used.

During initialization, GMixupDataset does the following (assuming generate_graphons is True):
1. Store all the graphs sorted by class in the list self.graphs_by_class
2. For each class of graphs, generate a graphon approximation using the method described in the paper "Matrix estimation by Universal Singular Value Thresholding" by Chatterjee (2015).
    a. For each graph in the class, compute the adjacency matrix A
    b. If align_graphs is True, sort the nodes of A by degree
    c. Normalize A to be in the range [-1, 1]
    d. Compute the singular value decomposition of A
    e. Threshold the singular values using the following formula as described in the paper:
        (attach screenshot)
    f. Compute the graphon approximation W using the formula W = UΣ'V^T, where Σ' is the thresholded singular value matrix
    g. Clip W to be in range [-1, 1], renormalize it to be in range [0, 1], then interpolate the matrix to the maximum size of graphs in the dataset. This is to ensure that each graphon has the same dimensions and can be mixed with other graphons.*
    h. Average the graphons of all graphs in the same class to get the final graphon approximation for that class, and store it in self.graphons.
* Note that the paper "GMixup: Graph Data Augmentation for Graph Classification" by Han et al. (2022) expresses a preference to use the average node count of the dataset as the size of the graphon approximation matrices, as opposed to the maximum node count of the dataset as we have done. We found that using the maximum node count yielded better and more consistent results in our experiments.

A user can request a synthetic graph from the GMixupDataset object by calling generate_graphs, which takes in the following arguments:
* size: (optional, default 1) the number of synthetic graphs to generate
* idx_1: the index of the first class of graphs to mix
* idx_2: the index of the second class of graphs to mix
* mixing_param: (optional, default 0.5) the mixing parameter to use when interpolating the graphons of the two classes. This parameter can be varied between 0 and 1 to generate graphs that are more similar to the first class (when closer to 0) or more similar to the second class (when closer to 1).
* K (optional, default 10): the number of nodes to sample for the output synthetic graph.
* method (optional, default 'random'): the method to use for sampling nodes from the graphon. Can be 'random' to sample nodes uniformly at random, or 'uniform' to guarantee a uniform distribution of nodes across the graphon.

When a synthetic graph is requested, GMixupDataset does the following:
1. Get the graphons of the two classes specified by idx_1 and idx_2
2. Interpolate the graphons using the mixing parameter to get a new, mixed_graphon = mixing_param * graphon_1 + (1 - mixing_param) * graphon_2
3. Sample a K node graph from the mixed_graphon:
    a. Generate K "u_values" in the interval [0, 1], either uniformly at random if method is 'random', or uniformly spaced if method is 'uniform'. Multiply these values by the size of the graphon matrix to get the corresponding index into the graphon matrix. We refer to these as 
    "u_values_indices".
    b. For any pair of nodes i, j, mixed_graphon[ u_values_index_i, u_values_index_j ] is the probability that an edge exists between nodes i and j. Sample a Bernoulli random variable with this probability to determine if an edge exists between nodes i and j for all pairs of nodes. This gives us the adjacency matrix of the synthetic graph.
    c. Generate the synthetic graph using the adjacency matrix.
4. Repeat the above process size times to generate size synthetic graphs.


For example, to use the GMixupDataset class to augment the GNNBenchmarkDataset dataset, one would initialize it as follows:
```python
gnn_benchmark = GNNBenchmarkDataset(root='tmp/gnn_benchmark', name='CSL')
gmixup_gnn_benchmark = GMixupDataset(gnn_benchmark, log=True, align_graphs=True, threshold=2.02, generate_graphons=True)
```

To generate a 40-node synthetic graph between the first and second classes of graphs in the dataset, one would call:
```python
synthetic_graph = gmixup_gnn_benchmark.generate_graphs(idx_1=0, idx_2=1, mixing_param=0.5, K=40, method='random', size=1)





TODO
Code snippets (5 points)