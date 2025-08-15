# The below is to suppress the warning on torch.nn.conv.MeshCNNConv::update
# pyright: reportIncompatibleMethodOverride=false
import warnings
from typing import Optional

import torch
from torch.nn import Linear, Module, ModuleList

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Tensor


class MeshCNNConv(MessagePassing):
    r"""The convolutional layer introduced by the paper
    `"MeshCNN: A Network With An Edge" <https://arxiv.org/abs/1809.05910>`_.

    Recall that, given a set of categories :math:`C`,
    MeshCNN is a function that takes as its input
    a triangular mesh
    :math:`\mathcal{m} = (V, F) \in \mathbb{R}^{|V| \times 3} \times
    \{0,...,|V|-1\}^{3 \times |F|}`, and returns as its output
    a :math:`|C|`-dimensional vector, whose :math:`i` th component denotes
    the probability of the input mesh belonging to category :math:`c_i \in C`.

    Let :math:`X^{(k)} \in \mathbb{R}^{|E| \times \text{Dim-Out}(k)}`
    denote the output value of the prior (e.g. :math:`k` th )
    layer of our neural network. The :math:`i` th row of :math:`X^{(k)}` is a
    :math:`\text{Dim-Out}(k)`-dimensional vector that represents the features
    computed by the :math:`k` th layer for edge :math:`e_i` of the input mesh
    :math:`\mathcal{m}`. Let :math:`A \in \{0, ..., |E|-1\}^{2 \times 4*|E|}`
    denote the *edge adjacency* matrix of our input mesh :math:`\mathcal{m}`.
    The :math:`j` th column of :math:`A` returns a pair of indices
    :math:`k,l \in \{0,...,|E|-1\}`, which means that edge
    :math:`e_k` is adjacent to edge :math:`e_l`
    in our input mesh :math:`\mathcal{m}`.
    The definition of edge adjacency in a triangular
    mesh is illustrated in Figure 1.
    In a triangular
    mesh, each edge :math:`e_i` is expected to be adjacent to exactly :math:`4`
    neighboring edges, hence the number of columns of :math:`A`: :math:`4*|E|`.
    We write *the neighborhood* of edge :math:`e_i` as
    :math:`\mathcal{N}(i) = (a(i), b(i), c(i), d(i))` where

    1. :math:`a(i)` denotes the index of the *first* counter-clockwise
    edge of the face *above* :math:`e_i`.

    2. :math:`b(i)` denotes the index of the *second* counter-clockwise
    edge of the face *above* :math:`e_i`.

    3. :math:`c(i)` denotes the index of the *first* counter-clockwise edge
    of the face *below* :math:`e_i`.

    4. :math:`d(i)` denotes the index of the *second*
    counter-clockwise edge of the face *below* :math:`e_i`.

    .. figure:: ../_figures/meshcnn_edge_adjacency.svg
        :align: center
        :width: 80%

        **Figure 1:** The neighbors of edge :math:`\mathbf{e_1}`
        are :math:`\mathbf{e_2}, \mathbf{e_3}, \mathbf{e_4}` and
        :math:`\mathbf{e_5}`, respectively.
        We write this as
        :math:`\mathcal{N}(1) = (a(1), b(1), c(1), d(1)) = (2, 3, 4, 5)`


    Because of this ordering constrait, :obj:`MeshCNNConv` **requires
    that the columns of** :math:`A`
    **be ordered in the following way**:

    .. math::
        &A[:,0] = (0, \text{The index of the "a" edge for edge } 0) \\
        &A[:,1] = (0, \text{The index of the "b" edge for edge } 0) \\
        &A[:,2] = (0, \text{The index of the "c" edge for edge } 0) \\
        &A[:,3] = (0, \text{The index of the "d" edge for edge } 0) \\
        \vdots \\
        &A[:,4*|E|-4] =
            \bigl(|E|-1,
                a\bigl(|E|-1\bigr)\bigr) \\
        &A[:,4*|E|-3] =
            \bigl(|E|-1,
                b\bigl(|E|-1\bigr)\bigr) \\
        &A[:,4*|E|-2] =
            \bigl(|E|-1,
                c\bigl(|E|-1\bigr)\bigr) \\
        &A[:,4*|E|-1] =
            \bigl(|E|-1,
                d\bigl(|E|-1\bigr)\bigr)


    Stated a bit more compactly, for every edge :math:`e_i` in the input mesh,
    :math:`A`, should have the following entries

    .. math::
        A[:, 4*i] &= (i, a(i)) \\
        A[:, 4*i + 1] &= (i, b(i)) \\
        A[:, 4*i + 2] &= (i, c(i)) \\
        A[:, 4*i + 3] &= (i, d(i))

    To summarize so far, we have defined 3 things:

    1. The activation of the prior (e.g. :math:`k` th) layer,
    :math:`X^{(k)} \in \mathbb{R}^{|E| \times \text{Dim-Out}(k)}`

    2. The edge adjacency matrix and the definition of edge adjacency.
    :math:`A \in \{0,...,|E|-1\}^{2 \times 4*|E|}`

    3. The ways the columns of :math:`A` must be ordered.



    We are now finally able to define the :obj:`MeshCNNConv` class/layer.
    In the following definition
    we assume :obj:`MeshCNNConv` is at the :math:`k+1` th layer of our
    neural network.

    The :obj:`MeshCNNConv` layer is a function,

    .. math::
        \text{MeshCNNConv}^{(k+1)}(X^{(k)}, A) = X^{(k+1)},

    that, given the prior layer's output
    :math:`X^{(k)} \in \mathbb{R}^{|E| \times \text{Dim-Out}(k)}`
    and the edge adjacency matrix :math:`A`
    of the input mesh (graph) :math:`\mathcal{m}` ,
    returns a new edge feature tensor
    :math:`X^{(k+1)} \in \mathbb{R}^{|E| \times \text{Dim-Out}(k+1)}`,
    where the :math:`i` th row of :math:`X^{(k+1)}`, denoted by
    :math:`x^{(k+1)}_i`,
    represents the :math:`\text{Dim-Out}(k+1)`-dimensional feature vector
    of edge :math:`e_i`, **and is defined as follows**:

    .. math::
        x^{(k+1)}_i &= W^{(k+1)}_0 x^{(k)}_i \\
        &+ W^{(k+1)}_1 \bigl| x^{(k)}_{a(i)} - x^{(k)}_{c(i)} \bigr| \\
        &+ W^{(k+1)}_2 \bigl( x^{(k)}_{a(i)} + x^{(k)}_{c(i)} \bigr) \\
        &+ W^{(k+1)}_3 \bigl| x^{(k)}_{b(i)} - x^{(k)}_{d(i)} \bigr| \\
        &+ W^{(k+1)}_4 \bigl( x^{(k)}_{b(i)} + x^{(k)}_{d(i)} \bigr).

    :math:`W_0^{(k+1)},W_1^{(k+1)},W_2^{(k+1)},W_3^{(k+1)}, W_4^{(k+1)}
    \in \mathbb{R}^{\text{Dim-Out}(k+1) \times \text{Dim-Out}(k)}`
    are trainable linear functions (i.e. "the weights" of this layer).
    :math:`x_i` is the :math:`\text{Dim-Out}(k)`-dimensional feature of
    edge :math:`e_i` vector computed by the prior (e.g. :math:`k`) th layer.
    :math:`x^{(k)}_{a(i)}, x^{(k)}_{b(i)}, x^{(k)}_{c(i)}`, and
    :math:`x^{(k)}_{d(i)}` are the :math:`\text{Dim-Out}(k)`-feature vectors,
    computed in the :math:`k` th layer, that are associated with the :math:`4`
    neighboring edges of :math:`e_i`.


    Args:
        in_channels (int): Corresonds to :math:`\text{Dim-Out}(k)`
            in the above overview. This
            represents the output dimension of the prior layer. For the given
            input mesh :math:`\mathcal{m} = (V, F)`, the prior layer is
            expected to output a
            :math:`X \in \mathbb{R}^{|E| \times \textit{in_channels}}`
            feature matrix.
            Assuming the instance of this class
            is situated at layer :math:`k+1`, we write that
            :math:`X^{(k)} \in \mathbb{R}^{|E| \times \textit{in_channels}}`.
        out_channels (int): Corresponds to :math:`\text{Dim-Out}(k+1)` in the
            above overview. This represents the output dimension of this layer.
            Assuming the instance of this class
            is situated at layer :math:`k+1`, we write that
            :math:`X^{(k+1)}
            \in \mathbb{R}^{|E| \times \textit{out_channels}}`.
        kernels (torch.nn.ModuleList, optional): A list of length of 5,
            where each
            element is a :class:`torch.nn.module` (i.e a neural network),
            that each MUST take as input a vector
            of dimension :`obj:in_channels` and return a vector of dimension
            :obj:`out_channels`. In particular,
            `obj:kernels[0]` is :math:`W^{(k+1)}_0` in the above overview
            (see :obj:`MeshCNNConv`), `obj:kernels[1]` is :math:`W^{(k+1)}_1`,
            `obj:kernels[2]` is :math:`W^{(k+1)}_2`,
            `obj:kernels[3]` is :math:`W^{(k+1)}_3`
            `obj:kernels[4]` is :math:`W^{(k+1)}_4`.
            Note that this input is optional, in which case
            each of the 5 elements in the kernels will be a linear
            neural network :class:`torch.nn.modules.Linear`
            correctly configured to take as input
            :attr:`in_channels`-dimensional vectors and return
            a vector of dimensions :attr:`out_channels`.

    Discussion:
        The key difference that seperates :obj:`MeshCNNConv` from a traditional
        message passing graph neural network is that :obj:`MeshCNNConv`
        requires the set of neighbors for a node
        :math:`\mathcal{N}(u) = (v_1, v_2, ...)`
        to *be an ordered set* (i.e. a tuple). In
        fact, :obj:`MeshCNNConv` goes further, requiring
        that :math:`\mathcal{N}(u)` always return a set of size :math:`4`.
        This is different to most message passing graph neural networks,
        which assume that :math:`\mathcal{N}(u) = \{v_1, v_2, ...\}` returns an
        ordered set. This lends :obj:`MeshCNNConv` more expressive power,
        at the cost of no longer being permutation invariant to
        :math:`\mathbb{S}_4`. Put more plainly, in tradition message passing
        GNNs, the network is *unable* to distinguish one neighboring node
        from another.
        In constrast, in :obj:`MeshCNNConv`, each of the 4 neighbors has a
        "role", either the "a", "b", "c", or "d" neighbor. We encode this fact
        by requiring that :math:`\mathcal{N}` return the 4-tuple,
        where the first component is the "a" neighbor, and so on.

        To summarize this comparison, it may re-define
        :obj:`MeshCNNConv` in terms of :math:`\text{UPDATE}` and
        :math:`\text{AGGREGATE}`
        functions, which is a general way to define a traditional GNN layer.
        If we let :math:`x_i^{(k+1)}`
        denote the output of a GNN layer for node :math:`i` at
        layer :math:`k+1`, and let
        :math:`\mathcal{N}(i)` denote the set of nodes adjacent
        to node :math:`i`,
        then we can describe the :math:`k+1` th layer as traditional GNN
        as

        .. math::
            x_i^{(k+1)} = \text{UPDATE}^{(k+1)}\bigl(x^{(k)}_i,
            \text{AGGREGATE}^{(k+1)}\bigl(\mathcal{N}(i)\bigr)\bigr).

        Here, :math:`\text{UPDATE}^{(k+1)}` is a function of :math:`2`
        :math:`\text{Dim-Out}(k)`-dimensional vectors, and returns a
        :math:`\text{Dim-Out}(k+1)`-dimensional vector.
        :math:`\text{AGGREGATE}^{(k+1)}` function
        is a function of a *unordered set*
        of nodes that are neighbors of node :math:`i`, as defined by
        :math:`\mathcal{N}(i)`. Usually the size of this set varies across
        different nodes :math:`i`, and one of the most basic examples
        of such a function is the "sum aggregation", defined as
        :math:`\text{AGGREGATE}^{(k+1)}(\mathcal{N}(i)) =
        \sum_{j \in \mathcal{N}(i)} x^{(k)}_j`.
        See
        :class:`SumAggregation <torch_geometric.nn.aggr.basic.SumAggregation>`
        for more.

        In contrast, while :obj:`MeshCNNConv` 's :math:`\text{UPDATE}`
        function follows
        a tradition GNN, its :math:`\text{AGGREGATE}` is a function of a tuple
        (i.e. an ordered set) of neighbors
        rather than a unordered set of neighbors.
        In particular, while the :math:`\text{UPDATE}`
        function of :obj:`MeshCNNConv` for :math:`e_i` is

        .. math::
            x_i^{(k+1)} = \text{UPDATE}^{(k+1)}(x_i^{(k)}, s_i^{(k+1)})
            = W_0^{(k+1)}x_i^{(k)} + s_i^{(k+1)},

        in contrast, :obj:`MeshCNNConv` 's :math:`\text{AGGREGATE}` function is

        .. math::
            s_i^{(k+1)} = \text{AGGREGATE}^{(k+1)}(A, B, C, D)
            &= W_1^{(k+1)}\bigl|A - C \bigr| \\
            &= W_2^{(k+1)}\bigl(A + C \bigr) \\
            &= W_3^{(k+1)}\bigl|B - D \bigr| \\
            &= W_4^{(k+1)}\bigl(B + D \bigr),

        where :math:`A=x_{a(i)}^{(k)}, B=x_{b(i)}^{(k)}, C=x_{c(i)}^{(k)},`
        and :math:`D=x_{d(i)}^{(k)}`.

        ..

            The :math:`i` th row of
            :math:`V \in \mathbb{R}^{|V| \times 3}`
            holds the cartesian :math:`xyz`
            coordinates for node :math:`v_i` in the mesh, and the :math:`j` th
            column in :math:`F \in \{1,...,|V|\}^{3 \times |V|}`
            holds the :math:`3` indices
            :math:`(k,l,m)` that correspond to the :math:`3` nodes
            :math:`(v_k, v_l, v_m)` that construct face :math:`j` of the mesh.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernels: Optional[ModuleList] = None):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels

        if kernels is None:
            self.kernels = ModuleList(
                [Linear(in_channels, out_channels) for _ in range(5)])

        else:
            # ensures kernels is properly formed, otherwise throws
            # the appropriate error.
            self._assert_kernels(kernels)
            self.kernels = kernels

    def forward(self, x: Tensor, edge_index: Tensor):
        r"""Forward pass.

        Args:
            x(torch.Tensor): :math:`X^{(k)} \in
                \mathbb{R}^{|E| \times \textit{in_channels}}`.
                The edge feature tensor returned by the prior layer
                (e.g. :math:`k`). The tensor is of shape
                :math:`|E| \times \text{Dim-Out}(k)`, or equivalently,
                :obj:`(|E|, self.in_channels)`.

            edge_index(torch.Tensor):
                :math:`A \in \{0,...,|E|-1\}^{2 \times 4*|E|}`.
                The edge adjacency tensor of the networks input mesh
                :math:`\mathcal{m} = (V, F)`. The edge adjacency tensor
                **MUST** have the following form:

                .. math::
                    &A[:,0] = (0,
                        \text{The index of the "a" edge for edge } 0) \\
                    &A[:,1] = (0,
                        \text{The index of the "b" edge for edge } 0) \\
                    &A[:,2] = (0,
                        \text{The index of the "c" edge for edge } 0) \\
                    &A[:,3] = (0,
                        \text{The index of the "d" edge for edge } 0) \\
                    \vdots \\
                    &A[:,4*|E|-4] =
                        \bigl(|E|-1,
                            a\bigl(|E|-1\bigr)\bigr) \\
                    &A[:,4*|E|-3] =
                        \bigl(|E|-1,
                            b\bigl(|E|-1\bigr)\bigr) \\
                    &A[:,4*|E|-2] =
                        \bigl(|E|-1,
                            c\bigl(|E|-1\bigr)\bigr) \\
                    &A[:,4*|E|-1] =
                        \bigl(|E|-1,
                            d\bigl(|E|-1\bigr)\bigr)

                See :obj:`MeshCNNConv` for what
                "index of the 'a'(b,c,d) edge for edge i" means, and also
                for the general definition of edge adjacency in MeshCNN.
                These definitions are also provided in the
                `paper <https://arxiv.org/abs/1809.05910>`_ itself.

        Returns:
           torch.Tensor:
           :math:`X^{(k+1)} \in \mathbb{R}^{|E| \times \textit{out_channels}}`.
           The edge feature tensor for this (e.g. the :math:`k+1` th) layer.
           The :math:`i` th row of :math:`X^{(k+1)}` is computed according
           to the formula

            .. math::
                x^{(k+1)}_i &= W^{(k+1)}_0 x^{(k)}_i \\
                &+ W^{(k+1)}_1 \bigl| x^{(k)}_{a(i)} - x^{(k)}_{c(i)} \bigr| \\
                &+ W^{(k+1)}_2 \bigl( x^{(k)}_{a(i)} + x^{(k)}_{c(i)} \bigr) \\
                &+ W^{(k+1)}_3 \bigl| x^{(k)}_{b(i)} - x^{(k)}_{d(i)} \bigr| \\
                &+ W^{(k+1)}_4 \bigl( x^{(k)}_{b(i)} + x^{(k)}_{d(i)} \bigr),

            where :math:`W_0^{(k+1)},W_1^{(k+1)},
            W_2^{(k+1)},W_3^{(k+1)}, W_4^{(k+1)}
            \in \mathbb{R}^{\text{Dim-Out}(k+1) \times \text{Dim-Out}(k)}`
            are the trainable linear functions (i.e. the trainable
            "weights") of this layer, and
            :math:`x^{(k)}_{a(i)}, x^{(k)}_{b(i)}, x^{(k)}_{c(i)}`,
            :math:`x^{(k)}_{d(i)}` are the
            :math:`\text{Dim-Out}(k)`-dimensional edge feature vectors
            computed by the prior (:math:`k` th) layer,
            that are associated with the :math:`4`
            neighboring edges of :math:`e_i`.

        """
        return self.propagate(edge_index, x=x)

    def message(self, x_j: Tensor) -> Tensor:
        r"""The messaging passing step of :obj:`MeshCNNConv`.


        Args:
          x_j: A :obj:`[4*|E|, num_node_features]` tensor.
          Its ith row holds the value
            stored by the source node in the previous layer of edge i.

        Returns:
            A :obj:`[|E|, num_node_features]` tensor,
            whose ith row will be the value
            that the target node of edge i will receive.
        """
        # The following variables names are taken from the paper
        # MeshCNN computes the features associated with edge
        # e by (|a - c|, a + c, |b - c|, b + c), where a, b, c, d are the
        # neighboring edges of e, a being the 1 edge of the upper face,
        # b being the second edge of the upper face, c being the first edge
        # of the lower face,
        # and d being the second edge of the lower face of the input Mesh

        # TODO: It is unclear  if view is faster. If it is not,
        # then we should prefer the strided method commented out below

        E4, in_channels = x_j.size()  # E4 = 4|E|, i.e. num edges in line graph
        # Option 1
        n_a = x_j[0::4]  # shape: |E| x in_channels
        n_b = x_j[1::4]  # shape: |E| x in_channels
        n_c = x_j[2::4]  # shape: |E| x in_channels
        n_d = x_j[3::4]  # shape: |E| x in_channels
        m = torch.empty(E4, self.out_channels)
        m[0::4] = self.kernels[1].forward(torch.abs(n_a - n_c))
        m[1::4] = self.kernels[2].forward(n_a + n_c)
        m[2::4] = self.kernels[3].forward(torch.abs(n_b - n_d))
        m[3::4] = self.kernels[4].forward(n_b + n_d)
        return m

        # Option 2
        # E4, in_channels = x_j.size()
        # E = E4 // 4
        # x_j = x_j.view(E, 4, in_channels)  # shape: (|E| x 4 x in_channels)
        # n_a, n_b, n_c, n_d = x_j.unbind(
        #     dim=1)  # shape: (4 x |E| x in_channels)
        # m = torch.stack(
        #     [
        #         (n_a - n_c).abs(),  # shape: |E| x in_channels
        #         n_a + n_c,
        #         (n_b - n_d).abs(),
        #         n_b + n_d,
        #     ],
        #     dim=1)  # shape: (|E| x 4 x in_channels)
        # m.view(E4, in_channels)  # shape 4*|E| x in_channels
        # return m

    def update(self, inputs: Tensor, x: Tensor) -> Tensor:
        r"""The UPDATE step, in reference to the UPDATE and AGGREGATE
        formulation of message passing convolution.

        Args:
           inputs(torch.Tensor): The :attr:`in_channels`-dimensional vector
            returned by aggregate.
           x(torch.Tensor): :math:`X^{(k)}`. The original inputs to this layer.

        Returns:
            torch.Tensor: :math:`X^{(k+1)}`. The output of this layer, which
            has shape :obj:`(|E|, out_channels)`.
        """
        return self.kernels[0].forward(x) + inputs

    def _assert_kernels(self, kernels: ModuleList):
        r"""Ensures that :obj:`kernels` is a list of 5 :obj:`torch.nn.Module`
        modules (i.e. networks). In addition, it also ensures that each network
        takes in input of dimension :attr:`in_channels`, and returns output
        of dimension :attr:`out_channels`.
        This method throws an error otherwise.

        .. warn::
            This method throws an error if :obj:`kernels` is
            not valid. (Otherwise this method returns nothing)

        """
        assert isinstance(kernels, ModuleList), \
            f"Parameter 'kernels' must be a \
            torch.nn.module.ModuleList with 5 memebers, but we got \
            {type(kernels)}."

        assert len(kernels) == 5, "Parameter 'kernels' must be a \
            torch.nn.module.ModuleList of with exactly 5 members"

        for i, network in enumerate(kernels):
            assert isinstance(network, Module), \
                f"kernels[{i}] must be torch.nn.Module, got \
                {type(network)}"
            if not hasattr(network, "in_channels") and \
                    not hasattr(network, "in_features"):
                warnings.warn(
                    f"kernel[{i}] does not have attribute 'in_channels' nor "
                    f"'out_features'. The network must take as input a "
                    f"{self.in_channels}-dimensional tensor.", stacklevel=2)
            else:
                input_dimension = getattr(network, "in_channels",
                                          network.in_features)
                assert input_dimension == self.in_channels, f"The input \
                dimension of the neural network in kernel[{i}] must \
                be \
                equal to 'in_channels', but input_dimension = \
                {input_dimension}, and \
                self.in_channels={self.in_channels}."

            if not hasattr(network, "out_channels") and \
                    not hasattr(network, "out_features"):
                warnings.warn(
                    f"kernel[{i}] does not have attribute 'in_channels' nor "
                    f"'out_features'. The network must take as input a "
                    f"{self.in_channels}-dimensional tensor.", stacklevel=2)
            else:
                output_dimension = getattr(network, "out_channels",
                                           network.out_features)
                assert output_dimension == self.out_channels, f"The output \
                    dimension of the neural network in kernel[{i}] must \
                    be \
                    equal to 'out_channels', but out_dimension = \
                    {output_dimension}, and \
                    self.out_channels={self.out_channels}."
