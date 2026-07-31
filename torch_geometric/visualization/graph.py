from math import sqrt
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from torch import Tensor

BACKENDS = {'graphviz', 'networkx'}


def has_graphviz() -> bool:
    try:
        import graphviz
    except ImportError:
        return False

    try:
        graphviz.Digraph().pipe()
    except graphviz.backend.ExecutableNotFound:
        return False

    return True


def visualize_graph(
    edge_index: Tensor,
    edge_weight: Optional[Tensor] = None,
    path: Optional[str] = None,
    backend: Optional[str] = None,
    node_labels: Optional[List[str]] = None,
) -> Any:
    r"""Visualizes the graph given via :obj:`edge_index` and (optional)
    :obj:`edge_weight`.

    Args:
        edge_index (torch.Tensor): The edge indices.
        edge_weight (torch.Tensor, optional): The edge weights.
        path (str, optional): The path to where the plot is saved.
            If set to :obj:`None`, will visualize the plot on-the-fly.
            (default: :obj:`None`)
        backend (str, optional): The graph drawing backend to use for
            visualization (:obj:`"graphviz"`, :obj:`"networkx"`).
            If set to :obj:`None`, will use the most appropriate
            visualization backend based on available system packages.
            (default: :obj:`None`)
        node_labels (List[str], optional): The labels/IDs of nodes.
            (default: :obj:`None`)
    """
    if edge_weight is not None:  # Normalize edge weights.
        edge_weight = edge_weight - edge_weight.min()
        edge_weight = edge_weight / edge_weight.max()

    if edge_weight is not None:  # Discard any edges with zero edge weight:
        mask = edge_weight > 1e-7
        edge_index = edge_index[:, mask]
        edge_weight = edge_weight[mask]

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1))

    if backend is None:
        backend = 'graphviz' if has_graphviz() else 'networkx'

    if backend.lower() == 'networkx':
        return _visualize_graph_via_networkx(edge_index, edge_weight, path,
                                             node_labels)
    elif backend.lower() == 'graphviz':
        return _visualize_graph_via_graphviz(edge_index, edge_weight, path,
                                             node_labels)

    raise ValueError(f"Expected graph drawing backend to be in "
                     f"{BACKENDS} (got '{backend}')")


def _visualize_graph_via_graphviz(
    edge_index: Tensor,
    edge_weight: Tensor,
    path: Optional[str] = None,
    node_labels: Optional[List[str]] = None,
) -> Any:
    import graphviz

    suffix = path.split('.')[-1] if path is not None else None
    g = graphviz.Digraph('graph', format=suffix)
    g.attr('node', shape='circle', fontsize='11pt')

    for node in edge_index.view(-1).unique().tolist():
        g.node(str(node) if node_labels is None else node_labels[node])

    for (src, dst), w in zip(edge_index.t().tolist(), edge_weight.tolist()):
        hex_color = hex(255 - round(255 * w))[2:]
        hex_color = f'{hex_color}0' if len(hex_color) == 1 else hex_color
        if node_labels is not None:
            src = node_labels[src]
            dst = node_labels[dst]
        g.edge(str(src), str(dst), color=f'#{hex_color}{hex_color}{hex_color}')

    if path is not None:
        path = '.'.join(path.split('.')[:-1])
        g.render(path, cleanup=True)
    else:
        g.view()

    return g


def _visualize_graph_via_networkx(
    edge_index: Tensor,
    edge_weight: Tensor,
    path: Optional[str] = None,
    node_labels: Optional[List[str]] = None,
) -> Any:
    import matplotlib.pyplot as plt
    import networkx as nx

    g = nx.DiGraph()
    node_size = 800

    for node in edge_index.view(-1).unique().tolist():
        g.add_node(node if node_labels is None else node_labels[node])

    for (src, dst), w in zip(edge_index.t().tolist(), edge_weight.tolist()):
        if node_labels is not None:
            src = node_labels[src]
            dst = node_labels[dst]
        g.add_edge(src, dst, alpha=w)

    ax = plt.gca()
    pos = nx.spring_layout(g)
    for src, dst, data in g.edges(data=True):
        ax.annotate(
            '',
            xy=pos[src],
            xytext=pos[dst],
            arrowprops=dict(
                arrowstyle="<-",
                alpha=data['alpha'],
                shrinkA=sqrt(node_size) / 2.0,
                shrinkB=sqrt(node_size) / 2.0,
                connectionstyle="arc3,rad=0.1",
            ),
        )

    nx.draw_networkx_nodes(g, pos, node_size=node_size, node_color='white',
                           margins=0.1, edgecolors='black')
    nx.draw_networkx_labels(g, pos, font_size=10)

    if path is not None:
        plt.savefig(path)
    else:
        plt.show()

    plt.close()


def visualize_hetero_graph(
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
        edge_weight_dict: Dict[Tuple[str, str, str], Tensor],
        path: Optional[str] = None,
        backend: Optional[str] = None,
        node_labels_dict: Optional[Dict[str, List[str]]] = None,
        node_weight_dict: Optional[Dict[str, Tensor]] = None,
        node_size_range: Tuple[float, float] = (50, 500),
        node_opacity_range: Tuple[float, float] = (1.0, 1.0),
        edge_width_range: Tuple[float, float] = (0.1, 2.0),
        edge_opacity_range: Tuple[float, float] = (1.0, 1.0),
) -> Any:
    """Visualizes a heterogeneous graph using networkx."""
    if backend is not None and backend != "networkx":
        raise ValueError("Only 'networkx' backend is supported")

    # Filter out edges with 0 weight
    filtered_edge_index_dict = {}
    filtered_edge_weight_dict = {}
    for edge_type in edge_index_dict.keys():
        mask = edge_weight_dict[edge_type] > 0
        if mask.sum() > 0:
            filtered_edge_index_dict[edge_type] = edge_index_dict[
                edge_type][:, mask]
            filtered_edge_weight_dict[edge_type] = edge_weight_dict[edge_type][
                mask]

    # Get all unique nodes that are still in the filtered edges
    remaining_nodes: Dict[str, Set[int]] = {}
    for edge_type, edge_index in filtered_edge_index_dict.items():
        src_type, _, dst_type = edge_type
        if src_type not in remaining_nodes:
            remaining_nodes[src_type] = set()
        if dst_type not in remaining_nodes:
            remaining_nodes[dst_type] = set()
        remaining_nodes[src_type].update(edge_index[0].tolist())
        remaining_nodes[dst_type].update(edge_index[1].tolist())

    # Filter node weights to only include remaining nodes
    if node_weight_dict is not None:
        filtered_node_weight_dict = {}
        for node_type, weights in node_weight_dict.items():
            if node_type in remaining_nodes:
                mask = torch.zeros(len(weights), dtype=torch.bool)
                mask[list(remaining_nodes[node_type])] = True
                filtered_node_weight_dict[node_type] = weights[mask]
        node_weight_dict = filtered_node_weight_dict

    # Filter node labels to only include remaining nodes
    if node_labels_dict is not None:
        filtered_node_labels_dict = {}
        for node_type, labels in node_labels_dict.items():
            if node_type in remaining_nodes:
                filtered_node_labels_dict[node_type] = [
                    label for i, label in enumerate(labels)
                    if i in remaining_nodes[node_type]
                ]
        node_labels_dict = filtered_node_labels_dict

    return _visualize_hetero_graph_via_networkx(
        filtered_edge_index_dict,
        filtered_edge_weight_dict,
        path,
        node_labels_dict,
        node_weight_dict,
        node_size_range,
        node_opacity_range,
        edge_width_range,
        edge_opacity_range,
    )


def _visualize_hetero_graph_via_networkx(
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
        edge_weight_dict: Dict[Tuple[str, str, str], Tensor],
        path: Optional[str] = None,
        node_labels_dict: Optional[Dict[str, List[str]]] = None,
        node_weight_dict: Optional[Dict[str, Tensor]] = None,
        node_size_range: Tuple[float, float] = (50, 500),
        node_opacity_range: Tuple[float, float] = (1.0, 1.0),
        edge_width_range: Tuple[float, float] = (0.1, 2.0),
        edge_opacity_range: Tuple[float, float] = (1.0, 1.0),
) -> Any:
    import matplotlib.pyplot as plt
    import networkx as nx

    g = nx.DiGraph()
    node_offsets: Dict[str, int] = {}
    current_offset = 0

    # First, collect all unique node types and their counts
    node_types = set()
    node_counts: Dict[str, int] = {}
    remaining_nodes: Dict[str, Set[int]] = {
    }  # Track which nodes are actually present in edges

    # Get all unique nodes that are in the edges
    for edge_type in edge_index_dict.keys():
        src_type, _, dst_type = edge_type
        node_types.add(src_type)
        node_types.add(dst_type)

        if src_type not in remaining_nodes:
            remaining_nodes[src_type] = set()
        if dst_type not in remaining_nodes:
            remaining_nodes[dst_type] = set()

        remaining_nodes[src_type].update(
            edge_index_dict[edge_type][0].tolist())
        remaining_nodes[dst_type].update(
            edge_index_dict[edge_type][1].tolist())

    # Set node counts based on remaining nodes
    for node_type in node_types:
        node_counts[node_type] = len(remaining_nodes[node_type])

    # Add nodes for each node type
    for node_type in node_types:
        num_nodes = node_counts[node_type]
        node_offsets[node_type] = current_offset

        # Get node weights if provided
        weights = None
        if node_weight_dict is not None and node_type in node_weight_dict:
            weights = node_weight_dict[node_type]
            if len(weights) != num_nodes:
                raise ValueError(f"Number of weights for node type "
                                 f"{node_type} ({len(weights)}) does not "
                                 f"match number of nodes ({num_nodes})")

        for i in range(num_nodes):
            node_id = current_offset + i
            label = (node_labels_dict[node_type][i]
                     if node_labels_dict is not None
                     and node_type in node_labels_dict else "")

            # Calculate node size and opacity if weights provided
            size = node_size_range[1]
            opacity = node_opacity_range[1]
            if weights is not None:
                w = weights[i].item()
                size = node_size_range[0] + w * \
                    (node_size_range[1] - node_size_range[0])
                opacity = node_opacity_range[0] + w * \
                    (node_opacity_range[1] - node_opacity_range[0])

            g.add_node(node_id, label=label, type=node_type, size=size,
                       alpha=opacity)

        current_offset += num_nodes

    # Add edges with remapped node indices
    for edge_type, edge_index in edge_index_dict.items():
        src_type, _, dst_type = edge_type
        edge_weight = edge_weight_dict[edge_type]
        src_offset = node_offsets[src_type]
        dst_offset = node_offsets[dst_type]

        # Create mappings for source and target nodes
        src_mapping = {
            old_idx: new_idx
            for new_idx, old_idx in enumerate(sorted(
                remaining_nodes[src_type]))
        }
        dst_mapping = {
            old_idx: new_idx
            for new_idx, old_idx in enumerate(sorted(
                remaining_nodes[dst_type]))
        }

        for (src, dst), w in zip(edge_index.t().tolist(),
                                 edge_weight.tolist()):
            # Remap node indices
            new_src = src_mapping[src] + src_offset
            new_dst = dst_mapping[dst] + dst_offset

            # Calculate edge width and opacity based on weight
            width = edge_width_range[0] + w * \
                (edge_width_range[1] - edge_width_range[0])
            opacity = edge_opacity_range[0] + w * \
                (edge_opacity_range[1] - edge_opacity_range[0])
            g.add_edge(new_src, new_dst, width=width, alpha=opacity)

    # Draw the graph
    ax = plt.gca()
    pos = nx.arf_layout(g)

    # Draw edges with arrows
    for src, dst, data in g.edges(data=True):
        ax.annotate(
            '',
            xy=pos[src],
            xytext=pos[dst],
            arrowprops=dict(
                arrowstyle="<-",
                alpha=data['alpha'],
                linewidth=data['width'],
                shrinkA=sqrt(g.nodes[src]['size']) / 2.0,
                shrinkB=sqrt(g.nodes[dst]['size']) / 2.0,
                connectionstyle="arc3,rad=0.1",
            ),
        )

    # Draw nodes colored by type
    node_colors = []
    node_sizes = []
    node_alphas = []

    # Use matplotlib tab20 colormap for consistent coloring
    tab10_cmap = plt.cm.tab10  # type: ignore[attr-defined]
    node_type_colors: Dict[str, Any] = {}  # Store color for each node type
    for node in g.nodes():
        node_type = g.nodes[node]['type']
        # Assign a consistent color for each node type
        if node_type not in node_type_colors:
            color_idx = len(node_type_colors) % 10  # Cycle through colors
            node_type_colors[node_type] = tab10_cmap(color_idx)
        node_colors.append(node_type_colors[node_type])
        node_sizes.append(g.nodes[node]['size'])
        node_alphas.append(g.nodes[node]['alpha'])

    nx.draw_networkx_nodes(g, pos, node_size=node_sizes,
                           node_color=node_colors, margins=0.1,
                           alpha=node_alphas)

    # Draw labels
    labels = nx.get_node_attributes(g, 'label')
    nx.draw_networkx_labels(g, pos, labels, font_size=10)

    # Add legend
    legend_elements = []
    for node_type, color in node_type_colors.items():
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', label=node_type,
                       markerfacecolor=color, markersize=10))
    ax.legend(handles=legend_elements, loc='upper right',
              bbox_to_anchor=(0.9, 1))

    if path is not None:
        plt.savefig(path, bbox_inches='tight')
    else:
        plt.show()

    plt.close()
