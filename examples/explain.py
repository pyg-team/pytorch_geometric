import matplotlib.pyplot as plt
import torch

from torch_geometric.data import HeteroData
from torch_geometric.explain import HeteroExplanation


def generate_example_explanation(
) -> tuple[HeteroExplanation, dict[str, list[str]]]:
    r"""Generates an example heterogeneous explanation for classifying papers into topics.

    The graph contains papers, authors, institutions and topics, with the following relationships:
    - paper -> written_by -> author
    - author -> affiliated_with -> institution
    - paper -> belongs_to -> topic
    - paper -> cites -> paper

    Returns:
        tuple[HeteroExplanation, dict[str, list[str]]]: A heterogeneous explanation object with meaningful node features
        and relationship patterns, and a dictionary mapping node types to their names.
    """
    data = HeteroData()

    # Create nodes
    # Papers: features represent TF-IDF vectors of content
    data['paper'].x = torch.tensor(
        [
            [0.9, 0.1, 0.1, 0.1],  # Paper 0: ML focused
            [0.1, 0.8, 0.1, 0.1],  # Paper 1: Systems focused
            [0.1, 0.1, 0.9, 0.1],  # Paper 2: Theory focused
        ],
        dtype=torch.float)
    data['paper'].num_nodes = 3

    # Authors: features represent research interests
    data['author'].x = torch.tensor(
        [
            [0.9, 0.1, 0.0],  # Author 0: ML researcher
            [0.1, 0.9, 0.0],  # Author 1: Systems researcher
            [0.0, 0.1, 0.9],  # Author 2: Theory researcher
        ],
        dtype=torch.float)
    data['author'].num_nodes = 3

    # Institutions: features represent research focus
    data['institution'].x = torch.tensor(
        [
            [0.8, 0.1, 0.1],  # Institution 0: Strong in ML
            [0.1, 0.8, 0.1],  # Institution 1: Strong in Systems
        ],
        dtype=torch.float)
    data['institution'].num_nodes = 2

    # Topics: features are one-hot encodings
    data['topic'].x = torch.eye(3)  # ML, Systems, Theory
    data['topic'].num_nodes = 3

    # Create edges
    # Paper-Author relationships
    data['paper', 'written_by', 'author'].edge_index = torch.tensor([
        [0, 1, 2],  # Papers
        [0, 1, 2],  # Authors
    ])

    # Author-Institution relationships
    data['author', 'affiliated_with',
         'institution'].edge_index = torch.tensor([
             [0, 1, 2],  # Authors
             [0, 1, 0],  # Institutions
         ])

    # Paper-Topic relationships
    data['paper', 'belongs_to', 'topic'].edge_index = torch.tensor([
        [0, 1, 2],  # Papers
        [0, 1, 2],  # Topics (ML, Systems, Theory)
    ])

    # Paper citations
    data['paper', 'cites', 'paper'].edge_index = torch.tensor([
        [1, 2],  # Citing papers
        [0, 1],  # Cited papers
    ])

    # Create explanation
    explanation = HeteroExplanation()

    # Copy node features
    explanation['paper'].x = data['paper'].x
    explanation['author'].x = data['author'].x
    explanation['institution'].x = data['institution'].x
    explanation['topic'].x = data['topic'].x

    # Copy edge indices
    explanation['paper', 'written_by',
                'author'].edge_index = data['paper', 'written_by',
                                            'author'].edge_index
    explanation['author', 'affiliated_with',
                'institution'].edge_index = data['author', 'affiliated_with',
                                                 'institution'].edge_index
    explanation['paper', 'belongs_to',
                'topic'].edge_index = data['paper', 'belongs_to',
                                           'topic'].edge_index
    explanation['paper', 'cites',
                'paper'].edge_index = data['paper', 'cites',
                                           'paper'].edge_index

    # Add meaningful node masks
    # Higher values indicate more importance for classification
    # Papers: ML papers are more important for this explanation
    explanation['paper'].node_mask = torch.tensor([
        [0.95],  # Deep Learning Survey - very important
        [0.2],  # Distributed Systems - less important
        [0.1],  # Graph Theory - least important
    ])

    # Authors: ML researchers are more important
    explanation['author'].node_mask = torch.tensor([
        [0.99],  # Dr. Smith (ML researcher) - very important
        [0.15],  # Dr. Johnson (Systems researcher) - less important
        [0.03]  # Dr. Brown (Theory researcher) - least important
    ])

    # Institutions: AI Research Lab is more important
    explanation['institution'].node_mask = torch.tensor([
        [0.9],  # AI Research Lab - very important
        [0.2]  # Systems Engineering Center - less important
    ])

    # Topics: ML topic is most important
    explanation['topic'].node_mask = torch.tensor([
        [0.95],  # Machine Learning - very important
        [0.3],  # Systems - less important
        [0.1]  # Theory - least important
    ])

    # Add meaningful edge masks
    # Higher values indicate more important relationships
    # Paper-Author relationships: ML papers and authors are more important
    explanation['paper', 'written_by', 'author'].edge_mask = torch.tensor([
        0.95,  # Deep Learning Survey -> Dr. Smith (ML paper by ML author)
        0.5,  # Distributed Systems -> Dr. Johnson (Systems paper by Systems author)
        0.3,  # Graph Theory -> Dr. Brown (Theory paper by Theory author)
    ])

    # Author-Institution relationships: ML researchers at AI Lab are more important
    explanation[
        'author', 'affiliated_with', 'institution'].edge_mask = torch.tensor([
            0.95,  # Dr. Smith -> AI Research Lab (ML researcher at ML institution)
            0.33,  # Dr. Johnson -> Systems Engineering Center (Systems researcher at Systems institution)
            0.12  # Dr. Brown -> AI Research Lab (Theory researcher at ML institution)
        ])

    # Paper-Topic relationships: ML papers and topics are more important
    explanation['paper', 'belongs_to', 'topic'].edge_mask = torch.tensor([
        0.95,  # Deep Learning Survey -> Machine Learning
        0.6,  # Distributed Systems -> Systems
        0.4,  # Graph Theory -> Theory
    ])

    # Paper citations: ML paper citations are more important
    explanation['paper', 'cites', 'paper'].edge_mask = torch.tensor([
        0.5,  # Distributed Systems -> Deep Learning Survey
        0.3  # Graph Theory -> Distributed Systems
    ])

    # Create dictionary of node names
    node_names = {
        'paper': [
            'Transformers are \n really just GNNs?',
            'GNNs are \n all you need', 'Graph Theory \n is pretty fun'
        ],
        'author': ['Dr. Smith', 'Dr. Johnson', 'Dr. Brown'],
        'institution': ['AI Lab', 'Systems \nCenter'],
        'topic': ['ML', 'Systems', 'Theory']
    }

    return explanation, node_names


if __name__ == '__main__':
    # For interactive testing
    explanation, node_names = generate_example_explanation()

    # Create a single figure with spring layout
    fig, ax = plt.subplots(figsize=(8, 8))

    # Use the node names from the dictionary
    explanation.visualize_graph(
        node_labels=node_names,
        node_size_range=(100, 750),
        node_opacity_range=(0.1, 1.0),
        edge_width_range=(0.1, 2.0),
        edge_opacity_range=(0.1, 1.0),
    )

    # Add title to the figure
    ax.set_title('Example Visualization with Spring Layout')

    plt.savefig('explanation.png')

    # Show the figure
    plt.show()
