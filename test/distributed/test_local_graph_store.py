import torch
import logging
from torch_geometric.distributed import LocalGraphStore
from torch_geometric.testing import get_random_edge_index


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("test.log"),
            logging.StreamHandler()
        ]
    )


def test_local_graph_store():
    setup_logging()
    logger = logging.getLogger(__name__)

    graph_store = LocalGraphStore()

    edge_index = get_random_edge_index(100, 100, 300)
    edge_id = torch.tensor([1, 2, 3, 5, 8, 4])

    graph_store.put_edge_index(edge_index, edge_type=None, layout='coo',
                               size=(100, 100))

    graph_store.put_edge_id(edge_id, edge_type=None, layout='coo',
                            size=(100, 100))

    assert len(graph_store.get_all_edge_attrs()) == 1
    logger.info("Assertion 1: Length of all edge attributes is as expected")

    edge_attr = graph_store.get_all_edge_attrs()[0]
    assert torch.equal(graph_store.get_edge_index(edge_attr), edge_index)
    logger.info("Assertion 2: Edge index retrieved correctly")

    assert torch.equal(graph_store.get_edge_id(edge_attr), edge_id)
    logger.info("Assertion 3: Edge IDs retrieved correctly")

    graph_store.remove_edge_index(edge_attr)
    graph_store.remove_edge_id(edge_attr)
    assert len(graph_store.get_all_edge_attrs()) == 0
    logger.info("Assertion 4: All edge attributes are removed")


def test_homogeneous_graph_store():
    setup_logging()
    logger = logging.getLogger(__name__)

    edge_id = torch.randperm(300)
    edge_index = get_random_edge_index(100, 100, 300)

    graph_store = LocalGraphStore.from_data(edge_id, edge_index, num_nodes=100)

    assert len(graph_store.get_all_edge_attrs()) == 1
    logger.info("Assertion 1: Length of all edge attributes is as expected")

    edge_attr = graph_store.get_all_edge_attrs()[0]
    assert edge_attr.edge_type is None
    assert edge_attr.layout.value == 'coo'
    assert not edge_attr.is_sorted
    assert edge_attr.size == (100, 100)
    logger.info("Assertion 2: Edge attribute properties are correct")

    assert torch.equal(
        graph_store.get_edge_id(edge_type=None, layout='coo'),
        edge_id,
    )
    assert torch.equal(
        graph_store.get_edge_index(edge_type=None, layout='coo'),
        edge_index,
    )
    logger.info("Assertion 3: Edge IDs and edge index retrieved correctly")


def test_heterogeneous_graph_store():
    setup_logging()
    logger = logging.getLogger(__name__)

    edge_type = ('paper', 'to', 'paper')
    edge_id_dict = {edge_type: torch.randperm(300)}
    edge_index_dict = {edge_type: get_random_edge_index(100, 100, 300)}

    graph_store = LocalGraphStore.from_hetero_data(
        edge_id_dict, edge_index_dict, num_nodes_dict={'paper': 100})

    assert len(graph_store.get_all_edge_attrs()) == 1
    logger.info("Assertion 1: Length of all edge attributes is as expected")

    edge_attr = graph_store.get_all_edge_attrs()[0]
    assert edge_attr.edge_type == edge_type
    assert edge_attr.layout.value == 'coo'
    assert not edge_attr.is_sorted
    assert edge_attr.size == (100, 100)
    logger.info("Assertion 2: Edge attribute properties are correct")

    assert torch.equal(
        graph_store.get_edge_id(edge_type, layout='coo'),
        edge_id_dict[edge_type],
    )
    assert torch.equal(
        graph_store.get_edge_index(edge_type, layout='coo'),
        edge_index_dict[edge_type],
    )
    logger.info("Assertion 3: Edge IDs and edge index retrieved correctly")
