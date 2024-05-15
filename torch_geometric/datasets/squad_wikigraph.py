import random
from typing import List, Optional

import torch

from torch_geometric.data import Data, InMemoryDataset
import tqdm
try:
    import datasets
    WITH_DATASETS = True
except ImportError as e:  # noqa
    WITH_DATASETS = False
from torch_geometric.nn.text import SentenceTransformer, text2embedding
from torch_geometric.nn.text.llm import llama2_str_name


def get_wiki_data(question: str, model: SentenceTransformer,
                  seed_nodes: int = 3, fan_out: int = 3,
                  num_hops: int = 2, label: Optional[str] = None) -> Data:
    """Performs neighborsampling on wikipedia
    """
    import wikipedia
    search_list = wikipedia.search(question)
    seed_doc_names = search_list[:seed_nodes]
    seed_docs = [wikipedia.page(doc_name) for doc_name in seed_doc_names]
    # initialize our doc graph with seed docs
    doc_contents = [doc.content for doc in seed_docs]
    title_2_node_id_map = {doc.title: i for doc in enumerate(seed_docs)}
    # do neighborsampling and create graph
    src_n_ids = []
    dst_n_ids = []
    for hop in num_hops:
        next_hops_seed_docs = []
        for src_doc in seed_docs:
            full_fan_links = src_doc.links
            randomly_chosen_neighbor_links = list(
                random.sample(full_fan_links, k=fan_out))
            new_seed_docs = [
                wikipedia.page(link) for link in randomly_chosen_neighbor_links
            ]
            for dst_doc in new_seed_docs:
                dst_doc_title = dst_doc.title
                if dst_doc_title not in title_2_node_id_map:
                    # add new node to graph
                    title_2_node_id_map[dst_doc_name] = len(
                        title_2_node_id_map)
                    doc_contents.append(dst_doc.content)
                next_hops_seed_docs.append(doc)
                src_n_ids.append(title_2_node_id_map[src_doc.title])
                dst_n_ids.append(title_2_node_id_map[dst_doc.title])

        # root nodes for the next hop
        seed_docs = next_hops_seed_docs

    # put docs into model
    embedded_docs = text2embedding(model, doc_contents, batch_size=8)
    del doc_contents

    # create node features, x
    x = torch.cat(embedded_docs)

    # construct and return Data object
    return Data(x=x, edge_index=torch.tensor([src_n_ids, dst_n_ids]),
                n_id=torch.tensor(title_2_node_id_map.values()),
                question=question, label=label).to("cpu")


# create SQUAD_WikiGraph dataset by calling wikiloader for each SQUAD question
class SQUAD_WikiGraph(InMemoryDataset):
    r"""This dataset uses the SQUAD dataset for Questions and Answers.
    It uses Wikipedia for the corresponding knowledge graphs.
    SQUAD source: https://huggingface.co/datasets/rajpurkar/squad

    Args:
        root (str): Root directory where the dataset should be saved.
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """
    def __init__(
        self,
        root: str = "",
        force_reload: bool = False,
    ) -> None:
        if not WITH_DATASETS:
            raise ImportError("Please pip install datasets")
        super().__init__(root, None, None, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return [
            "list_of_wiki_graphs.pt", "pre_filter_4_wiki.pt",
            "pre_transform_4_wiki.pt"
        ]

    def download(self) -> None:
        dataset = datasets.load_dataset("rajpurkar/squad_v2")
        # Squad Test Set is Hidden
        self.raw_dataset = datasets.concatenate_datasets(
            [dataset["train"], dataset["validation"]])
        # split official eval set into eval and test
        # note that the eval set has odd number of instances
        self.split_idxs = {
            "train":
            torch.arange(len(dataset["train"])),
            "val":
            torch.arange(len(dataset["validation"]) // 2) +
            len(dataset["train"]),
            "test":
            torch.arange(len(dataset["validation"]) // 2) +
            len(dataset["train"]) + len(dataset["validation"]) // 2 +
            (len(dataset["validation"]) % 2)
        }

    def process(self) -> None:
        self.model = SentenceTransformer(llama2_str_name, autocast_dtype=torch.bfloat16)
        self.model.eval()
        self.questions = [i["question"] for i in self.raw_dataset]
        list_of_data_objs = []
        for index in tqdm(range(len(self.raw_dataset))):
            data_i = self.raw_dataset[index]
            question = f"Question: {data_i['question']}\nAnswer: "
            label = data_i["answer"].lower()
            pyg_data_obj = get_wiki_data(question, self.model,
                                         label=label)
            list_of_data_objs.append(pyg_data_obj)

        self.save(list_of_data_objs, self.processed_paths[0])
