import gc
from typing import List, Tuple

import torch
from g_retriever import inference_step, load_params_dict

from torch_geometric import seed_everything
from torch_geometric.data import Data
from torch_geometric.nn.models import GRetriever
from torch_geometric.nn.nlp import LLM, SentenceTransformer


def make_data_obj(text_encoder: SentenceTransformer, question: str,
                  nodes: List[Tuple[str, str]],
                  edges: List[Tuple[str, str, str]]) -> Data:
    data = Data()
    num_nodes = 0
    # list of size 1 to simulate batchsize=1
    # true inference setting would batch user queries
    data.question = [question]
    data.num_nodes = len(nodes)
    data.n_id = torch.arange(data.num_nodes).to(torch.int64)

    # Model expects batches sampled from Dataloader
    # hardcoding values for single item batch
    data.batch = torch.zeros(data.num_nodes).to(torch.int64)
    data.ptr = torch.tensor([0, data.num_nodes]).to(torch.int64)

    graph_text_description = "node_id,node_attr" + "\n"
    # collect node attributes
    to_encode = []
    for node_id, node_attr in nodes:
        to_encode.append(node_attr)
        graph_text_description += str(node_id) + "," + str(node_attr) + "\n"

    # collect edge info
    data.num_edges = len(edges)
    graph_text_description += "src,edge_attr,dst" + "\n"
    src_ids, dst_ids, e_attrs = [], [], []
    for src_id, e_attr, dst_id in edges:
        src_ids.append(int(src_id))
        dst_ids.append(int(dst_id))
        e_attrs.append(e_attr)
        graph_text_description += str(src_id) + "," + str(e_attr) + "," + str(
            dst_id) + "\n"
    to_encode += e_attrs

    # encode text
    encoded_text = text_encoder.encode(to_encode)

    # store processed data
    data.x = encoded_text[:data.num_nodes]
    data.edge_attr = encoded_text[data.num_nodes:data.num_nodes +
                                  data.num_edges]
    data.edge_index = torch.tensor([src_ids, dst_ids]).to(torch.int64)
    data.desc = [graph_text_description[:-1]]  # remove last newline

    return data


def user_input_data():
    q_input = input("Please enter your Question:\n")
    question = f"Question: {q_input}\nAnswer: "
    print(
        "\nPlease enter the node attributes with format 'n_id,textual_node_attribute'."
    )  # noqa
    print("Please ensure to order n_ids from 0, 1, 2, ..., num_nodes-1.")
    print("Use [[stop]] to stop inputting.")
    nodes = []
    most_recent_node = ""
    while True:
        most_recent_node = input()
        if most_recent_node == "[[stop]]":
            break
        else:
            nodes.append(tuple(most_recent_node.split(',')))
    print(
        "\nPlease enter the edge attributes with format 'src_id,textual_edge_attribute,dst_id'"
    )  # noqa
    print("Use [[stop]] to stop inputting.")
    edges = []
    most_recent_edge = ""
    while True:
        most_recent_edge = input()
        if most_recent_edge == "[[stop]]":
            break
        else:
            edges.append(tuple(most_recent_edge.split(',')))
    print("Creating data object...")
    text_encoder = SentenceTransformer()
    data_obj = make_data_obj(text_encoder, question, nodes, edges)
    print("Done!")
    print("data =", data_obj)
    return data_obj


if __name__ == "__main__":
    continue_input = True
    while continue_input:
        seed_everything(42)
        data_obj = user_input_data()
        with torch.no_grad():
            print("Loading GNN+LLM model...")
            gnn_llm_model = load_params_dict(GRetriever(), "gnn_llm.pt").eval()
            print("Querying GNN+LLM model...")
            gnn_llm_answer = inference_step(gnn_llm_model, data_obj,
                                            "gnn_llm")["pred"][0].split("|")[0]
            print("Answer:", gnn_llm_answer)
            del gnn_llm_model
            gc.collect()
            torch.cuda.empty_cache()
            print("Done!")
            print("Loading finetuned LLM model for comparison...")
            finetuned_llm_model = load_params_dict(LLM(), "llm.pt").eval()
            print("Querying LLM...")
            llm_answer = inference_step(finetuned_llm_model, data_obj,
                                        "llm")["pred"][0].split("|")[0]
            print("Answer:", llm_answer)
            del finetuned_llm_model
            gc.collect()
            torch.cuda.empty_cache()
            print("Done!")
        continue_input = input(
            "Would you like to try another? y/n:").lower() == "y"
