import torch
from torch_geometric.data import Data
from g_retriever import inference_step, load_params_dict
from typing import List, Tuple
from torch_geometric.nn.nlp import SentenceTransformer, LLM
from torch_geometric.nn.models import GRetriever

def make_data_obj(text_encoder: SentenceTransformer, question: str, nodes: List[Tuple[str, str]] , edges:List[Tuple[str, str, str]]) -> Data:
	data = Data()
	num_nodes = 0
	data.question = question
	data.num_nodes = len(nodes)
	data.n_id = torch.arange(data.num_nodes).to(torch.int64)

	# Model expects batches sampled from Dataloader
	# hardcoding values for single item batch
	data.batch = torch.zeros(data.num_nodes).to(torch.int64)
	data.ptr = torch.tensor([0, data.num_nodes]).to(torch.int64)

	graph_text_description = "node_id,node_attr"
	# collect node attributes
	for node_id, node_attr in nodes:
		to_encode.append(node_attr)
		graph_text_description += str(node_id) + "," + str(node_attr) + "\n"
	to_encode = [node[1] for node in nodes]

	# collect edge info
	data.num_edges = len(edges)
	graph_text_description += "src,edge_attr,dst"
	src_ids, dst_ids, e_attrs = [], [], []
	for src_id, e_attr, dst_id in edges:
		src_ids.append(int(src_id))
		dst_ids.append(int(dst_id))
		e_attrs.append(e_attr)
		graph_text_description += str(src_id) + "," + str(e_attr) + "," + str(dst_id) +"\n"
	to_encode += e_attrs

	# encode text
	encoded_text = text_encoder.encode(to_encode)

	# store processed data
	data.x = encoded_text[:data.num_nodes]
	data.edge_attr = encoded_text[data.num_nodes:data.num_nodes+data.num_edges]
	data.edge_index = torch.tensor([src_ids, dst_ids]).to(torch.int64)
	data.desc = graph_text_description

	return data

def user_input_data():
	question = input("Please enter your Question:\n")
	print("\nPlease enter the node attributes with format 'n_id,textual_node_attribute'.") # noqa
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
	print("\nPlease enter the edge attributes with format 'src_id,textual_edge_attribute,dst_id'") # noqa
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
	print("data.edge_index=", data_obj.edge_index)
	return data_obj


if __name__ == "__main__":
	print("Loading GNN+LLM model...")
	gnn_llm_model = load_params_dict(GRetriever(), "gnn_llm.pt").eval()
	data_obj = user_input_data()	
	with torch.no_grad():
		print("Querying GNN+LLM model...")
		gnn_llm_answer = inference_step(gnn_llm_model, data_obj, "gnn_llm")["pred"]
		print("Answer=", gnn_llm_answer)
		del gnn_llm_model
		gc.collect()
		torch.cuda.empty_cache()
		print("Done!")
		print("Loading finetuned LLM model for comparison...")
		finetuned_llm_model = load_params_dict(LLM(), "llm.pt").eval()
		print("Querying LLM...")
		llm_answer = inference_step(llm_model, data_obj, "llm")["pred"]
		print("Answer=", llm_answer)
		print("Done!")