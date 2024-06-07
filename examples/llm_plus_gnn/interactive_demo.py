import torch
from torch_geometric.data import Data
from g_retriever import inference_step
from typing import List, Tuple
from torch_geometric.nn.nlp import SentenceTransformer

def make_data_obj(text_encoder: SentenceTransformer, question: str, nodes: List[Tuple[str, str]] , edges:List[Tuple[str, str, str]]) -> Data:
	data_obj = Data()
	num_nodes = 0
	data_obj.question = question
	data_obj.num_nodes = len(nodes)
	data_obj.n_id = torch.arange(data_obj.num_nodes)

	# Model expects batches sampled from Dataloader
	# hardcoding values for single item batch
	data_obj.batch = torch.zeros(data_obj.num_nodes)
	data_obj.ptr = torch.tensor([0, data_obj.num_nodes])

	# encode node attributes
	to_encode = [node[1] for node in nodes]
	data.num_edges = len(edges)
	edge_index = torch.zeros((2, data.num_edges))
	e_attrs = []
	for src_id, e_attr, dst_id in edges:
		edge_index[:, len(e_attrs)] = (src_id, dst_id)
		e_attrs.append(e_attr)
	to_encode += e_attrs
	encoded_text = text_encoder.encode(to_encode)
	data_obj.x = encoded_text[:data_obj.num_nodes]
	data_obj.e_attr = encoded_text[data_obj.num_nodes:data_obj.num_nodes+data.num_edges]

	return data_obj

if __name__ == "__main__":
	question = input("Please enter your Question:")
	print("Please enter the node attributes with format \
		'n_id,textual_node_attribute'")
	print("Use [[stop]] to stop inputting.")
	nodes = []
	most_recent_node = ""
	while True:
		most_recent_node = input()
		if most_recent_node == "[[stop]]":
			break
		else:
			nodes.append(tuple(most_recent_node.split(',')))
	print("Please enter the edge attributes with format \
		'src_id,textual_edge_attribute,dst_id'")
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
	data_obj = make_data_obj(text_encoder, question, nodes[:-1], edges[:-1])
	print("Done!")
	with torch.no_grad():
		print("Loading and querying GNN+LLM model...")
		gnn_llm_model = torch.load("gnn_llm.pt").eval()
		gnn_llm_answer = inference_step(gnn_llm_model, data_obj, "gnn_llm")["pred"]
		del gnn_llm_model
		gc.collect()
		torch.cuda.empty_cache()
		print("Done!")
		print("Loading and querying finetuned LLM model...")
		finetuned_llm_model = torch.load("llm.pt").eval()
		llm_answer = inference_step(llm_model, data_obj, "llm")["pred"]
		print("Done!")