import torch
from torch_geometric.data import Data
from .g_retriever import inference_step
def convert_kg_desc_2_data_obj(nodes, edges):
	data_obj = Data()
	# fill it out
	return data_obj

def inference_step(model, batch, model_save_name):
    if model_save_name == "llm":
        return model.inference(batch.question, batch.desc)
    else:
        return model.inference(batch.question, batch.x, batch.edge_index,
                               batch.batch, batch.ptr, batch.edge_attr,
                               batch.desc)

if __name__ == "__main__":
	question = input("Please enter your Question:")
	print("Please enter the node attributes with format \
		'n_id,textual_node_attribute'")
	print("Use [[stop]] to stop inputting.")
	nodes = []
	most_recent_node = ""
	while most_recent_node != "[[stop]]":
		most_recent_node = input()
		nodes.append(most_recent_node)
	edges = []
	most_recent_edge = ""
	print("Please enter the edge attributes with format \
		'src_id,textual_edge_attribute,dst_id'")
	print("Use [[stop]] to stop inputting.")
	while most_recent_edge != "[[stop]]":
		most_recent_edge = input()
		edges.append(most_recent_edge)
	data_obj = convert_kg_desc_2_data_obj(nodes, edges)

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