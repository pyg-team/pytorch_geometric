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

def cleanup():
	

if __name__ == "__main__":
	question = input("Please enter your Question:")
	nodes = # get node ids and attributes
	edges = # get edge ids and attributes
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