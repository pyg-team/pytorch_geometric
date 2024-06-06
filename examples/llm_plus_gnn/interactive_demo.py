import torch
from torch_geometric.data import Data
from .g_retriever import inference_step
def convert_kg_desc_2_data_obj(kg_desc):
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
	with torch.no_grad():
		print("Loading and querying GNN+LLM model...")
		gnn_llm_model = torch.load("gnn_llm.pt").eval()
		gnn_llm_answer = inference_step(gnn_llm_model, data_pt, "gnn_llm")["pred"]
		print("Done!")
		print("Loading and querying finetuned LLM model...")
		finetuned_llm_model = torch.load("llm.pt").eval()
		llm_answer = inference_step(llm_model, data_pt, "llm")["pred"]
		print("Done!")