from torch_geometric.datasets import web_qsp_dataset
from torch_geometric.profile import nvtxit
import torch

# Apply Patches
web_qsp_dataset.retrieval_via_pcst = nvtxit(n_warmups=1, n_iters=10)(web_qsp_dataset.retrieval_via_pcst)


if __name__ == "__main__":
    with torch.autograd.profiler.emit_nvtx():
        ds = web_qsp_dataset.WebQSPDataset('baseline')