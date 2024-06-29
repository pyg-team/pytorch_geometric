from torch_geometric.datasets import web_qsp_dataset
from torch_geometric.profile import nvtxit
import torch

# Apply Patches
web_qsp_dataset.retrieval_via_pcst = nvtxit(n_warmups=1, n_iters=10)(web_qsp_dataset.retrieval_via_pcst)
web_qsp_dataset.WebQSPDataset.process = nvtxit()(web_qsp_dataset.WebQSPDataset.process)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-torch-kernels", "-k", action="store_true")
    args = parser.parse_args()
    if args.capture_torch_kernels:
        with torch.autograd.profiler.emit_nvtx():
            ds = web_qsp_dataset.WebQSPDataset('baseline')
    else:
        ds = web_qsp_dataset.WebQSPDataset('baseline')