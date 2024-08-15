from torch_geometric.datasets import updated_web_qsp_dataset
from torch_geometric.profile import nvtxit
import torch
import argparse

# Apply Patches
updated_web_qsp_dataset.UpdatedWebQSPDataset.process = nvtxit()(updated_web_qsp_dataset.UpdatedWebQSPDataset.process)
updated_web_qsp_dataset.UpdatedWebQSPDataset._build_graph = nvtxit()(updated_web_qsp_dataset.UpdatedWebQSPDataset._build_graph)
updated_web_qsp_dataset.UpdatedWebQSPDataset._retrieve_subgraphs = nvtxit()(updated_web_qsp_dataset.UpdatedWebQSPDataset._retrieve_subgraphs)
updated_web_qsp_dataset.SentenceTransformer.encode = nvtxit()(updated_web_qsp_dataset.SentenceTransformer.encode)
updated_web_qsp_dataset.retrieval_via_pcst = nvtxit()(updated_web_qsp_dataset.retrieval_via_pcst)

updated_web_qsp_dataset.get_features_for_triplets_groups = nvtxit()(updated_web_qsp_dataset.get_features_for_triplets_groups)
updated_web_qsp_dataset.LargeGraphIndexer.get_node_features = nvtxit()(updated_web_qsp_dataset.LargeGraphIndexer.get_node_features)
updated_web_qsp_dataset.LargeGraphIndexer.get_edge_features = nvtxit()(updated_web_qsp_dataset.LargeGraphIndexer.get_edge_features)
updated_web_qsp_dataset.LargeGraphIndexer.get_edge_features_iter = nvtxit()(updated_web_qsp_dataset.LargeGraphIndexer.get_edge_features_iter)
updated_web_qsp_dataset.LargeGraphIndexer.get_node_features_iter = nvtxit()(updated_web_qsp_dataset.LargeGraphIndexer.get_node_features_iter)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-torch-kernels", "-k", action="store_true")
    args = parser.parse_args()
    if args.capture_torch_kernels:
        with torch.autograd.profiler.emit_nvtx():
            ds = updated_web_qsp_dataset.UpdatedWebQSPDataset('update_ds', force_reload=True)
    else:
        ds = updated_web_qsp_dataset.UpdatedWebQSPDataset('update_ds', force_reload=True)