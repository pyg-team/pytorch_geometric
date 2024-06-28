from torch_geometric.datasets import updated_web_qsp_dataset
from torch_geometric.profile import nvtxit
import torch

# Apply Patches
updated_web_qsp_dataset.UpdatedWebQSPDataset._build_graph = nvtxit(n_iters=1)(updated_web_qsp_dataset.UpdatedWebQSPDataset._build_graph)
updated_web_qsp_dataset.UpdatedWebQSPDataset._retrieve_subgraphs = nvtxit(n_iters=1)(updated_web_qsp_dataset.UpdatedWebQSPDataset._retrieve_subgraphs)
updated_web_qsp_dataset.retrieval_via_pcst = nvtxit(n_iters=10)(updated_web_qsp_dataset.retrieval_via_pcst)

updated_web_qsp_dataset.get_features_for_triplets_groups = nvtxit()(updated_web_qsp_dataset.get_features_for_triplets_groups)
updated_web_qsp_dataset.LargeGraphIndexer.get_node_features = nvtxit()(updated_web_qsp_dataset.LargeGraphIndexer.get_node_features)
updated_web_qsp_dataset.LargeGraphIndexer.get_edge_features = nvtxit()(updated_web_qsp_dataset.LargeGraphIndexer.get_edge_features)
updated_web_qsp_dataset.LargeGraphIndexer.get_edge_features_iter = nvtxit(n_warmups=1, n_iters=10)(updated_web_qsp_dataset.LargeGraphIndexer.get_edge_features_iter)
updated_web_qsp_dataset.LargeGraphIndexer.get_node_features_iter = nvtxit(n_warmups=1, n_iters=10)(updated_web_qsp_dataset.LargeGraphIndexer.get_node_features_iter)

if __name__ == "__main__":
    with torch.autograd.profiler.emit_nvtx():
        ds = updated_web_qsp_dataset.UpdatedWebQSPDataset('small_ds', force_reload=True, limit=10)