import argparse
import math
import os
import os.path as osp
import sys
import timeit
from pathlib import Path

import pytest
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn.models import EdgeBankPredictor


# ==================
# ==================
# ==================
def test_edge_bank_pred():
    from tgb.linkproppred.dataset import LinkPropPredDataset
    from tgb.linkproppred.evaluate import Evaluator
    from tgb.utils.utils import save_results, set_random_seed

    def helper_func(data, test_mask, neg_sampler, split_mode):
        r"""
        Evaluated the dynamic link prediction
        Evaluation happens as 'one vs. many', meaning that each positive edge is evaluated against many negative edges

        Parameters:
            data: a dataset object
            test_mask: required masks to load the test set edges
            neg_sampler: an object that gives the negative edges corresponding to each positive edge
            split_mode: specifies whether it is the 'validation' or 'test' set to correctly load the negatives
        Returns:
            perf_metric: the result of the performance evaluaiton
        """
        num_batches = math.ceil(len(data['sources'][test_mask]) / BATCH_SIZE)
        perf_list = []
        for batch_idx in tqdm(range(num_batches)):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE,
                          len(data['sources'][test_mask]))
            pos_src, pos_dst, pos_t = (
                torch.tensor(
                    data['sources'][test_mask][start_idx:end_idx]).to(device),
                torch.tensor(data['destinations'][test_mask]
                             [start_idx:end_idx]).to(device),
                torch.tensor(data['timestamps'][test_mask]
                             [start_idx:end_idx]).to(device),
            )
            neg_batch_list = neg_sampler.query_batch(pos_src, pos_dst, pos_t,
                                                     split_mode=split_mode)

            for idx, neg_batch in enumerate(neg_batch_list):
                query_src = torch.tensor([
                    int(pos_src[idx]) for _ in range(len(neg_batch) + 1)
                ]).to(device).reshape(1, -1)
                query_dst = torch.cat([
                    torch.tensor([int(pos_dst[idx])]),
                    torch.tensor(neg_batch),
                ]).to(device).reshape(1, -1)
                query_edge_index = torch.cat((query_src, query_dst))
                y_pred = edgebank.predict_link(query_edge_index)
                # compute MRR
                input_dict = {
                    "y_pred_pos": y_pred[0],
                    "y_pred_neg": y_pred[1:],
                    "eval_metric": [metric],
                }
                perf_list.append(evaluator.eval(input_dict)[metric])

            # update edgebank memory after each positive batch
            pos_edge_index = torch.cat(
                (pos_src.reshape(1, -1), pos_dst.reshape(1, -1)))
            edgebank.update_memory(pos_edge_index, pos_t)

        perf_metrics = float(torch.mean(torch.tensor(perf_list)))

        return perf_metrics

    # set hyperparameters
    DATA = 'tgbl-coin'
    BATCH_SIZE = 200
    K_VALUE = 10
    SEED = 1
    set_random_seed(SEED)
    MEMORY_MODE = "unlimited"
    TIME_WINDOW_RATIO = .15

    # ==================
    # ==================
    # ==================

    start_overall = timeit.default_timer()

    MODEL_NAME = 'EdgeBank'

    # data loading with `numpy`
    dataset = LinkPropPredDataset(name=DATA, root="datasets", preprocess=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("dataset=", dataset)
    data = dataset.full_data
    metric = dataset.eval_metric

    # get masks
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask

    #data for memory in edgebank
    hist_src = torch.tensor(data['sources'][train_mask]).to(device)
    hist_dst = torch.tensor(data['destinations'][train_mask]).to(device)
    hist_edge_index = torch.cat(
        (hist_src.reshape(1, -1), hist_dst.reshape(1, -1)))
    hist_ts = torch.tensor(data['timestamps'][train_mask]).to(device)
    print("dataset.full_data=", dataset.full_data)
    print("dataset.eval_metric=", dataset.eval_metric)
    print("dataset.train_mask=", dataset.train_mask)

    # Set EdgeBank with memory updater
    edgebank = EdgeBankPredictor(memory_mode=MEMORY_MODE,
                                 time_window_ratio=TIME_WINDOW_RATIO)
    edgebank.update_memory(hist_edge_index, hist_ts)

    print("==========================================================")
    print(
        f"============*** {MODEL_NAME}: {MEMORY_MODE}: {DATA} ***=============="
    )
    print("==========================================================")

    evaluator = Evaluator(name=DATA)
    neg_sampler = dataset.negative_sampler
    print("dataset.negative_sampler=", dataset.negative_sampler)

    # ==================================================== Test
    # loading the validation negative samples
    dataset.load_val_ns()

    # testing ...
    start_val = timeit.default_timer()
    perf_metric_test = helper_func(data, val_mask, neg_sampler,
                                   split_mode='val')
    end_val = timeit.default_timer()

    print(f"INFO: val: Evaluation Setting: >>> ONE-VS-MANY <<< ")
    print(f"\tval: {metric}: {perf_metric_test: .4f}")
    test_time = timeit.default_timer() - start_val
    print(f"\tval: Elapsed Time (s): {test_time: .4f}")

    # ==================================================== Test
    # loading the test negative samples
    dataset.load_test_ns()

    # testing ...
    start_test = timeit.default_timer()
    perf_metric_test = helper_func(data, test_mask, neg_sampler,
                                   split_mode='test')
    end_test = timeit.default_timer()

    print(f"INFO: Test: Evaluation Setting: >>> ONE-VS-MANY <<< ")
    print(f"\tTest: {metric}: {perf_metric_test: .4f}")
    test_time = timeit.default_timer() - start_test
    print(f"\tTest: Elapsed Time (s): {test_time: .4f}")
