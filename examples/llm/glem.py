"""This example run GLEM model using PyG.
Original Paper: https://arxiv.org/abs/2210.14709
“Learning on Large-scale Text-attributed Graphs via Variational Inference“.
Requirements on top of basic PyG:
`pip install ogb transformers peft tqdm`.
GLEM is a data augmentation co-training strategy for LM and GNN, our
implementation extended original implementation from LM to LLM and opt for LoRA
from peft.

``note::
    use additional trick, please add your external prediction by assigning
    `ext_pred_path` and combine it into pretraining phase and node features
"""

import argparse
import os
import os.path as osp
import time

import psutil
import torch
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset

from torch_geometric import seed_everything
from torch_geometric.data import download_google_url
from torch_geometric.datasets import TAGDataset
from torch_geometric.llm import GLEM
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.nn.models import GAT, GCN, GraphSAGE


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def main(args):
    gpu = args.gpu
    dataset_name = args.dataset
    text_type = args.text_type if args.dataset == 'arxiv' else 'raw_text'
    root = osp.join('data', 'ogb')
    hf_model = args.hf_model
    pl_ratio = args.pl_ratio
    gnn_lr = args.gnn_lr
    lm_lr = args.lm_lr
    em_order = args.em_order
    gnn_epochs = args.gnn_epochs
    lm_epochs = args.lm_epochs
    patience = args.patience
    verbose = args.verbose
    out_dir = args.out_dir
    lm_batch_size = args.lm_batch_size
    gnn_batch_size = args.gnn_batch_size
    lm_use_lora = args.lm_use_lora
    token_on_disk = args.token_on_disk
    num_em_iters = args.num_em_iters
    start_time = time.time()
    train_without_ext_pred = args.train_without_ext_pred
    ext_pred = None
    pretrain_augmented = False
    ext_pseudo_labels = None
    device = torch.device(
        f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Running on: {torch.cuda.get_device_name({gpu})}')
    torch.cuda.empty_cache()

    if not train_without_ext_pred:
        ext_pred_path = download_google_url(
            id='15sO2m7BeW7C1Upmdw3Cx1JS__6nxTAzY',
            folder='data/ogb/ogbn_products/ext_preds',
            filename='giant_sagn_scr.pt', log=True)
        ext_pred = torch.load(ext_pred_path, map_location=device)
        ext_pseudo_labels = ext_pred.argmax(dim=-1)
        pretrain_augmented = True

    seed_everything(42)

    dataset = PygNodePropPredDataset(f'ogbn-{dataset_name}', root=root)
    split_idx = dataset.get_idx_split()
    data = dataset._data

    tag_dataset = TAGDataset(root, dataset, hf_model,
                             token_on_disk=token_on_disk)
    text_dataset = tag_dataset.to_text_dataset(text_type)
    print(tag_dataset.num_classes, tag_dataset.raw_file_names)

    num_classes = tag_dataset.num_classes
    num_features = data.num_features
    # =========================== LM Data split ===============================
    split_idx = tag_dataset.get_idx_split()

    # GLEM train with augmented data, mark original train data as gold data,
    gold_idx = split_idx['train']
    split_idx['valid']
    test_idx = split_idx['test']

    # random sample pseudo labels nodes, generate their index
    num_pseudo_labels = int(gold_idx.numel() * pl_ratio)
    idx_to_select = torch.randperm(test_idx.numel())[:num_pseudo_labels]
    pseudo_labels_idx = test_idx[idx_to_select]
    train_idx = torch.cat(
        (gold_idx, pseudo_labels_idx))  # augmented train_indx

    print(f'train_idx: {train_idx.size(0)}, '
          f'gold_idx: {gold_idx.size(0)}, '
          f'pseudo labels ratio: {pl_ratio}, '
          f'{train_idx.size(0)/gold_idx.size(0) - 1.0}')
    gold_dataset = torch.utils.data.Subset(dataset=text_dataset,
                                           indices=gold_idx)
    train_dataset = torch.utils.data.Subset(dataset=text_dataset,
                                            indices=train_idx)
    # ========================== LM Data Loader ===============================

    print('Building language model dataloader...', end='-->')

    # if set train_without_ext_pred == True, use this for pretrain
    text_pretrain_loader = DataLoader(gold_dataset, batch_size=lm_batch_size,
                                      drop_last=False, pin_memory=True,
                                      shuffle=True)
    # training with augmented data,
    text_train_loader = DataLoader(train_dataset, batch_size=lm_batch_size,
                                   drop_last=False, pin_memory=True,
                                   shuffle=True)
    text_test_loader = DataLoader(text_dataset, batch_size=lm_batch_size * 4,
                                  drop_last=False, pin_memory=True,
                                  shuffle=False)
    print('done')

    # =========================== GNN Data Loader =============================
    initial_memory = torch.cuda.memory_allocated()
    data = data.to(device)
    if ext_pred is not None:
        data.x = torch.cat((data.x, ext_pred), dim=1)
        num_features += ext_pred.size(1)
    current_memory_1 = torch.cuda.max_memory_allocated()
    # 1 GB = 1073741824 Byte
    gpu_usage = float(current_memory_1 - initial_memory) / 1073741824
    # Print the maximum memory usage after running the model
    print(f'GPU memory usage -- data to gpu: {gpu_usage:.2f} GB')

    print('build GNN dataloader(GraphSAGE NeighborLoader)', end='-->')

    # train on gold data w/o pseudo labels
    graph_pretrain_loader = NeighborLoader(
        data,
        input_nodes=gold_idx,
        num_neighbors=[15, 10, 5],
        batch_size=gnn_batch_size,
        shuffle=True,
        num_workers=12,
        persistent_workers=True,
    )

    # graph data loader w/ pseudo labels in M-step
    graph_train_loader = NeighborLoader(
        data,
        input_nodes=train_idx,
        num_neighbors=[15, 10, 5],
        batch_size=gnn_batch_size,
        shuffle=True,
        num_workers=12,
        persistent_workers=True,
    )

    # for gnn inference
    subgraph_loader = NeighborLoader(
        data,
        input_nodes=None,
        num_neighbors=[-1],
        batch_size=gnn_batch_size * 4,
        num_workers=12,
        persistent_workers=True,
    )
    # =========================== internal function ===========================

    evaluator = Evaluator(name=f'ogbn-{dataset_name}')

    def evaluate(out, split):
        y_true = data.y.cpu()
        y_pred = out.argmax(dim=-1, keepdim=True)
        train_acc, val_acc, test_acc = None, None, None
        if 'train' in split:
            train_acc = evaluator.eval({
                'y_true': y_true[split_idx['train']],
                'y_pred': y_pred[split_idx['train']],
            })['acc']
        if 'valid' in split:
            val_acc = evaluator.eval({
                'y_true': y_true[split_idx['valid']],
                'y_pred': y_pred[split_idx['valid']],
            })['acc']
        if 'test' in split:
            test_acc = evaluator.eval({
                'y_true': y_true[split_idx['test']],
                'y_pred': y_pred[split_idx['test']],
            })['acc']

        return train_acc, val_acc, test_acc

    # =========================== Build GNN Model =============================
    gnn = None
    if args.gnn_model == 'SAGE':
        gnn = GraphSAGE(
            in_channels=num_features,
            hidden_channels=args.gnn_hidden_channels,
            num_layers=args.gnn_num_layers,
            out_channels=dataset.num_classes,
        )
    elif args.gnn_model == 'GAT':
        gnn = GAT(in_channels=num_features,
                  hidden_channels=args.gnn_hidden_channels,
                  num_layers=args.gnn_num_layers,
                  out_channels=dataset.num_classes, heads=args.gat_heads)
    else:
        gnn = GCN(
            in_channels=num_features,
            hidden_channels=args.gnn_hidden_channels,
            num_layers=args.gnn_num_layers,
            out_channels=dataset.num_classes,
        )

    print("# GNN Params:", get_n_params(gnn))
    # =========================== Build LM Model ==============================

    model = GLEM(lm_to_use=hf_model, gnn_to_use=gnn, out_channels=num_classes,
                 lm_use_lora=lm_use_lora, device=device)
    lm = model.lm
    print("# LM Params:", get_n_params(lm))
    gnn_opt = torch.optim.Adam(gnn.parameters(), lr=gnn_lr)
    lm_opt = torch.optim.Adam(lm.parameters(), lr=lm_lr)

    def load_model(em_phase):
        print(f'Move {em_phase} model from cpu memory')
        if em_phase == 'lm':
            model.lm = model.lm.to(device, non_blocking=True)
            optimizer = torch.optim.Adam(model.lm.parameters(), lr=lm_lr)
        if em_phase == 'gnn':
            model.gnn = model.gnn.to(device, non_blocking=True)
            optimizer = torch.optim.Adam(model.gnn.parameters(), lr=gnn_lr)
        return optimizer

    # ================================= Run GLEM ==============================
    preds_filename = 'lm_pretrain'
    preds_dir = f'{out_dir}preds/{dataset_name}/'
    gnn_test_acc = 0.0
    lm_test_acc = 0.0
    # =============================== GLEM pretraining ========================
    pretrain_phase = 'lm'
    if em_order == 'lm':
        pretrain_phase = 'gnn'
    pretrain_start_time = time.time()
    # pretraining
    pretrain_loader = graph_pretrain_loader
    test_loader = subgraph_loader
    pretrain_num_epochs = gnn_epochs
    pretrain_opt = gnn_opt
    if pretrain_phase == 'gnn':
        model.gnn = model.gnn.to(device)
        print('pretraining gnn to generate pseudo labels')
        if not train_without_ext_pred:
            pretrain_loader = graph_train_loader
        preds_filename = 'gnn_pretrain'
    elif pretrain_phase == 'lm':
        model.lm = model.lm.to(device)
        print('pretraining lm to generate pseudo labels')
        pretrain_num_epochs = lm_epochs
        pretrain_loader = text_pretrain_loader
        test_loader = text_test_loader
        pretrain_opt = lm_opt
        if not train_without_ext_pred:
            pretrain_loader = text_train_loader
        preds_filename = 'lm_pretrain'

    early_stopping = 0
    best_val_acc = 0.0
    for epoch in range(1, pretrain_num_epochs + 1):
        acc, loss = model.train(pretrain_phase, pretrain_loader, pretrain_opt,
                                ext_pseudo_labels, epoch, pretrain_augmented,
                                verbose)
        if epoch >= 5 or epoch == pretrain_num_epochs:
            pretrain_preds = model.inference(pretrain_phase, test_loader,
                                             verbose=verbose)
            train_acc, val_acc, _ = evaluate(pretrain_preds,
                                             ['train', 'valid'])

            print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}')

            if val_acc <= best_val_acc:
                early_stopping += 1
                if early_stopping > patience:
                    print(f'Pretrain Early stopped by Epoch: {epoch}')
                    break
            else:
                best_val_acc = val_acc
    preds = model.inference(pretrain_phase, test_loader, verbose=verbose)
    train_acc, val_acc, test_acc = evaluate(preds, ['train', 'valid', 'test'])
    if pretrain_phase == 'gnn':
        gnn_test_acc = max(gnn_test_acc, test_acc)
        model.gnn = model.gnn.to('cpu', non_blocking=True)
    else:
        lm_test_acc = max(lm_test_acc, test_acc)
        model.lm = model.lm.to('cpu', non_blocking=True)
    torch.cuda.empty_cache()

    pretrain_phase_time = time.time() - pretrain_start_time
    print(f'Pretrain {pretrain_phase} time: {pretrain_phase_time:.2f}s')
    os.makedirs(osp.dirname(preds_dir), exist_ok=True)
    torch.save(preds, osp.join(preds_dir, f'{preds_filename}.pt'))
    print(
        f'Saved predictions to {osp.join(preds_dir, f"{preds_filename}.pt")}')
    train_acc, val_acc, test_acc = evaluate(preds, ['train', 'valid', 'test'])
    print(f'Pretraining acc: {train_acc:.4f}, Val: {val_acc:.4f}, '
          f'Test: {test_acc:.4f}')

    # EM iterations

    em_phase = em_order
    """
    We run E-step(LM training) and M-Step(GNN training) alternatively in each
    em iterations, so the total number of iterations is num_em_iter * 2 and
    we switch the em_phase at end of each iteration in following loop
    """
    gnn_val_acc = lm_val_acc = 0.0
    for em_it in range(1, num_em_iters * 2 + 1):
        pseudo_labels = preds.argmax(dim=-1)
        best_val_acc = 0.0
        print(f'EM iteration: {em_it}, EM phase: {em_phase}')
        optimizer = load_model(em_phase)
        num_epochs = lm_epochs
        train_loader = text_train_loader
        test_loader = text_test_loader
        early_stopping = 0
        if em_phase == 'gnn':
            train_loader = graph_train_loader
            num_epochs = gnn_epochs
            test_loader = subgraph_loader
        for epoch in range(1, num_epochs + 1):
            acc, loss = model.train(em_phase, train_loader, optimizer,
                                    pseudo_labels, epoch, True, verbose)
            if epoch >= 5 or epoch == num_epochs:
                cur_preds = model.inference(em_phase, test_loader,
                                            verbose=verbose)
                train_acc, val_acc, _ = evaluate(cur_preds, ['train', 'valid'])

                print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f},')

                if val_acc <= best_val_acc:
                    early_stopping += 1
                    if early_stopping > patience:
                        print(f'''Early stopped by Epoch: {epoch}, \
                            Best acc: {best_val_acc}''')
                        break
                else:
                    best_val_acc = val_acc

        preds = model.inference(em_phase, test_loader, verbose=verbose)
        if em_phase == 'gnn':
            gnn_val_acc = max(gnn_val_acc, best_val_acc)
            model.gnn = model.gnn.to('cpu', non_blocking=True)
            em_phase = 'lm'
        else:
            lm_val_acc = max(lm_val_acc, best_val_acc)
            model.lm = model.lm.to('cpu', non_blocking=True)
            em_phase = 'gnn'
        torch.cuda.empty_cache()
    print(f'Best GNN validation acc: {gnn_val_acc},'
          f'LM validation acc: {lm_val_acc}')
    print('============================')
    if gnn_val_acc > lm_val_acc:
        em_phase = 'gnn'
        model.gnn = model.gnn.to(device, non_blocking=True)
        test_loader = subgraph_loader
    else:
        em_phase = 'lm'
        model.lm = model.lm.to(device, non_blocking=True)
        test_loader = text_test_loader
    test_preds = model.inference(em_phase, test_loader, verbose=verbose)
    train_acc, val_acc, test_acc = evaluate(test_preds,
                                            ['train', 'valid', 'test'])
    final_test_acc = max(gnn_test_acc, max(lm_test_acc, test_acc))
    print(f'Best test acc: {final_test_acc}, model: {em_phase}')
    end_time = time.time()
    running_time = (end_time - start_time) / 3600
    print(f'Total running time: {running_time:.2f} hours')


if __name__ == '__main__':
    available_gb = psutil.virtual_memory().available / (1024**3)
    if available_gb < 80:
        print(f"  WARNING: This test may require more RAM than available.\n"
              f"    Estimated RAM needed: ~80 GB\n"
              f"    Detected available RAM: {available_gb:.2f} GB\n"
              "    If the program crashes or is killed, consider upgrading "
              "system memory.")

    parser = argparse.ArgumentParser(description='GLEM Example:')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_runs', type=int, default=10,
                        help='number of runs')
    parser.add_argument('--num_em_iters', type=int, default=1,
                        help='number of iterations')
    parser.add_argument("--dataset", type=str, default='arxiv',
                        help='arxiv or products')
    parser.add_argument(
        "--text_type", type=str, default='llm_explanation',
        help="type of text, support raw_text, llm_explanation,"
        "all for arxiv and raw_text for products")
    parser.add_argument("--pl_ratio", type=float, default=0.5,
                        help="pseudo labels ratio")
    parser.add_argument('--hf_model', type=str, default='prajjwal1/bert-tiny',
                        help='huggingface model repo id')
    parser.add_argument(
        '--gnn_model', type=str, default='SAGE',
        help='gnn model for node classification,'
        'options: SAGE, GAT, GCN')
    parser.add_argument('--gnn_hidden_channels', type=int, default=256)
    parser.add_argument('--gnn_num_layers', type=int, default=3)
    parser.add_argument('--gat_heads', type=int, default=4,
                        help='Number of multi-head-attentions for GAT ')
    parser.add_argument('--lm_batch_size', type=int, default=256)
    parser.add_argument('--gnn_batch_size', type=int, default=1024)
    parser.add_argument(
        '--external_pred_path', type=str, default=None,
        help="Other model's output logits during the "
        "pretraining phase or simply concatenate it with"
        "node features as augmented data for gnn")
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='pseudo label weight in E-step')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='pseudo label weight in M-step')
    parser.add_argument('--lm_epochs', type=int, default=10)
    parser.add_argument('--gnn_epochs', type=int, default=50)
    parser.add_argument('--gnn_lr', type=float, default=0.002)
    parser.add_argument('--lm_lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=3,
                        help='Patience for early stopping')
    parser.add_argument('--verbose', action='store_true',
                        help='show progress bar during training or not')
    parser.add_argument('--em_order', type=str, default='lm',
                        help='decide train LM first or GNN first')
    parser.add_argument('--lm_use_lora', action='store_true',
                        help='use Lora to fine-tune model or not')
    parser.add_argument(
        '--token_on_disk', action='store_true',
        help='save token on disk and load token from disk'
        'for reducing duplicated tokenizing')
    parser.add_argument('--out_dir', type=str, default='output/',
                        help='output directory')
    parser.add_argument(
        '--train_without_ext_pred', action='store_true',
        help='train glem without using additional pseudo labels '
        'for augmenting data only available for ogbn-products')
    args = parser.parse_args()
    print(args)
    main(args)
