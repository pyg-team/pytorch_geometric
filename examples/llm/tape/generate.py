import torch
from ogb.nodeproppred import PygNodePropPredDataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

from torch_geometric.datasets import TAGDataset
from torch_geometric.loader import DataLoader


def main():
    dataset_name = 'arxiv'
    root = './data/ogb'
    hf_model = 'prajjwal1/bert-tiny'
    token_on_disk = True

    dataset = PygNodePropPredDataset(f'ogbn-{dataset_name}', root=root)
    split_idx = dataset.get_idx_split()
    tag_dataset = TAGDataset(root, dataset, hf_model,
                             token_on_disk=token_on_disk)
    raw_text_dataset = tag_dataset.to_text_dataset()
    llm_explanation_dataset = tag_dataset.to_text_dataset(
        text_type='llm_explanation')
    print(tag_dataset.num_classes, tag_dataset.raw_file_names)
    print(raw_text_dataset)
    print(llm_explanation_dataset)

    # Train LM =========================================
    lm_batch_size = 256
    train_dataset = torch.utils.data.Subset(
        llm_explanation_dataset,
        split_idx['train'].nonzero().squeeze().tolist())
    val_dataset = torch.utils.data.Subset(
        llm_explanation_dataset,
        split_idx['valid'].nonzero().squeeze().tolist())
    test_dataset = torch.utils.data.Subset(
        llm_explanation_dataset,
        split_idx['test'].nonzero().squeeze().tolist())

    print('Building language model dataloader...', end='-->')

    train_loader = DataLoader(train_dataset, batch_size=lm_batch_size,
                              drop_last=False, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=lm_batch_size,
                            drop_last=False, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=lm_batch_size * 4,
                             drop_last=False, pin_memory=True, shuffle=False)
    print(f'{len(train_loader)} | {len(val_loader)} | {len(test_loader)}')

    device = torch.device('cuda')
    lm = AutoModelForSequenceClassification.from_pretrained(
        hf_model,
        num_labels=tag_dataset.num_classes,
        torch_dtype=torch.bfloat16,
        offload_folder='offload',
        trust_remote_code=True,
    ).to(device)
    optimizer = torch.optim.Adam(lm.parameters(), lr=1e-3)
    lm_loss = torch.nn.CrossEntropyLoss(reduction='mean')
    # import pdb; pdb.set_trace()

    # Pretrain language model
    num_epochs = 100
    patience = 3
    verbose = True
    best_acc = 0
    early_stopping = 0
    for epoch in range(1, num_epochs + 1):
        # ========================================
        all_out = []
        total_loss = total_correct = 0
        num_nodes = len(train_loader.dataset.indices)
        lm.train()
        if verbose:
            pbar = tqdm(total=num_nodes)
            pbar.set_description(f'Epoch {epoch:02d}')
        for batch in train_loader:
            inputs = {k: v.to(device) for k, v in batch['input'].items()}
            out = lm(**inputs).logits
            labels = batch['labels'].to(device).squeeze()
            loss = lm_loss(out, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            all_out.append(out)
            total_correct += int(out.argmax(dim=-1).eq(labels).sum())
            total_loss += float(loss)
            if verbose:
                pbar.update(batch['n_id'].size(0))

        all_out = torch.cat(all_out, dim=0)
        approx_acc = total_correct / num_nodes
        loss = total_loss / len(train_loader)
        if verbose:
            pbar.close()
        print(f'Epoch {epoch:02d} Loss: {loss:.4f} '
              f'Approx. Train: {approx_acc:.4f}')
        acc = approx_acc
        # ===================================================
        if acc < best_acc:
            early_stopping += 1
            if early_stopping > patience:
                print(f'Early stopped by Epoch: {epoch}, '
                      f'Best acc: {best_acc}')
                break
        best_acc = max(best_acc, acc)


if __name__ == '__main__':
    main()
