from ogb.nodeproppred import PygNodePropPredDataset

from torch_geometric.datasets import TAGDataset


def main():
    dataset_name = 'arxiv'
    root = './data/ogb'
    hf_model = 'prajjwal1/bert-tiny'
    token_on_disk = True

    dataset = PygNodePropPredDataset(f'ogbn-{dataset_name}', root=root)
    dataset.get_idx_split()

    tag_dataset = TAGDataset(root, dataset, hf_model,
                             token_on_disk=token_on_disk)
    raw_text_dataset = tag_dataset.to_text_dataset()
    llm_explanation_dataset = tag_dataset.to_text_dataset(
        text_type='llm_explanation')
    print(tag_dataset.num_classes, tag_dataset.raw_file_names)
    print(raw_text_dataset)
    print(llm_explanation_dataset)


if __name__ == '__main__':
    main()
