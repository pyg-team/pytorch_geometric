import hydra

from torch_geometric.graphgym.config_store import config_store  # noqa


@hydra.main(config_path='.', config_name='my_config')
def main(cfg):
    print(cfg)

    dataset = hydra.utils.instantiate(cfg.dataset)
    print(dataset)
    data = dataset[0]
    print(data)


if __name__ == '__main__':
    main()
