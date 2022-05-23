import os.path as osp

import ignite
import ignite.contrib.handlers.tensorboard_logger
import ignite.contrib.handlers.tqdm_logger
import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric import seed_everything
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GIN, MLP, global_add_pool


class Model(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 hidden_channels: int = 64, num_layers: int = 3,
                 dropout: float = 0.5):
        super().__init__()

        self.gnn = GIN(in_channels, hidden_channels, num_layers,
                       dropout=dropout, jk='cat')

        self.classifier = MLP([hidden_channels, hidden_channels, out_channels],
                              batch_norm=True, dropout=dropout)

    def forward(self, data):
        x = self.gnn(data.x, data.edge_index)
        x = global_add_pool(x, data.batch)
        x = self.classifier(x)
        return x


def main():
    seed_everything(42)

    root = osp.join('data', 'TUDataset')
    dataset = TUDataset(root, 'IMDB-BINARY', pre_transform=T.OneHotDegree(135))

    dataset = dataset.shuffle()
    test_dataset = dataset[:len(dataset) // 10]
    val_dataset = dataset[len(dataset) // 10:2 * len(dataset) // 10]
    train_dataset = dataset[2 * len(dataset) // 10:]

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(dataset.num_node_features, dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    metrics = {'acc': ignite.metrics.Accuracy()}

    def prepare_batch_fn(batch, device, non_blocking):
        return (batch.to(device, non_blocking=non_blocking),
                batch.y.to(device, non_blocking=non_blocking))

    trainer = ignite.engine.create_supervised_trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=F.cross_entropy,
        device=device,
        prepare_batch=prepare_batch_fn,
        output_transform=lambda x, y, y_pred, loss: loss.item(),
        amp_mode='amp',
    )

    # Progress bar for each epoch:
    pbar = ignite.contrib.handlers.tqdm_logger.ProgressBar()
    pbar.attach(trainer, output_transform=lambda x: {'loss': x})

    def log_metrics(evaluator, loader, tag):
        def logger(trainer):
            evaluator.run(loader)
            print(f'{tag:10} Epoch: {trainer.state.epoch:02d}, '
                  f'Acc: {evaluator.state.metrics["acc"]:.4f}')

        return logger

    train_evaluator = ignite.engine.create_supervised_evaluator(
        model=model,
        metrics=metrics,
        device=device,
        prepare_batch=prepare_batch_fn,
        output_transform=lambda x, y, y_pred: (y_pred, y),
        amp_mode='amp',
    )
    trainer.on(ignite.engine.Events.EPOCH_COMPLETED(every=1))(log_metrics(
        train_evaluator, train_loader, 'Training'))

    val_evaluator = ignite.engine.create_supervised_evaluator(
        model=model,
        metrics=metrics,
        device=device,
        prepare_batch=prepare_batch_fn,
        output_transform=lambda x, y, y_pred: (y_pred, y),
        amp_mode='amp',
    )
    trainer.on(ignite.engine.Events.EPOCH_COMPLETED(every=1))(log_metrics(
        val_evaluator, val_loader, 'Validation'))

    test_evaluator = ignite.engine.create_supervised_evaluator(
        model=model,
        metrics=metrics,
        device=device,
        prepare_batch=prepare_batch_fn,
        output_transform=lambda x, y, y_pred: (y_pred, y),
        amp_mode='amp',
    )
    trainer.on(ignite.engine.Events.EPOCH_COMPLETED(every=1))(log_metrics(
        test_evaluator, test_loader, 'Test'))

    # Save checkpoint of the model based on Accuracy on the validation set:
    checkpoint_handler = ignite.handlers.Checkpoint(
        {'model': model},
        'runs/gin',
        n_saved=2,
        score_name=list(metrics.keys())[0],
        filename_pattern='best-{global_step}-{score_name}-{score}.pt',
        global_step_transform=ignite.handlers.global_step_from_engine(trainer),
    )
    val_evaluator.add_event_handler(ignite.engine.Events.EPOCH_COMPLETED,
                                    checkpoint_handler)

    # Create a tensorboard logger to write logs:
    tb_logger = ignite.contrib.handlers.tensorboard_logger.TensorboardLogger(
        log_dir=osp.join('runs/example', 'tb_logs'))

    tb_logger.attach_output_handler(
        trainer, event_name=ignite.engine.Events.ITERATION_COMPLETED,
        tag='training', output_transform=lambda loss: {'loss_iteration': loss})
    tb_logger.attach_output_handler(
        trainer, event_name=ignite.engine.Events.EPOCH_COMPLETED,
        tag='training', output_transform=lambda loss: {'loss_epoch': loss})
    tb_logger.attach_output_handler(
        train_evaluator,
        event_name=ignite.engine.Events.EPOCH_COMPLETED,
        tag='training',
        metric_names='all',
        global_step_transform=ignite.handlers.global_step_from_engine(trainer),
    )
    tb_logger.attach_output_handler(
        val_evaluator,
        event_name=ignite.engine.Events.EPOCH_COMPLETED,
        tag='validation',
        metric_names='all',
        global_step_transform=ignite.handlers.global_step_from_engine(trainer),
    )
    tb_logger.attach_output_handler(
        test_evaluator,
        event_name=ignite.engine.Events.EPOCH_COMPLETED,
        tag='test',
        metric_names='all',
        global_step_transform=ignite.handlers.global_step_from_engine(trainer),
    )
    tb_logger.close()

    trainer.run(train_loader, max_epochs=50)


if __name__ == '__main__':
    main()
