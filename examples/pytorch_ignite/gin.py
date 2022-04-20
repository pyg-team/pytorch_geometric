import os.path as osp

import ignite
import ignite.contrib.handlers.tensorboard_logger
import ignite.contrib.handlers.tqdm_logger
import torch

import torch_geometric as pyg


class Model(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 hidden_channels: int = 64, num_layers: int = 3,
                 dropout: float = 0.5):
        super().__init__()
        self.gnn = pyg.nn.GIN(in_channels, hidden_channels, num_layers,
                              dropout=dropout, jk='cat')

        self.classifier = pyg.nn.MLP(
            [hidden_channels, hidden_channels, out_channels], batch_norm=True,
            dropout=dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.gnn(x, edge_index)
        x = pyg.nn.global_add_pool(x, batch)
        x = self.classifier(x)
        return x


def main():
    pyg.seed_everything(42)

    device = torch.device('cuda')
    root = osp.join('data', 'TUDataset')
    log_dir = 'runs/example/'
    dataset = pyg.datasets.TUDataset(
        root, 'IMDB-BINARY', pre_transform=pyg.transforms.OneHotDegree(135))

    amp_mode = 'amp'
    batch_size = 16

    dataset = dataset.shuffle()
    test_dataset = dataset[:len(dataset) // 10]
    val_dataset = dataset[len(dataset) // 10:2 * len(dataset) // 10]
    train_dataset = dataset[2 * len(dataset) // 10:]
    loader_trn = pyg.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    loader_val = pyg.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    loader_test = pyg.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = Model(dataset.num_node_features, dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss = torch.nn.functional.cross_entropy
    metrics = {'acc': ignite.metrics.Accuracy()}

    def prepare_batch_fn(batch, device, non_blocking):
        # return a tuple of (x, y) that can be directly runned as
        # `loss_fn(model(x), y)`
        # dato che ignite non dà argomenti multipli a model, dovremo
        # avere modelli che estraggono dentro forward() i dati di input
        # da `data`
        return (batch.to(device, non_blocking=non_blocking),
                batch.y.to(device, non_blocking=non_blocking))

    trainer = ignite.engine.create_supervised_trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=F.cross_entropy,
        device=device,
        prepare_batch=prepare_batch_fn,
        output_transform=lambda x, y, y_pred, loss: loss.item(),
        amp_mode=amp_mode,
    )
    # progress bar for each epoch
    pbar = ignite.contrib.handlers.tqdm_logger.ProgressBar()
    pbar.attach(trainer, output_transform=lambda x: {"loss": x})

    trn_evaluator = ignite.engine.create_supervised_evaluator(
        model=model, metrics=metrics, device=device,
        prepare_batch=prepare_batch_fn, output_transform=lambda x, y, y_pred:
        (y_pred, y), amp_mode=amp_mode)

    @trainer.on(ignite.engine.Events.EPOCH_COMPLETED(every=1))
    def log_training_metrics(trainer):
        trn_evaluator.run(loader_trn)
        metrics = trn_evaluator.state.metrics
        print(f"{'Training Results - Epoch: ':30}{trainer.state.epoch} ",
              end='')
        for key, value in metrics.items():
            print(f"{key}: {value:.2f} ", end='')
        print()

    val_evaluator = ignite.engine.create_supervised_evaluator(
        model=model,
        metrics=metrics,
        device=device,
        prepare_batch=prepare_batch_fn,
        output_transform=lambda x, y, y_pred: (y_pred, y),
        amp_mode=amp_mode,
    )

    @trainer.on(ignite.engine.Events.EPOCH_COMPLETED(every=1))
    def log_validation_metrics(trainer):
        val_evaluator.run(loader_val)
        metrics = val_evaluator.state.metrics
        print(f"{'Validation Results - Epoch: ':30}{trainer.state.epoch} ",
              end='')
        for key, value in metrics.items():
            print(f"{key}: {value:.2f} ", end='')
        print()

    test_evaluator = ignite.engine.create_supervised_evaluator(
        model=model,
        metrics=metrics,
        device=device,
        prepare_batch=prepare_batch_fn,
        output_transform=lambda x, y, y_pred: (y_pred, y),
        amp_mode=amp_mode,
    )

    @trainer.on(ignite.engine.Events.EPOCH_COMPLETED(every=1))
    def log_validation_metrics(trainer):
        test_evaluator.run(loader_test)
        metrics = test_evaluator.state.metrics
        print(f"{'Test Results - Epoch: ':30}{trainer.state.epoch} ", end='')
        for key, value in metrics.items():
            print(f"{key}: {value:.2f} ", end='')
        print()

    # create an handler to save checkpoints of the
    # model based on Accuracy on the training set
    checkpoint_handler = ignite.handlers.Checkpoint(
        {'model': model}, log_dir, n_saved=2,
        score_name=list(metrics.keys())[0],
        filename_pattern="best_trn_epoch-{global_step}" +
        "-{score_name}-{score}.pt",
        global_step_transform=ignite.handlers.global_step_from_engine(trainer))
    trn_evaluator.add_event_handler(ignite.engine.Events.EPOCH_COMPLETED,
                                    checkpoint_handler)

    # create an handler to save checkpoints of the
    # model based on Accuracy on the validation set
    checkpoint_handler = ignite.handlers.Checkpoint(
        {'model': model}, log_dir, n_saved=2,
        score_name=list(metrics.keys())[0],
        filename_pattern="best_val_epoch-{global_step}" +
        "-{score_name}-{score}.pt",
        global_step_transform=ignite.handlers.global_step_from_engine(trainer))
    val_evaluator.add_event_handler(ignite.engine.Events.EPOCH_COMPLETED,
                                    checkpoint_handler)

    # create a tensorboard logger to write logs
    tb_logger = ignite.contrib.handlers.tensorboard_logger.TensorboardLogger(
        log_dir=osp.join(log_dir, "tb_logs"))
    # Attach the logger to the trainer to log training loss at each iteration
    tb_logger.attach_output_handler(
        trainer, event_name=ignite.engine.Events.ITERATION_COMPLETED,
        tag="training", output_transform=lambda loss: {"loss_iteration": loss})
    # Attach the logger to the trainer to log training loss at each epoch
    tb_logger.attach_output_handler(
        trainer, event_name=ignite.engine.Events.EPOCH_COMPLETED,
        tag="training", output_transform=lambda loss: {"loss_epoch": loss})
    # Attach the logger to the train evaluator
    # to log training metrics at each epoch
    tb_logger.attach_output_handler(
        trn_evaluator,
        event_name=ignite.engine.Events.EPOCH_COMPLETED,
        tag="training",
        metric_names='all',
        global_step_transform=ignite.handlers.global_step_from_engine(trainer),
    )
    # Attach the logger to the validation evaluator
    # to log validation metrics at each epoch
    tb_logger.attach_output_handler(
        val_evaluator,
        event_name=ignite.engine.Events.EPOCH_COMPLETED,
        tag="validation",
        metric_names='all',
        global_step_transform=ignite.handlers.global_step_from_engine(trainer),
    )
    # Attach the logger to the test evaluator to log test metrics at each epoch
    tb_logger.attach_output_handler(
        test_evaluator,
        event_name=ignite.engine.Events.EPOCH_COMPLETED,
        tag="test",
        metric_names='all',
        global_step_transform=ignite.handlers.global_step_from_engine(trainer),
    )
    tb_logger.close()

    # run the trainer
    trainer.run(loader_trn, max_epochs=50)


if __name__ == '__main__':
    main()
