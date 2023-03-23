import argparse
import os
import time
import numpy as np
import torch
import util
from model import Model
"""
An example program for GraphWaveNet
Code in this example has been adapted
from https://github.com/nnzhan/Graph-WaveNet

Usage:
```
python main.py
```

Help:
```
python main.py --help
```

Note: Before running this file, run the following command
to generate the dataset
```
python generate_data.py
```
"""

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu',
                    help='device to run the model on')
parser.add_argument('--data', type=str, default='store/METR-LA',
                    help='data path')
parser.add_argument('--adjdata', type=str, default='store/adj_mx.pkl',
                    help='adj data path')
parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=207,
                    help='number of nodes')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='weight decay rate')
parser.add_argument('--epochs', type=int, default=100, help='')
parser.add_argument('--print_every', type=int, default=50, help='')
parser.add_argument('--save', type=str, default='store/checkpoint',
                    help='save path')

args = parser.parse_args()


def main():
    device = torch.device(args.device)
    _, _, adj_mx = util.load_adj(args.adjdata)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size,
                                   args.batch_size)
    # Mean / std dev scaling is performed to the model output
    scaler = dataloader['scaler']

    engine = Model(scaler, args.num_nodes, args.learning_rate,
                   args.weight_decay, device, adj_mx)

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []
    for i in range(1, args.epochs + 1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x,
                   y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:, 0, :, :])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: ' + \
                    '{:.4f}, Train RMSE: {:.4f}'
                print(
                    log.format(iter, train_loss[-1], train_mape[-1],
                               train_rmse[-1]), flush=True)
        t2 = time.time()
        train_time.append(t2 - t1)
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:, 0, :, :])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        if np.argmin(his_loss) == len(his_loss) - 1:
            torch.save(engine.gwnet.state_dict(),
                       args.save + "/epoch_" + str(i) + ".pth")

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, ' + \
            'Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, ' + \
            'Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(
            log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss,
                       mvalid_mape, mvalid_rmse, (t2 - t1)), flush=True)
    print("Average Training Time: {:.4f} secs/epoch".format(
        np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    # testing
    engine.gwnet.load_state_dict(
        torch.load(args.save + "_epoch_" + str(i) + ".pth"))
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.gwnet(testx).transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    print("Training finished")

    amae = []
    amape = []
    armse = []
    for i in range(12):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = util.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE' + \
            '{:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: ' + \
        '{:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
