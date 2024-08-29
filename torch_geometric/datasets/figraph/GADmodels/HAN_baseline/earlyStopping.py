import numpy as np
import pandas as pd
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss, epoch):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        else:
            if score < self.best_score + self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.best_epoch = epoch
                self.counter = 0

best_data = []
best_epoch = 0
best_score = 0

def new_early_stopping():
    times = 5
    valid_year = 2013
    test_year = 2014
    for time in range(times):
        data_valid = pd.read_csv("result/" + str(time) + "_" + str(valid_year + time * 2) + ".csv")
        data_test = pd.read_csv("result/" + str(time) + "_" + str(test_year + time * 2) + ".csv")
        print(data_valid.shape)

        # data_test = pd.read_csv(test_path)
        loss_valid = data_valid['loss'].values.tolist()

        early_stopping = EarlyStopping(patience=100)
        for epoch, loss in enumerate(loss_valid):
            early_stopping(loss, epoch)

        best_epoch = early_stopping.best_epoch
        best_score = early_stopping.best_score
        print(best_epoch)

        best_data.append(data_test[(data_test['epoch'] == best_epoch)])

    pd.concat(best_data).to_csv("result/result.csv", index=False, encoding='utf-8')

def old_early_stopping():
    times = 5
    valid_year = 2017
    test_year = 2018
    for time in range(times):
        data_valid = pd.read_csv("result/" + str(time) + "_" + str(valid_year + time) + ".csv")
        data_test = pd.read_csv("result/" + str(time) + "_" + str(test_year + time) + ".csv")
        print(data_valid.shape)

        # data_test = pd.read_csv(test_path)
        loss_valid = data_valid['loss'].values.tolist()

        early_stopping = EarlyStopping()
        for epoch, loss in enumerate(loss_valid):
            early_stopping(loss, epoch)

        best_epoch = early_stopping.best_epoch
        best_score = early_stopping.best_score
        print(best_epoch)

        best_data.append(data_test[(data_test['epoch'] == best_epoch)])

    pd.concat(best_data).to_csv("result/result.csv", index=False, encoding='utf-8')

if __name__ == "__main__":
    # new_early_stopping()
    new_early_stopping()