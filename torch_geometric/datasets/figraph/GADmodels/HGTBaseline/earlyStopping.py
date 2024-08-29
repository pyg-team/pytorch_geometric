import pandas as pd


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, delta=0):
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


def early_stopping_73(year, total_year, times, dirname):
    best_data = []
    best_epoch = 0
    all_best_data = []
    for each_year in range(total_year):
        best_data = []
        for t in range(times):
            data_valid = pd.read_csv(dirname + '/' + str(each_year) + "_" +
                                     str(t) + "_" + str(each_year + year) +
                                     "_valid.csv")
            data_test = pd.read_csv(dirname + '/' + str(each_year) + "_" +
                                    str(t) + "_" + str(each_year + year) +
                                    "_test.csv")

            # data_test = pd.read_csv(test_path)
            loss_valid = data_valid['loss'].values.tolist()

            early_stopping = EarlyStopping()
            for epoch, loss in enumerate(loss_valid):
                early_stopping(loss, epoch)

            best_epoch = early_stopping.best_epoch
            early_stopping.best_score
            print(best_epoch)

            best_data.append(data_test[(data_test['epoch'] == best_epoch)])

        pd.concat(best_data).to_csv(
            dirname + '/result' + "_" + str(each_year + year) + ".csv",
            index=False, encoding='utf-8')
        all_best_data.extend(best_data)
    pd.concat(all_best_data).to_csv(dirname + '/result.csv', index=False,
                                    encoding='utf-8')


if __name__ == "__main__":
    # new_early_stopping()
    # old_early_stopping()
    # ACM_early_stopping(100)
    # IMDB_early_stopping(100)
    # DBLP_early_stopping(100)
    early_stopping_73(2018, 5, 1, 'HGT_result')
