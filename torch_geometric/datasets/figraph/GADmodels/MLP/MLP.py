import copy

import numpy as np
import pandas as pd
from metrics_utils import evaluate
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def load_data(data, split_year):
    trainset = copy.deepcopy(data[(data['Year'] >= split_year['train'][0])
                                  & (data['Year'] <= split_year['train'][-1])])
    validset = copy.deepcopy(data[(data['Year'] >= split_year['valid'][0])
                                  & (data['Year'] <= split_year['valid'][-1])])
    testset = copy.deepcopy(data[(data['Year'] >= split_year['test'][0])
                                 & (data['Year'] <= split_year['test'][-1])])

    trainset.drop(columns=['nodeID', 'Year'], axis=1, inplace=True)
    validset.drop(columns=['nodeID', 'Year'], axis=1, inplace=True)
    testset.drop(columns=['nodeID', 'Year'], axis=1, inplace=True)

    cols = list(data.columns)
    std_cols = [col for col in cols if col not in ['nodeID', 'Year', 'Label']]
    trainset[std_cols] = StandardScaler().fit_transform(trainset[std_cols])
    validset[std_cols] = StandardScaler().fit_transform(validset[std_cols])
    testset[std_cols] = StandardScaler().fit_transform(testset[std_cols])

    train_y = trainset['Label'].values
    train_x = trainset.drop(columns=['Label']).values

    valid_y = validset['Label'].values
    valid_x = validset.drop(columns=['Label']).values

    test_y = testset['Label'].values
    test_x = testset.drop(columns=['Label']).values

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


def get_evaluate(model, feats, labels, settype, params, dataset_name, os_name,
                 repeat):
    y_preds = model.predict(feats)
    if len(np.unique(labels)) <= 2:
        y_probs = model.predict_proba(feats)[:, -1]
    elif len(np.unique(labels)) > 2:
        y_probs = model.predict_proba(feats)
    result = evaluate(labels=labels, y_preds=y_preds, y_probs=y_probs,
                      epo=None, loss=None, params=str(params))
    result['settype'] = settype
    result['Dataset'] = dataset_name
    result['os_name'] = os_name
    result['repeat'] = repeat
    return result


if __name__ == '__main__':
    data = pd.read_csv(
        'ListedCompanyFeatures_AP_missvalue.csv')  # 传入不同的缺失值处理后的文件

    split_dicts = {
        0: {
            'train': [2014, 2015, 2016],
            'valid': [2017],
            'test': [2018]
        },
        1: {
            'train': [2014, 2015, 2016, 2017],
            'valid': [2018],
            'test': [2019]
        },
        2: {
            'train': [2014, 2015, 2016, 2017, 2018],
            'valid': [2019],
            'test': [2020]
        },
        3: {
            'train': [2014, 2015, 2016, 2017, 2018, 2019],
            'valid': [2020],
            'test': [2021]
        },
        4: {
            'train': [2014, 2015, 2016, 2017, 2018, 2019, 2020],
            'valid': [2021],
            'test': [2022]
        }
    }

    hidden_layer_sizes = (256, 32)
    solver = 'adam'
    activation = 'relu'
    lr = 0.0001
    alpha = 0.1
    max_iter = 600

    seed = 2024
    result_list = []
    for repeat in [0]:
        split_year = split_dicts[repeat]
        (train_x, train_y), (valid_x,
                             valid_y), (test_x,
                                        test_y) = load_data(data, split_year)

        clf = MLPClassifier(random_state=seed,
                            hidden_layer_sizes=hidden_layer_sizes,
                            solver=solver, activation=activation,
                            learning_rate_init=lr, alpha=alpha,
                            max_iter=max_iter)

        params = {
            'random_state': seed,
            'hidden_layer_sizes': hidden_layer_sizes,
            'solver': solver,
            'activation': activation,
            'lr': lr,
            'alpha': alpha,
            'max_iter': max_iter
        }
        clf.fit(train_x, train_y)
        print('ok')

        eval_list = [(train_x, train_y, 'train'), (valid_x, valid_y, 'valid'),
                     (test_x, test_y, 'test')]
        for feats, labels, settype in eval_list:
            result = get_evaluate(model=clf, feats=feats, labels=labels,
                                  settype=settype, params=str(params),
                                  dataset_name=None, os_name=None,
                                  repeat=repeat)
            result_list.append(result)

    result_os = pd.concat(result_list, axis=0)
    result_os.index = range(result_os.shape[0])

    result_os.to_csv('./result/MLP_result.csv', index=False, encoding='utf-8')
