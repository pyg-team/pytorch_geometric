import argparse
import time
from collections import OrderedDict

import numpy as np
import torch
import yaml
from src.model_handler import ModelHandler

################################################################################
# Main #
################################################################################


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main(config):
    print_config(config)
    set_random_seed(config['seed'])
    model = ModelHandler(config)
    f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test = model.train()
    print(f"F1-Macro: {f1_mac_test}")
    print(f"AUC: {auc_test}")
    print(f"G-Mean: {gmean_test}")


def multi_run_main(config):
    print_config(config)
    hyperparams = []
    for k, v in config.items():
        if isinstance(v, list):
            hyperparams.append(k)

    f1_list, f1_1_list, f1_0_list, auc_list, gmean_list = [], [], [], [], []
    configs = grid(config)
    for i, cnf in enumerate(configs):
        print(f'Running {i}:\n')
        for k in hyperparams:
            cnf['save_dir'] += f'{k}_{cnf[k]}_'
        print(cnf['save_dir'])
        set_random_seed(cnf['seed'])
        st = time.time()
        model = ModelHandler(cnf)
        f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test = model.train()
        f1_list.append(f1_mac_test)
        f1_1_list.append(f1_1_test)
        f1_0_list.append(f1_0_test)
        auc_list.append(auc_test)
        gmean_list.append(gmean_test)
        print(f"Running {i} done, elapsed time {time.time()-st}s")

    print(f"F1-Macro: {f1_list}")
    print(f"AUC: {auc_list}")
    print(f"G-Mean: {gmean_list}")

    f1_mean, f1_std = np.mean(f1_list), np.std(f1_list, ddof=1)
    f1_1_mean, f1_1_std = np.mean(f1_1_list), np.std(f1_1_list, ddof=1)
    f1_0_mean, f1_0_std = np.mean(f1_0_list), np.std(f1_0_list, ddof=1)
    auc_mean, auc_std = np.mean(auc_list), np.std(auc_list, ddof=1)
    gmean_mean, gmean_std = np.mean(gmean_list), np.std(gmean_list, ddof=1)

    print(f"F1-Macro: {f1_mean}+{f1_std}")
    print(f"F1-binary-1: {f1_1_mean}+{f1_1_std}")
    print(f"F1-binary-0: {f1_0_mean}+{f1_0_std}")
    print(f"AUC: {auc_mean}+{auc_std}")
    print(f"G-Mean: {gmean_mean}+{gmean_std}")


################################################################################
# ArgParse and Helper Functions #
################################################################################
def get_config(config_path="config.yml"):
    with open(config_path) as setting:
        config = yaml.load(setting, Loader=yaml.FullLoader)
    return config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--config',
                        default='./config/pcgnn_yelpchi.yml', type=str,
                        help='path to the config file')
    parser.add_argument('--multi_run', action='store_true',
                        help='flag: multi run')
    args = vars(parser.parse_args())
    return args


def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = f"{key}" + (" " * (24 - len(key)))
        print(f"{keystr} -->   {val}")
    print("**************** MODEL CONFIGURATION ****************")


def grid(kwargs):
    """Builds a mesh grid with given keyword arguments for this Config class.
    If the value is not a list, then it is considered fixed
    """
    class MncDc:
        """This is because np.meshgrid does not always work properly..."""
        def __init__(self, a):
            self.a = a  # tuple!

        def __call__(self):
            return self.a

    def merge_dicts(*dicts):
        """Merges dictionaries recursively. Accepts also `None` and returns always a (possibly empty) dictionary
        """
        from functools import reduce

        def merge_two_dicts(x, y):
            z = x.copy()  # start with x's keys and values
            z.update(y)  # modifies z with y's keys and values & returns None
            return z

        return reduce(lambda a, nd: merge_two_dicts(a, nd
                                                    if nd else {}), dicts, {})

    sin = OrderedDict({k: v for k, v in kwargs.items() if isinstance(v, list)})
    for k, v in sin.items():
        copy_v = []
        for e in v:
            copy_v.append(MncDc(e) if isinstance(e, tuple) else e)
        sin[k] = copy_v

    grd = np.array(np.meshgrid(*sin.values()),
                   dtype=object).T.reshape(-1, len(sin.values()))
    return [
        merge_dicts(
            {
                k: v
                for k, v in kwargs.items() if not isinstance(v, list)
            }, {
                k: vv[i]() if isinstance(vv[i], MncDc) else vv[i]
                for i, k in enumerate(sin)
            }) for vv in grd
    ]


################################################################################
# Module Command-line Behavior #
################################################################################
if __name__ == '__main__':
    cfg = get_args()
    config = get_config(cfg['config'])
    if cfg['multi_run']:
        multi_run_main(config)
    else:
        main(config)
