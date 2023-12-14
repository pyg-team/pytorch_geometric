import ast
import json
import os
import os.path as osp

from torch_geometric.io import fs


def string_to_python(string):
    try:
        return ast.literal_eval(string)
    except Exception:
        return string


def dict_to_json(dict, fname):
    """Dump a :python:`Python` dictionary to a JSON file.

    Args:
        dict (dict): The :python:`Python` dictionary.
        fname (str): The output file name.
    """
    with open(fname, 'a') as f:
        json.dump(dict, f)
        f.write('\n')


def dict_list_to_json(dict_list, fname):
    """Dump a list of :python:`Python` dictionaries to a JSON file.

    Args:
        dict_list (list of dict): List of :python:`Python` dictionaries.
        fname (str): the output file name.
    """
    with open(fname, 'a') as f:
        for dict in dict_list:
            json.dump(dict, f)
            f.write('\n')


def json_to_dict_list(fname):
    dict_list = []
    epoch_set = set()
    with open(fname) as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            dict = json.loads(line)
            if dict['epoch'] not in epoch_set:
                dict_list.append(dict)
            epoch_set.add(dict['epoch'])
    return dict_list


def dict_to_tb(dict, writer, epoch):
    """Add a dictionary of statistics to a Tensorboard writer.

    Args:
        dict (dict): Statistics of experiments, the keys are attribute names,
        the values are the attribute values
        writer: Tensorboard writer object
        epoch (int): The current epoch
    """
    for key in dict:
        writer.add_scalar(key, dict[key], epoch)


def dict_list_to_tb(dict_list, writer):
    for dict in dict_list:
        assert 'epoch' in dict, 'Key epoch must exist in stats dict'
        dict_to_tb(dict, writer, dict['epoch'])


def makedirs_rm_exist(dir):
    """Make a directory, remove any existing data.

    Args:
        dir (str): The directory to be created.
    """
    if osp.isdir(dir):
        fs.rm(dir)
    os.makedirs(dir, exist_ok=True)
