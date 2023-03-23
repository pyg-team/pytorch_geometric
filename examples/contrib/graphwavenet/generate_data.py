from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import argparse
import os

import numpy as np
import pandas as pd
import requests


def generate_graph_seq2seq_io_data(df, x_offsets, y_offsets,
                                   add_time_in_day=True):

    # The individual time steps are aggregated to generate the model input and output
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    feature_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values -
                    df.index.values.astype("datetime64[D]")) / np.timedelta64(
                        1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_in_day)

    data = np.concatenate(feature_list, axis=-1)
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):  # t is the index of the last observation.
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def generate_train_val_test(args):
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y

    # If the file is not present, it is downloaded
    filename = args.traffic_df_filename
    if filename == None:
        url = "https://github.com/Kumbong/CS224W-GraphWavenet/blob/main/data/metr-la.h5?raw=true"
        filename = 'metr-la.h5'
        f = open(filename, 'wb')
        f.write(requests.get(url).content)
        f.close()
        df = pd.read_hdf(filename)
        os.remove(filename)
    else:
        df = pd.read_hdf(args.traffic_df_filename)

    # 0 is the latest observed sample.
    x_offsets = np.sort(
        np.concatenate((np.arange(-(seq_length_x - 1), 1, 1), )))
    # Predict the next one hour
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))
    x, y = generate_graph_seq2seq_io_data(df, x_offsets=x_offsets,
                                          y_offsets=y_offsets,
                                          add_time_in_day=True)

    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = (
        x[num_train:num_train + num_val],
        y[num_train:num_train + num_val],
    )
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/METR-LA",
                        help="Output directory.")
    parser.add_argument(
        "--traffic_df_filename",
        type=str,
        default=None,
        help="Raw traffic readings.",
    )
    parser.add_argument(
        "--seq_length_x",
        type=int,
        default=12,
        help="Sequence Length.",
    )
    parser.add_argument(
        "--seq_length_y",
        type=int,
        default=12,
        help="Sequence Length.",
    )
    parser.add_argument(
        "--y_start",
        type=int,
        default=1,
        help="Y pred start",
    )

    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        reply = str(
            input(
                f'{args.output_dir} exists. Do you want to overwrite it? (y/n)'
            )).lower().strip()
        if reply[0] != 'y': exit
    else:
        os.makedirs(args.output_dir)
    generate_train_val_test(args)
