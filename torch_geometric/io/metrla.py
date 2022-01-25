import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Union, Optional


class MetrLaIo:

    def __init__(self,
                 n_readings,
                 n_previous_steps,
                 n_future_steps,
                 add_time_of_day,
                 add_day_of_week,
                 normalized_k):
        self.n_readings = n_readings
        self.n_previous_steps = n_previous_steps
        self.n_future_steps = n_future_steps

        self.add_time_of_day = add_time_of_day
        self.add_day_of_week = add_day_of_week

        self.normalized_k = normalized_k

        self.x: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.adjacency_matrix: Optional[np.ndarray] = None

        self.previous_offsets = np.arange(start=-self.n_previous_steps + 1,
                                          stop=1,
                                          step=1)
        self.future_offsets = np.arange(start=1,
                                        stop=self.n_future_steps + 1,
                                        step=1)

    @property
    def min_t(self):
        return abs(min(self.previous_offsets))

    @property
    def max_t(self):
        return abs(self.n_readings - abs(max(self.future_offsets)))

    @property
    def dataset_len(self):
        return self.max_t - self.min_t - 1

    @property
    def data(self):
        return self.x, self.y

    def load_metrla_data(self,
                         data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the MetrLA data.

        The returned values X (features/predictors/previous steps) and Y (target/next steps) are of shapes:
        X(n_intervals, n_previous_steps, n_nodes, n_features), concretely (..., 207, 1).
        Y(n_intervals, n_next_steps, n_nodes, n_features), concretely (..., 207, 1).

        :param df:

        :return: A tuple containing the X and Y tensors.
        """

        data_df = pd.read_hdf(data_path)
        n_samples, n_nodes = data_df.shape
        data = np.expand_dims(data_df.values, axis=-1)

        data = [data]
        if self.add_time_of_day:
            time_idx = (data_df.index.values - data_df.index.values.astype("datetime64[D]")) / np.timedelta(1, "D")
            time_of_day = np.tile(time_id, [1, num_nodes, 1]).transpose((2, 1, 0))
            data.append(time_of_day)
        if self.add_day_of_week:
            day_of_week = np.zeros(shape=(n_samples, n_nodes, 7))
            day_of_week[np.arange(n_samples), :, data_df.index.dayofweek] = 1
            data.append(day_of_week)

        data = np.concatenate(data, axis=-1)

        x, y = [], []

        for t in range(self.min_t, self.max_t):
            x.append(data[t + self.previous_offsets, ...])
            y.append(data[t + self.future_offsets, ...])

        self.x = np.stack(x, axis=0)
        self.y = np.stack(y, axis=0)

    def generate_adjacency_matrix(self,
                                  distances_path,
                                  sensor_ids_path) -> tuple[list, dict, np.ndarray]:
        """
        Generates the adjacency matrix of a distance graph using a thresholded Gaussian filter.
        Source: https://github.com/liyaguang/DCRNN/blob/master/scripts/gen_adj_mx.py

        :param distances_df: A dataframe with real-road distances between sensors, of form (to, from, dist).
        :param sensor_ids:
        :param normalized_k:
        :return:
        """

        distances_df = pd.read_csv(filepath_or_buffer=distances_path)
        sensor_ids = self.read_sensor_ids(path=sensor_ids_path)

        n_nodes = len(sensor_ids)

        # Just to optimize for membership
        sensor_ids = set(sensor_ids)

        distances = np.full(shape=(n_nodes, n_nodes),
                            fill_value=np.inf,
                            dtype=np.float32)

        sensor_id_to_idx = {}
        for idx, sensor_id in enumerate(sensor_ids):
            sensor_id_to_idx[sensor_id] = idx 

        for index, series in distances_df.iterrows():
            src, dst, value = series.items()
            if src in sensor_ids and dst in sensor_ids:
                distances[src, dst] = dist

        std = distances[~np.isinf(distances)].flatten().std()

        adjacency_matrix = np.exp(-np.square(distances / std))

        adjacency_matrix[adjacency_matrix < self.normalized_k] = 0.

        self.adjacency_matrix = adjacency_matrix

    @staticmethod
    def read_sensor_ids(path: Union[str, Path]) -> list[str]:
        with open(path, "r") as input_file:
            sensor_ids = input_file.read()
            return list(map(int, sensor_ids.split(",")))
