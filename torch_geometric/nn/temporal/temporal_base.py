"""
EdgeBank is a simple strong baseline for dynamic link prediction
it predicts the existence of edges based on their history of occurrence

Reference:
    - https://github.com/fpour/DGB/tree/main
"""

import warnings

import torch


class TemporalBase(torch.nn.Module):
    def __init__(
        self,
        memory_mode:
        str = 'unlimited',  # could be `unlimited` or `fixed_time_window`
        time_window_ratio: float = 0.15,
        pos_prob: float = 1.0,
    ):
        r"""
        intialize module and specify the memory mode
        Parameters:
            memory_mode: 'unlimited' or 'fixed_time_window'
            time_window_ratio: the ratio of the time window length to the total time length
            pos_prob: the probability of the link existence for the edges in memory
        """
        raise NotImplementedError

    def update_memory(self, edge_index: torch.tensor, ts: torch.tensor):
        r"""
        generate the current and correct state of the memory with the observed edges so far
        note that historical edges may include training, validation, and already observed test edges
        Parameters:
            edge_index: [2, num_edges] tensor of edge indices
            ts: timestamp of the edges
        """
        if self.memory_mode == 'unlimited':
            self._update_unlimited_memory(edge_index)  #ignores time
        elif self.memory_mode == 'fixed_time_window':
            self._update_time_window_memory(edge_index, ts)
        else:
            raise ValueError("Invalide memory mode!")

    @property
    def start_time(self) -> int:
        """
        return the start of time window for edgebank `fixed_time_window` only
        Returns:
            start of time window
        """
        raise NotImplementedError

    @property
    def end_time(self) -> int:
        """
        return the end of time window for edgebank `fixed_time_window` only
        Returns:
            end of time window
        """
        raise NotImplementedError

    def _update_unlimited_memory(self, update_edge_index: torch.tensor):
        r"""
        update self.memory with newly arrived edge indices
        Parameters:
            update_edge_index: [2, num_edges] tensor of edge indices
        """
        raise NotImplementedError

    def _update_time_window_memory(self, update_edge_index: torch.tensor,
                                   update_ts: torch.tensor) -> None:
        r"""
        move the time window forward until end of dst timestamp here
        also need to remove earlier edges from memory which is not in the time window
        Parameters:
            update_edge_index: [2, num_edges] tensor of edge indices
            update_ts: timestamp of the edges
        """

        raise NotImplementedError

    def predict_link(self, query_edge_indices: torch.tensor) -> torch.tensor:
        r"""
        predict the probability from query src,dst pair given the current memory,
        all edges not in memory will return 0.0 while all observed edges in memory will return self.pos_prob
        Parameters:
            query_edge_indices: [2, num_edges] tensor of edge indices
        Returns:
            pred: the prediction for all query edges
        """
        raise NotImplementedError
