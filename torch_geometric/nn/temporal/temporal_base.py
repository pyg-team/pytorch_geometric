"""
EdgeBank is a simple strong baseline for dynamic link prediction
it predicts the existence of edges based on their history of occurrence

Reference:
    - https://github.com/fpour/DGB/tree/main
"""

import warnings

import torch


class TemporalBase(object):
    def __init__(
        self,
        src: torch.tensor,
        dst: torch.tensor,
        ts: torch.tensor,
        memory_mode:
        str = 'unlimited',  # could be `unlimited` or `fixed_time_window`
        time_window_ratio: float = 0.15,
        pos_prob: float = 1.0,
    ):
        r"""
        intialize edgebank and specify the memory mode
        Parameters:
            src: source node id of the edges for initialization
            dst: destination node id of the edges for initialization
            ts: timestamp of the edges for initialization
            memory_mode: 'unlimited' or 'fixed_time_window'
            time_window_ratio: the ratio of the time window length to the total time length
            pos_prob: the probability of the link existence for the edges in memory
        """
        raise NotImplementedError

    def update_memory(self, src: torch.tensor, dst: torch.tensor,
                      ts: torch.tensor):
        r"""
        generate the current and correct state of the memory with the observed edges so far
        note that historical edges may include training, validation, and already observed test edges
        Parameters:
            src: source node id of the edges
            dst: destination node id of the edges
            ts: timestamp of the edges
        """
        raise NotImplementedError

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

    def _update_unlimited_memory(self, update_src: torch.tensor,
                                 update_dst: torch.tensor):
        r"""
        update self.memory with newly arrived src and dst
        Parameters:
            src: source node id of the edges
            dst: destination node id of the edges
        """
        raise NotImplementedError

    def _update_time_window_memory(self, update_src: torch.tensor,
                                   update_dst: torch.tensor,
                                   update_ts: torch.tensor) -> None:
        r"""
        move the time window forward until end of dst timestamp here
        also need to remove earlier edges from memory which is not in the time window
        Parameters:
            update_src: source node id of the edges
            update_dst: destination node id of the edges
            update_ts: timestamp of the edges
        """

        raise NotImplementedError

    def predict_link(self, query_src: torch.tensor,
                     query_dst: torch.tensor) -> torch.tensor:
        r"""
        predict the probability from query src,dst pair given the current memory,
        all edges not in memory will return 0.0 while all observed edges in memory will return self.pos_prob
        Parameters:
            query_src: source node id of the query edges
            query_dst: destination node id of the query edges
        Returns:
            pred: the prediction for all query edges
        """
        raise NotImplementedError
