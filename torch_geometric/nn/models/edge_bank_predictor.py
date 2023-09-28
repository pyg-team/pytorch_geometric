"""
EdgeBank is a simple strong baseline for dynamic link prediction
it predicts the existence of edges based on their history of occurrence
Reference:
    - https://github.com/fpour/DGB/tree/main
"""

import warnings

import torch


class EdgeBankPredictor(torch.nn.Module):
    def __init__(
        self,
        memory_mode:
        str = 'unlimited',  # could be `unlimited` or `fixed_time_window`
        time_window_ratio: float = 0.15,
        pos_prob: float = 1.0,
    ):
        r"""
        intialize edgebank and specify the memory mode
        Parameters:
            memory_mode: 'unlimited' or 'fixed_time_window'
            time_window_ratio: the ratio of the time window length to the total time length
            pos_prob: the probability of the link existence for the edges in memory
        """
        assert memory_mode in ['unlimited', 'fixed_time_window'
                               ], "Invalide memory mode for EdgeBank!"
        # (TODO Rishi) update for new edge_index usage
        self.memory_mode = memory_mode
        if self.memory_mode == 'fixed_time_window':
            self.time_window_ratio = time_window_ratio
            #determine the time window size based on ratio from the given src, dst, and ts for initialization
            duration = ts.max() - ts.min()
            self.prev_t = ts.min() + duration * (
                1 - time_window_ratio
            )  #the time windows starts from the last ratio% of time
            self.cur_t = ts.max()
            self.duration = self.cur_t - self.prev_t
        else:
            self.time_window_ratio = -1
            self.prev_t = -1
            self.cur_t = -1
            self.duration = -1
        self.memory = None
        self.pos_prob = pos_prob

    def _edge_isin_mem(self, query_edge_indices: torch.tensor) -> torch.tensor:
        r"""
        Parameters:
            query_edge_indices: [2, num_edges] tensor of edge indices
        Returns:
            edge_isin_mem_tensor: [num_edges] boolean tensor representing
                whether each query edge is in memory already
        """
        mem_tensor = self.memory[0]
        # make into decimal index
        decimal_combined_edge_index = 10 * query_edge_indices[
            0, :] + query_edge_indices[1, :]
        decimal_combined_memory = 10 * mem_tensor[0, :] + mem_tensor[1, :]
        edge_isin_mem_tensor = torch.isin(decimal_combined_edge_index,
                                          decimal_combined_memory)

        return edge_isin_mem_tensor

    def _index_mem(self, query_edge_indices: torch.tensor) -> torch.tensor:
        r"""
        return indices in memory that match query_edge_indices
        Parameters:
            query_edge_indices: [2, num_edges] tensor of edge indices
        Returns:
            mem_indices: indices in memory
        """
        mem_tensor = self.memory[0]
        # make into decimal index
        decimal_combined_edge_index = 10 * query_edge_indices[
            0, :] + query_edge_indices[1, :]
        decimal_combined_memory = 10 * mem_tensor[0, :] + mem_tensor[1, :]
        mem_indices = torch.argwhere(
            torch.isin(decimal_combined_memory, decimal_combined_edge_index))
        return mem_indices

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
        if (self.memory_mode == "unlimited"):
            warnings.warn(
                "start_time is not defined for unlimited memory mode, returns -1"
            )
        return self.prev_t

    @property
    def end_time(self) -> int:
        """
        return the end of time window for edgebank `fixed_time_window` only
        Returns:
            end of time window
        """
        if (self.memory_mode == "unlimited"):
            warnings.warn(
                "end_time is not defined for unlimited memory mode, returns -1"
            )
        return self.cur_t

    def _update_unlimited_memory(self, update_edge_index: torch.tensor):
        r"""
        update self.memory with newly arrived edge indices
        Parameters:
            update_edge_index: [2, num_edges] tensor of edge indices
        """
        if self.memory is None:
            self.memory = (update_edge_index,
                           torch.ones(len(update_edge_index)))
            return None
        edge_isin_mem_tensor = self._edge_isin_mem(update_edge_index)
        indices_to_use = torch.argwhere(not edge_isin_mem_tensor)
        edges_to_cat = update_edge_index[:, indices_to_use]
        self.memory[0] = torch.cat((self.memory[0], edges_to_cat))
        ts_to_cat = torch.ones(len(update_edge_index))
        self.memory[1] = torch.cat((self.memory[1], ts_to_cat))

    def _update_time_window_memory(self, update_edge_index: torch.tensor,
                                   update_ts: torch.tensor) -> None:
        r"""
        move the time window forward until end of dst timestamp here
        also need to remove earlier edges from memory which is not in the time window
        Parameters:
            update_edge_index: [2, num_edges] tensor of edge indices
            update_ts: timestamp of the edges
        """
        #* initialize the memory if it is empty

        if self.memory is None:
            self.memory = (update_edge_index, update_ts)
            return None

        #* update the memory if it is not empty
        if (update_ts.max() > self.cur_t):
            self.cur_t = update_ts.max()
            self.prev_t = self.cur_t - self.duration

        #* update existing edges in memory
        mem_indices_to_use = self._index_mem(update_edge_index)
        self.memory[1][mem_indices_to_use] = update_ts

        #* add new edges to the time window
        edge_isin_mem_tensor = self._edge_isin_mem(query_edge_indices)
        indices_to_use = torch.argwhere(not edge_isin_mem_tensor)
        edges_to_cat = update_edge_index[:, indices_to_use]
        self.memory[0] = torch.cat((self.memory[0], edges_to_cat))
        self.memory[1] = torch.cat((self.memory[1], update_ts))

    def predict_link(self, query_edge_indices: torch.tensor) -> torch.tensor:
        r"""
        predict the probability from query src,dst pair given the current memory,
        all edges not in memory will return 0.0 while all observed edges in memory will return self.pos_prob
        Parameters:
            query_edge_indices: [2, num_edges] tensor of edge indices
        Returns:
            pred: the prediction for all query edges
        """
        pred = torch.zeros(query_edge_indices.size(1))
        edge_isin_mem_tensor = self._edge_isin_mem(query_edge_indices)
        edge_indices_to_use = torch.argwhere(edge_isin_mem_tensor)
        if (self.memory_mode == 'fixed_time_window'):
            selected_edges = query_edge_indices[edge_indices_to_use]
            edge_indices_to_use = torch.argwhere(
                self.memory[1][self._index_mem(selected_edges)] >= self.prev_t)
        pred[edge_indices_to_use] = self.pos_prob
        return pred
