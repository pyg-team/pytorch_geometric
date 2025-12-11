import argparse
import gc
import os
import os.path as osp
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch_geometric.nn import GNCAConv

# optional dependencies
try:
    from pygsp.graphs import Bunny
    import numpy as np
except ImportError:
    Bunny = None
    print("Please install 'pygsp' to run this example: pip install pygsp")

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    plt = None

class GNCAModel(nn.Module):
    r"""The full Graph Neural Cellular Automata model architecture.

    This model implements the three-part update rule described in 
    Appendix A.2 of the paper

    Args:
        input_channels (int): Dimensionality of the input state space
        output_channels (int): Dimensionality of the output state space.
        hidden_channels (int, optional): Number of hidden units in the MLPs 
             and convolution layer. (default: :obj:`256`)
        output_activation (str, optional): The activation to apply to 
            apply to the final output.
            Options: :obj:`"tanh"` (continuous bounded states), 
            :obj:`"sigmoid"` 
            (for binary states), or :obj:`None`. (default: :obj:`"tanh"`)
    """
    def __init__(self, 
                 input_channels: int, 
                 output_channels: int,
                 edge_dim: int = 0,
                 hidden_channels: int = 256,
                 output_activation: Optional[str] = 'tanh'):
        super().__init__()

        # 1. Pre-processing MLP
        # Linear -> ReLU -> Linear structure
        self.mlp_pre = nn.Sequential(
            nn.Linear(input_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # 2. Message Passing Layer (The GNCAConv)
        # Input size is hidden_channels (from mlp_pre).
        # Output size of the convolution message is hidden_channels.
        self.conv = GNCAConv(hidden_channels, hidden_channels,
                             edge_dim=edge_dim)
        
        # 3. Post-processing MLP
        # Input  2 * hidden_channels because GNCAConv concats 
        # [self || neighbor_agg]
        self.mlp_post = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, output_channels)
        )

        self.output_activation = output_activation

    def forward(self, x, edge_index, 
                edge_attr: Optional[torch.Tensor] = None):
        r"""Calculates the update delta.

        Args:
            x (Tensor): Node features.
            edge_index (LongTensor): Edge connectivity.
            edge_attr (Tensor, optional): Edge features.
        """
        # 1. Pre-process current state into hidden representation
        h = self.mlp_pre(x)

        # 2. Evolve hidden representation using GNCAConv
        # Returns shape [N, 2 * hidden_channels]
        h = self.conv(h, edge_index, edge_attr=edge_attr)

        # 3. Post-process to get the update/next state
        out = self.mlp_post(h)

        # 4. Apply specific output activation based on the task
        if self.output_activation == 'tanh':
            return torch.tanh(out)
        elif self.output_activation == 'sigmoid':
            return torch.sigmoid(out)
        else:
            return out


class StateCache:
    r"""Stores states of GNCA model for replay memory.
    Lets the model learn from previous states.
    """
    def __init__(self, size, initial_state):
        """
        Args:
            size (int): The number of states to keep in the cache.
            initial_state (torch.Tensor): The seed state (e.g., sphere).
        """
        self.size = size
        # Store initial_state that we can use for resets
        # Ensure we clone it so we don't change the original reference!
        self.init_state = initial_state
        self.cache = [initial_state.clone() for _ in range(size)] 
        
    def sample(self, count):
        """
        Samples a random batch of states from the cache.
        
        Args:
            count (int): Batch size to sample.
            
        Returns:
            idxs (torch.Tensor): The indices of the chosen states.
            batch (torch.Tensor): The stacked states of the chosen indices.
        """
        # Randomly select indices
        idxs = torch.randint(0, len(self.cache), (count,))
        batch = [self.cache[i] for i in idxs]
        
        # Stack them into one tensor for the model
        return idxs, torch.stack(batch)

    def update(self, idxs, new_states):
        """
        Updates the cache with the newly evolved states.
        
        Args:
            idxs (torch.Tensor): The indices of the states to update.
            new_states (torch.Tensor): The new states output by the model.
        """
        # Detach to stop gradient from growing forever
        for i, idx in enumerate(idxs):
            self.cache[idx] = new_states[i].detach().clone()
            
        # Replace one random sample with the initial seed
        # prevents model from "drifting too far" and 
        # forgetting how to grow from beginning
        replace_idx = torch.randint(0, len(self.cache), (1,)).item()
        self.cache[replace_idx] = self.init_state.detach().clone()

'''
BELOW:
DATA LOADING > PREPROCESSING > MODEL > TRAINING > VISUALIZATION 
'''

def get_bunny_data(device) -> Tuple[torch.Tensor, torch.Tensor]:
    graph = Bunny()
    pos = torch.tensor(graph.coords, dtype=torch.float)
    
    # 1. Center everything around the mean
    mean = pos.mean(dim=0)
    pos = pos - mean
    
    # 2. Scale by max norm to make the points fit within a unit sphere
    max_norm = pos.norm(p=2, dim=1).max()
    pos = pos / max_norm
    
    # 3. Get edges
    rows, cols = graph.W.nonzero()
    edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)
    
    return pos, edge_index

def sphericalize_state(target_pos):
    """Init points as a hollow sphere seed state."""
    norms = target_pos.norm(p=2, dim=1, keepdim=True)
    norms = torch.where(norms == 0, torch.ones_like(norms), norms)
    return target_pos / norms


def train():
    # keeping memory clean
    gc.collect()
    torch.cuda.empty_cache()
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Note: paper uses batch size 8, but to reduce memory, we're using 1
    # Then we accumulate gradients over 8 steps
    TARGET_BATCH_SIZE = 8
    PHYSICAL_BATCH_SIZE = 1
    ACCUM_STEPS = TARGET_BATCH_SIZE // PHYSICAL_BATCH_SIZE
    
    # training hyperparameters (seem to work well)
    # increasing learning rate makes things unstable
    LR = 0.001
    POOL_SIZE = 1024
    N_EPOCHS = 2000
    
    # data setup
    target_pos, edge_index = get_bunny_data() # final 3D target shape
    target_pos = target_pos.to(device)
    edge_index = edge_index.to(device)
    num_nodes = target_pos.size(0)
    # init seed state (normed) $\bar{s}_i = \hat{s}_i / ||\hat{s}_i||$
    seed_state = sphericalize_state(target_pos)

    # batched edge index
    edge_indices_list = []
    for i in range(PHYSICAL_BATCH_SIZE):
        edge_indices_list.append(edge_index + i * num_nodes)
    batched_edge_index = torch.cat(edge_indices_list, dim=1).to(device)

    # setup model (hidden=128 to reduce memory)
    model = GNCAModel(input_channels=3, output_channels=3, 
                        hidden_channels=128, 
                        output_activation='tanh').to(device)
    
    # Zero-init last layer helps training stability
    # Forces the model to start as "Identity" (Delta=0)
    # The rest of the network uses Xavier init
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
                
    with torch.no_grad():
        model.mlp_post[-1].weight.fill_(0.0)
        model.mlp_post[-1].bias.fill_(0.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # sample pool to store previous states to prevent catastrophic forgetting
    # this is the "cache" from the paper
    pool = StateCache(POOL_SIZE, seed_state) 
    
    print(f"STARTING TRAINING ON {device}", flush=True)
    
    optimizer.zero_grad()
    
    for epoch in range(N_EPOCHS):
        epoch_loss = 0
        
        # Accumulate Gradients
        for _ in range(ACCUM_STEPS):
            batch_idx, current_states = pool.sample(PHYSICAL_BATCH_SIZE)
            current_states = current_states.to(device)
            # at least one sample in every batch is the original starting seed
            # (so it doesn't forget how to grow from beginning)
            current_states[0] = seed_state.clone() 
            
            steps = torch.randint(10, 21, (1,)).item()
            x = current_states
            
            # unroll the model for a random number of steps between 10 and 20
            # this is the "growth" part of the model
            for _ in range(steps):
                x_flat = x.view(-1, 3)
                delta_flat = model(x_flat, batched_edge_index) # prediction
                delta = delta_flat.view(PHYSICAL_BATCH_SIZE, num_nodes, 3)
                x = x + delta
                x = torch.clamp(x, -1.0, 1.0)
                
            target_batch = target_pos.unsqueeze(0).expand(args.batch_size,
                                                          -1, -1)
            loss = F.mse_loss(x, target_batch) / args.accum_steps
            loss.backward() # backpropagate the loss through the model (over all steps)
            
            epoch_loss += loss.item()
            pool.update(batch_idx, x.detach())

        # Optimizer Step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        # Logging checkpoints
        if epoch % 50 == 0 or epoch < 10:
            print(f"Epoch {epoch} | Loss: {epoch_loss:.6f}", flush=True)
            
            if epoch % 50 == 0:
                os.makedirs('checkpoints', exist_ok=True)
                torch.save(model.state_dict(), 'checkpoints/bunny_gnca.pt')
                
            if epoch % 100 == 0:
                torch.cuda.empty_cache()

    print("TRAINING DONE SUCCESSFULLY")


def visualize():
    if plt is None:
        print("Matplotlib not installed.")
        return

    device = torch.device('cpu')
    if not osp.exists('checkpoints/bunny_gnca.pt'):
        print("Checkpoint not found! Run with --train first.")
        return
        
    target_pos, edge_index = get_bunny_data(device)
    
    # normalize target exactly as in the training
    target_mean = target_pos.mean(dim=0)
    target_scale = target_pos.abs().max()
    
    # Initial seed (collapsed state)
    # Very close to zero but slightly biased in the direction of pos
    current_state = ((target_pos - target_mean) / target_scale) * 0.01
    
    model = GNCAModel(input_channels=3, output_channels=3, 
                        output_activation='tanh')
    model.load_state_dict(torch.load('checkpoints/bunny_gnca.pt', 
                                    map_location=device))
    model.eval()
    
    # Setup Plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # update function for the animation
    def update(frame):
        nonlocal current_state
        ax.clear()
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.set_title(f"GNCA Growth - Step {frame}")
        
        np_state = current_state.numpy()
        ax.scatter(np_state[:, 0], np_state[:, 1], np_state[:, 2], 
                   c=np_state[:, 2], cmap='viridis', s=10) # Color by Z-height
        
        # Evolve
        delta = model(current_state, edge_index) # unrolling
        current_state = current_state + delta
        current_state = torch.clamp(current_state, -1.0, 1.0)
        
        return ax,
    
    ani = animation.FuncAnimation(fig, update, frames=60, interval=100)
    ani.save('bunny_growth.gif', writer='pillow', fps=10)
    print("Saved gif to bunny_growth.gif")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--visualize', action='store_true', 
                        help='Run visualization')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--pool_size', type=int, default=1024)
    parser.add_argument('--accum_steps', type=int, default=8)
    
    args = parser.parse_args()

    if args.train:
        train(args)
    elif args.visualize:
        visualize()
    else:
        print("Please choose either --train or --visualize")