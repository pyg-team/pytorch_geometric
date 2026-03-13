import warnings
from typing import Optional

import torch
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention


def get_random_attn_per_block(block_idx: int, start_idx: int, end_idx: int,
                              num_rand_blocks: int, window_left=1,
                              window_right=1, global_left=1, global_right=1):
    r"""Gets random attention for a given block index.

    Args:
        block_idx (int): Current block index
        start_idx (int): Starting block index for current region
        end_idx (int): Ending block index for current region
        num_rand_blocks (int): Number of random blocks to be included
        for current block
        window_left (int): Number of blocks left of current block
        window_right (int): Number of blocks right of current block
        global_left (int): Number of global blocks from left
        global_right (int): Number of global blocks from right

    Returns:
        Tensor: Tensor of shape [num_rand_blocks]
        containing indices of random blocks to attend to for the current block
    """
    all_blocks = torch.arange(start_idx, end_idx)

    # Getting all unpermitted blocks for current block
    unpermitted_blocks = set(
        range(block_idx - window_left, block_idx + window_right + 1))  # window
    unpermitted_blocks |= set(range(global_left))  # start
    unpermitted_blocks |= set(range(end_idx - global_right, end_idx))  # end

    # edge cases
    if block_idx == 1:
        unpermitted_blocks.add(end_idx - 2)
    if block_idx == end_idx - 2:
        unpermitted_blocks.add(1)

    # Collect valid blocks, permute, and select some them as random blocks
    mask = ~torch.isin(all_blocks, torch.tensor(list(unpermitted_blocks)))
    valid_blocks = all_blocks[mask]
    perm = valid_blocks[torch.randperm(len(valid_blocks))]
    return perm[:num_rand_blocks]


def build_rand_attention(seq_length: int, block_size: int, num_heads: int,
                         expected_q_block_lengths: list[int],
                         expected_num_rand_blocks: list[int],
                         window_left: int = 1, window_right: int = 1,
                         global_top: int = 1, global_bottom: int = 1,
                         global_left: int = 1, global_right: int = 1):
    """Builds random attention for bigbird attention with head dimension.

    Args:
        seq_length (int): Sequence length of input
        block_size (int): Block size in query sequence
        num_heads (int): Number of attention heads
        expected_q_block_lengths (List[int]): List of end indices of regions
                                of a sequence
        expected_num_rand_blocks (List[int]): List of number of random blocks
                                to be included in each of the regions
        window_left (int): Number of blocks to the left of current block
        window_right (int): Number of blocks to the right of current block
        global_top (int): Number of global blocks from the top of the sequence
        global_bottom (int): Number of global blocks from bottom of sequence
        global_left (int): Number of global blocks from left of the sequence
        global_right (int): Number of global blocks from right of the sequence

    Returns:
        Tensor: Random attention of shape
        [num_attention_heads, seq_length//block_size-2, num_rand_blocks]
    """
    n_blocks = seq_length // block_size
    blocks_per_region = torch.tensor(expected_q_block_lengths) // block_size
    max_idx = expected_q_block_lengths.index(seq_length)

    rand_size = sum(expected_num_rand_blocks[:max_idx + 1])
    rand_attn = [torch.zeros(n_blocks, rand_size) for _ in range(num_heads)]

    for exp_idx in range(max_idx + 1):
        rnd_r_cnt = 0
        if exp_idx > 0:
            if expected_num_rand_blocks[exp_idx] > 0:
                rnd_r_cnt = sum(expected_num_rand_blocks[:exp_idx])
                curr_r_cnt = sum(expected_num_rand_blocks[:exp_idx + 1])
                for block_row_idx in range(global_top,
                                           blocks_per_region[exp_idx - 1]):
                    for head in range(num_heads):
                        rand_attn[head][
                            block_row_idx,
                            rnd_r_cnt:curr_r_cnt] = get_random_attn_per_block(
                                block_idx=block_row_idx,
                                start_idx=blocks_per_region[exp_idx - 1],
                                end_idx=blocks_per_region[exp_idx],
                                num_rand_blocks=expected_num_rand_blocks[
                                    exp_idx], window_left=window_left,
                                window_right=window_right,
                                global_left=global_left,
                                global_right=global_right)

            for idx in range(exp_idx):
                if expected_num_rand_blocks[idx] == 0:
                    continue
                for block_row_idx in range(blocks_per_region[exp_idx - 1],
                                           blocks_per_region[exp_idx]):
                    rnd_r_cnt = 0
                    start_idx = 0
                    if idx > 0:
                        rnd_r_cnt = sum(expected_num_rand_blocks[:idx])
                        start_idx = blocks_per_region[idx - 1]
                    curr_r_cnt = sum(expected_num_rand_blocks[:idx + 1])

                    for head in range(num_heads):
                        rand_attn[head][
                            block_row_idx,
                            rnd_r_cnt:curr_r_cnt] = get_random_attn_per_block(
                                block_idx=block_row_idx, start_idx=start_idx,
                                end_idx=blocks_per_region[idx],
                                num_rand_blocks=expected_num_rand_blocks[idx],
                                window_left=window_left,
                                window_right=window_right,
                                global_left=global_left,
                                global_right=global_right)

        if expected_num_rand_blocks[exp_idx] == 0:
            continue
        curr_r_cnt = sum(expected_num_rand_blocks[:exp_idx + 1])
        q_start_id = global_top
        kv_start_id = 0

        if exp_idx > 0:
            rnd_r_cnt = sum(expected_num_rand_blocks[:exp_idx])
            q_start_id = blocks_per_region[exp_idx - 1]
            kv_start_id = blocks_per_region[exp_idx - 1]

        for block_row_idx in range(q_start_id, blocks_per_region[exp_idx]):
            for head in range(num_heads):
                rand_attn[head][
                    block_row_idx,
                    rnd_r_cnt:curr_r_cnt] = get_random_attn_per_block(
                        block_idx=block_row_idx, start_idx=kv_start_id,
                        end_idx=blocks_per_region[exp_idx],
                        num_rand_blocks=expected_num_rand_blocks[exp_idx],
                        window_left=window_left, window_right=window_right,
                        global_left=global_left, global_right=global_right)

    for h in range(num_heads):
        rand_attn[h] = rand_attn[h][global_top:n_blocks - global_bottom, :]

    return torch.stack(rand_attn, dim=0)


def get_random_attn_zones(q_seq_length, q_block_size, num_rand_blocks):
    r"""Gets the regions of the sequence to insert random attention.

    Args:
    q_seq_length (int): length of query sequence.
    q_block_size (int): size of block in query sequence.
    num_rand_blocks (int): Number of random chunks per row.

    Returns:
    expected_q_block_lengths,expected_num_rand_blocks (Tuple[List,List]):
    expected_q_block_lengths: list of end indices of regions
                                of a sequence
    expected_num_rand_blocks: list of number of random blocks to be included
                               in each of the regions
    """
    n_q_blocks = q_seq_length // q_block_size

    if (2 * num_rand_blocks + 5) < n_q_blocks:
        return ([int((2 * num_rand_blocks + 5) * q_block_size),
                 q_seq_length], [num_rand_blocks, 0])

    elif (num_rand_blocks + 5) < n_q_blocks:
        return ([int((num_rand_blocks + 5) * q_block_size), q_seq_length], [
            num_rand_blocks // 2, num_rand_blocks - (num_rand_blocks // 2)
        ])

    return ([q_seq_length], [num_rand_blocks])


def create_band_mask(q_block_mask, kv_block_mask):
    r"""Create band mask from input.

    Args:
        q_block_mask (Tensor): Mask for query blocks
            [batch_size, q_seq_length//q_block_size, from_block_size]
        kv_block_mask (Tensor): Mask for key/value blocks
            [batch_size, kv_seq_length//kv_block_size, kv_block_size]

    Returns:
        band_mask (Tensor): Mask for band attention
            [batch_size, 1, q_seq_length//q_block_size-4, q_block_size,
             3*kv_block_size]
    """
    expanded_kv_pad = torch.cat(
        (kv_block_mask[:, 1:-3], kv_block_mask[:, 2:-2], kv_block_mask[:,
                                                                       3:-1]),
        dim=2)

    band_mask = torch.einsum('blq,blk->blqk', q_block_mask[:, 2:-2],
                             expanded_kv_pad)
    band_mask.unsqueeze_(1)
    return band_mask


def create_random_mask(q_block_mask: Tensor, kv_block_mask: Tensor,
                       random_attn: Tensor, q_block_size: int):
    r"""Creates bigbird sparse random mask from input.

    Args:
        q_block_mask (Tensor): Mask for query blocks
            [batch_size, q_seq_length//q_block_size, from_block_size]
        kv_block_mask (Tensor): Mask for key/value blocks
            [batch_size, kv_seq_length//kv_block_size, kv_block_size]
        random_attn (Tensor): Tensor containing indices of random blocks
                              to attend to for each query block
            [batch_size, num_attention_heads,
            q_seq_length//q_block_size-2, num_rand_blocks]
        q_block_size (int): Block size in query sequence

    Returns:
        random_mask (Tensor): Mask for random attention
            [batch_size, num_attention_heads,
            q_seq_length//q_block_size-2, num_rand_blocks*kv_block_size]
    """
    B, H, n_windows, R = random_attn.size()

    batch_idx = torch.arange(B).view(B, 1, 1, 1)

    random_mask = torch.reshape(kv_block_mask[batch_idx, random_attn],
                                (B, H, n_windows, R * q_block_size))
    random_mask = torch.einsum('blq,bhlk->bhlqk', q_block_mask[:, 1:-1],
                               random_mask)

    return random_mask


def bigbird_sparse_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    q_mask: Tensor,
    kv_mask: Tensor,
    band_mask: Tensor,
    q_block_mask: Tensor,
    kv_block_mask: Tensor,
    random_attn: Tensor,
    q_block_size: int,
    kv_block_size: int,
) -> Tensor:
    r"""Implemetation of BigBird sparse attention as described
    in the paper "Big Bird: Transformers for Longer Sequences"
    (https://arxiv.org/abs/2007.14062).

    Args:
        query (Tensor): Query tensor of shape
            [batch_size, q_seq_length, dim]
        key (Tensor): Key tensor of shape
            [batch_size, kv_seq_length, dim]
        value (Tensor): Value tensor of shape
            [batch_size, kv_seq_length, dim]
        q_mask (Tensor): Mask for query sequence of shape
            [batch_size, 1, q_seq_length, 1]
        kv_mask (Tensor): Mask for key/value sequence of shape
            [batch_size, 1, 1, kv_seq_length]
        band_mask (Tensor): Mask for band attention of shape
            [batch_size, 1, q_seq_length//q_block_size-4,
            q_block_size, 3*kv_block_size]
        q_block_mask (Tensor): Mask for query blocks of shape
            [batch_size, q_seq_length//q_block_size, q_block_size]
        kv_block_mask (Tensor): Mask for key/value blocks of shape
            [batch_size, kv_seq_length//kv_block_size, kv_block_size]
        random_attn (Tensor): Tensor containing indices of random blocks
            to attend to for each query blockof shape
            [batch_size, num_attention_heads,
                q_seq_length//q_block_size-2, num_rand_blocks]
        q_block_size (int): Block size in query sequence
        kv_block_size (int): Block size in key/value sequence

    """
    q_seq_length, kv_seq_length = query.size(2), key.size(2)
    head_dim = query.size(3)

    assert q_seq_length // q_block_size == kv_seq_length // kv_block_size

    batch_size = query.size(0)

    # Add batch dimension to random attention
    random_attn = random_attn.unsqueeze(0).expand(batch_size, -1, -1,
                                                  -1).long()

    B, n_attn_heads, n_windows, n_rand_blocks = random_attn.size()

    # Create random mask
    random_mask = create_random_mask(q_block_mask, kv_block_mask, random_attn,
                                     q_block_size)

    # Getting blocked versions of query,key, and value matrices
    blocked_query = torch.reshape(query,
                                  (batch_size, n_attn_heads, q_seq_length //
                                   q_block_size, q_block_size, head_dim))
    blocked_key = torch.reshape(key,
                                (batch_size, n_attn_heads, kv_seq_length //
                                 kv_block_size, kv_block_size, head_dim))
    blocked_value = torch.reshape(value,
                                  (batch_size, n_attn_heads, kv_seq_length //
                                   kv_block_size, kv_block_size, head_dim))

    batch_idx = torch.arange(batch_size).view(batch_size, 1, 1, 1).long()
    head_idx = torch.arange(n_attn_heads).view(1, n_attn_heads, 1, 1).long()

    # Obtaining sparse key and value tensors corresponding to random attention
    sparse_key = torch.reshape(blocked_key[batch_idx, head_idx, random_attn],
                               (batch_size, n_attn_heads, n_windows,
                                n_rand_blocks * kv_block_size, head_dim))
    sparse_value = torch.reshape(
        blocked_value[batch_idx, head_idx, random_attn],
        (batch_size, n_attn_heads, n_windows, n_rand_blocks * kv_block_size,
         head_dim))

    # first query block attends to all key blocks.
    product_1 = torch.einsum('bhqd,bhkd->bhqk', blocked_query[:, :, 0], key)
    product_1 = product_1 / (head_dim**0.5)
    product_1.masked_fill_(kv_mask == 0, float('-inf'))
    attn_weights_1 = torch.nn.functional.softmax(product_1, dim=-1)
    context_layer_1 = torch.einsum('bhqk,bhkd->bhqd', attn_weights_1, value)
    context_layer_1 = context_layer_1.unsqueeze(2)

    # second query block attends to first and last key blocks (global),
    # second and third key blocks (local),
    # and rand_blocks (random)
    second_key = torch.cat(
        (blocked_key[:, :, 0], blocked_key[:, :, 1], blocked_key[:, :, 2],
         blocked_key[:, :, -1], sparse_key[:, :, 0]), dim=2)
    second_value = torch.cat(
        (blocked_value[:, :, 0], blocked_value[:, :, 1], blocked_value[:, :,
                                                                       2],
         blocked_value[:, :, -1], sparse_value[:, :, 0]), dim=2)
    product_2 = torch.einsum('bhqd,bhkd->bhqk', blocked_query[:, :, 1],
                             second_key)
    second_seq_pad = torch.cat(
        (kv_mask[:, :, :, :3 * kv_block_size], kv_mask[:, :, :,
                                                       -kv_block_size:],
         torch.ones_like(random_mask[:, :1, 0, :1])), dim=3)
    second_rand_pad = torch.cat((torch.ones_like(
        product_2[:, :, :, :4 * kv_block_size]), random_mask[:, :, 0]), dim=3)

    product_2 = product_2 / (head_dim**0.5)
    product_2.masked_fill_(
        torch.min(second_seq_pad, second_rand_pad) == 0, float('-inf'))
    attn_weights_2 = torch.nn.functional.softmax(product_2, dim=-1)
    context_layer_2 = torch.einsum('bhqk,bhkd->bhqd', attn_weights_2,
                                   second_value)
    context_layer_2 = context_layer_2.unsqueeze(2)

    # middle query blocks
    expanded_block_key = torch.cat(
        (blocked_key[:, :, 1:-3], blocked_key[:, :, 2:-2], blocked_key[:, :,
                                                                       3:-1]),
        dim=3)
    expanded_block_value = torch.cat(
        (blocked_value[:, :, 1:-3], blocked_value[:, :, 2:-2],
         blocked_value[:, :, 3:-1]), dim=3)

    middle_query = blocked_query[:, :, 2:-2]

    inner_product = torch.einsum('bhlqd,bhlkd->bhlqk', middle_query,
                                 expanded_block_key)
    inner_product = inner_product / (head_dim**0.5)
    rand_product = torch.einsum('bhlqd,bhlkd->bhlqk', middle_query,
                                sparse_key[:, :, 1:-1])
    rand_product = rand_product / (head_dim**0.5)

    first_outer_product = torch.einsum('bhlqd,bhkd->bhlqk', middle_query,
                                       blocked_key[:, :, 0])
    first_outer_product = first_outer_product / (head_dim**0.5)
    last_outer_product = torch.einsum('bhlqd,bhkd->bhlqk', middle_query,
                                      blocked_key[:, :, -1])
    last_outer_product = last_outer_product / (head_dim**0.5)

    inner_product.masked_fill_(band_mask == 0, float('-inf'))
    first_outer_product.masked_fill_(
        kv_mask[:, :, :, :kv_block_size].unsqueeze(3) == 0, float('-inf'))
    last_outer_product.masked_fill_(
        kv_mask[:, :, :, -kv_block_size:].unsqueeze(3) == 0, float('-inf'))
    rand_product.masked_fill_(random_mask[:, :, 1:-1] == 0, float('-inf'))

    band_product = torch.cat(
        (first_outer_product, inner_product, last_outer_product, rand_product),
        dim=-1)
    attn_weights_mid = torch.nn.functional.softmax(band_product, dim=-1)
    context_layer_middle = torch.einsum(
        'bhlqk,bhlkd->bhlqd',
        attn_weights_mid[:, :, :, :, kv_block_size:4 * kv_block_size],
        expanded_block_value)
    context_layer_middle += torch.einsum(
        'bhlqk,bhlkd->bhlqd',
        attn_weights_mid[:, :, :, :, 4 * kv_block_size:-kv_block_size],
        sparse_value[:, :, 1:-1])
    context_layer_middle += torch.einsum(
        'bhlqk,bhkd->bhlqd', attn_weights_mid[:, :, :, :, :kv_block_size],
        blocked_value[:, :, 0])
    context_layer_middle += torch.einsum(
        'bhlqk,bhkd->bhlqd', attn_weights_mid[:, :, :, :, -kv_block_size:],
        blocked_value[:, :, -1])

    # second last query block
    second_last_key = torch.cat(
        (blocked_key[:, :, 0], blocked_key[:, :, -3], blocked_key[:, :, -2],
         blocked_key[:, :, -1], sparse_key[:, :, -1]), dim=2)
    second_last_value = torch.cat(
        (blocked_value[:, :, 0], blocked_value[:, :, -3], blocked_value[:, :,
                                                                        -2],
         blocked_value[:, :, -1], sparse_value[:, :, -1]), dim=2)

    product_2nd_last = torch.einsum('bhqd,bhkd->bhqk', blocked_query[:, :, -2],
                                    second_last_key)
    second_last_seq_pad = torch.cat(
        (kv_mask[:, :, :, :kv_block_size], kv_mask[:, :, :,
                                                   -3 * kv_block_size:],
         torch.ones_like(random_mask[:, :1, 0, :1])), dim=3)
    second_last_rand_pad = torch.cat((torch.ones_like(
        product_2nd_last[:, :, :, :4 * kv_block_size]), random_mask[:, :, -1]),
                                     3)

    product_2nd_last = product_2nd_last / (head_dim**0.5)
    product_2nd_last.masked_fill_(
        torch.minimum(second_last_seq_pad, second_last_rand_pad) == 0,
        float('-inf'))
    attn_weights_2nd_last = torch.nn.functional.softmax(
        product_2nd_last, dim=-1)
    context_layer_2nd_last = torch.einsum('bhqk,bhkd->bhqd',
                                          attn_weights_2nd_last,
                                          second_last_value)
    context_layer_2nd_last = context_layer_2nd_last.unsqueeze(2)

    # last query block
    product_last = torch.einsum('bhqd,bhkd->bhqk', blocked_query[:, :, -1],
                                key)
    product_last = product_last / (head_dim**0.5)
    product_last.masked_fill_(kv_mask == 0, float('-inf'))
    attn_weights_last = torch.nn.functional.softmax(product_last, dim=-1)
    context_layer_last = torch.einsum('bhqk,bhkd->bhqd', attn_weights_last,
                                      value)
    context_layer_last = context_layer_last.unsqueeze(2)

    context_layer = torch.cat(
        (context_layer_1, context_layer_2, context_layer_middle,
         context_layer_2nd_last, context_layer_last), dim=2)
    context_layer = torch.reshape(
        context_layer,
        (batch_size, n_attn_heads, q_seq_length, head_dim)) * q_mask
    return context_layer


class BigBirdAttention(torch.nn.Module):
    def __init__(self, channels: int, n_heads: int, head_dim: int = 64,
                 num_rand_blocks: int = 3, block_size: int = 64,
                 attn_out_bias: bool = True, qkv_bias: bool = False,
                 dropout: float = 0.0):
        r"""Multihead attention Class implementing BigBird sparse attention.

        Args:
            channels (int): Size of each input sample.
            n_heads (int, optional): Number of parallel attention heads.
            head_dim (int, optional): Size of each attention head.
                (default: :obj:`64.`)
            num_rand_blocks (int, optional): Number of random blocks to be
                included. (default: :obj:`3`)
            block_size (int, optional): Block size to be used in attention.
                (default: :obj:`64`)
            qkv_bias (bool, optional): If specified, add bias to query, key
                and value in the self attention. (default: :obj:`False`)
            attn_out_bias (bool, optional): If specified, add bias to the
                attention output. (default: :obj:`True`)
            dropout (float, optional): Dropout probability of the final
                attention output. (default: :obj:`0.0`)
        """
        super().__init__()

        self.n_heads = n_heads
        self.num_rand_blocks = num_rand_blocks
        self.block_size = block_size
        self.head_dim = head_dim

        inner_dim = n_heads * head_dim
        self.q = torch.nn.Linear(channels, inner_dim, bias=qkv_bias)
        self.k = torch.nn.Linear(channels, inner_dim, bias=qkv_bias)
        self.v = torch.nn.Linear(channels, inner_dim, bias=qkv_bias)
        self.attn_out = torch.nn.Linear(inner_dim, channels,
                                        bias=attn_out_bias)

        self.dropout = torch.nn.Dropout(dropout)
        self.is_sparse = True
        self.pad_len = 0

    def forward(self, x, mask: Optional[Tensor] = None):
        B, N, _ = x.shape
        # print("Sequence Length: ", N)
        is_sparse = True

        if N <= (2 * self.num_rand_blocks + 5) * self.block_size:
            # If sequence length is too small, perform dense attention
            warnings.warn(
                f"""Sequence length does not satisfy the condition:
                    seq_length > (5 + 2*num_rand_blocks) * block_size.
                    Got seq_length={N},
                    num_rand_blocks={self.num_rand_blocks},
                    block_size={self.block_size}.
                    Falling back to dense attention""", stacklevel=2)

            is_sparse = False

        else:
            is_sparse = True
            # Pad if sequence length is not a multiple of block size
            if N % self.block_size != 0:
                warnings.warn(
                    """Padding sequence to make it a
                                 multiple of block size.""", stacklevel=2)

                self.pad_len = ((self.block_size - (N % self.block_size)) %
                                self.block_size)
                x = torch.nn.functional.pad(x, (0, 0, 0, self.pad_len),
                                            value=0)
                mask = torch.nn.functional.pad(mask, (0, self.pad_len),
                                               value=False)
                N = N + self.pad_len

            if mask is not None:
                q_mask = mask[:, None, :, None]
                kv_mask = mask[:, None, None, :]

                q_block_mask = torch.reshape(
                    mask, (B, N // self.block_size, self.block_size))
                kv_block_mask = torch.reshape(
                    mask, (B, N // self.block_size, self.block_size))
                band_mask = create_band_mask(q_block_mask, kv_block_mask)

                exp_block_lengths, exp_n_rand_blocks = get_random_attn_zones(
                    N, self.block_size, self.num_rand_blocks)
                random_attn = build_rand_attention(
                    seq_length=N, block_size=self.block_size,
                    num_heads=self.n_heads,
                    expected_q_block_lengths=exp_block_lengths,
                    expected_num_rand_blocks=exp_n_rand_blocks)

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q, k, v = map(
            lambda t: t.reshape(B, N, self.n_heads, self.head_dim).permute(
                0, 2, 1, 3), (q, k, v))

        if is_sparse:
            # print("Using sparse")
            out = bigbird_sparse_attention(
                query=q, key=k, value=v, q_mask=q_mask, kv_mask=kv_mask,
                q_block_mask=q_block_mask, kv_block_mask=kv_block_mask,
                band_mask=band_mask, random_attn=random_attn,
                q_block_size=self.block_size, kv_block_size=self.block_size)
        else:
            # print("Using Dense")
            attn_mask = mask[:, None, None, :].bool()
            out = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        out = out.reshape(B, N, -1)

        # Unpad padded tokens
        if self.pad_len > 0:
            out = out[:, :N - self.pad_len, :]

        out = self.attn_out(out)
        out = self.dropout(out)
        return out

    def _reset_parameters(self):
        self.q.reset_parameters()
        self.k.reset_parameters()
        self.v.reset_parameters()
        self.attn_out.reset_parameters()

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'heads={self.n_heads}, '
                f'head_dim={self.head_dim}, '
                f'block_size={self.block_size}, '
                f'num_rand_blocks={self.num_rand_blocks})')
