import torch


# TODO: refine loader
def get_batch(data, batch_size, seq_len, device):
    n = len(data)
    ix = torch.randint(0, n - seq_len, (batch_size,))

    # Single indexing operation - most efficient
    seq_idx = torch.arange(seq_len, device=device)
    x = torch.from_numpy(data[ix[:, None] + seq_idx]).to(device)
    y = torch.from_numpy(data[ix[:, None] + seq_idx + 1]).to(device)

    return x, y
