import torch
import numpy.typing as npt


def get_batch(data: npt.NDArray, batch_size, seq_len, device):
    n = len(data)
    ix = torch.randint(0, n - seq_len, (batch_size,))

    seq_idx = torch.arange(seq_len)
    x = torch.from_numpy(data[ix[:, None] + seq_idx]).to(device)
    y = torch.from_numpy(data[ix[:, None] + seq_idx + 1]).to(device)

    return x, y.long()
