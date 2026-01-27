import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embeddings_dim: int, device = None, dtype = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        std = 2 / (num_embeddings + embeddings_dim)
        tensor = torch.empty((num_embeddings, embeddings_dim), device=device, dtype=dtype).long()
        self.params = nn.Parameter(
            nn.init.trunc_normal_(tensor, 0, std, -3 * std, 3 * std)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.index_select(self.params, 0, x)
