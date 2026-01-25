import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device = None, dtype = None):
        super().__init__()
        std = 2 / (in_features + out_features)
        tensor = torch.empty((in_features, out_features), device=device, dtype=dtype)
        self.params = nn.Parameter(
            nn.init.trunc_normal_(tensor, 0, std, -3 * std, 3 * std)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.params)
