import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.params = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        squared = torch.square(x.to(torch.float32))
        rms = torch.sqrt(torch.mean(torch.add(self.eps, squared)))
        return torch.mul(torch.div(x, rms), self.params).to(in_type)
