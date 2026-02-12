import torch
import math
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


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embeddings_dim: int, device = None, dtype = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        std = 2 / (num_embeddings + embeddings_dim)
        tensor = torch.empty((num_embeddings, embeddings_dim), device=device, dtype=dtype)
        self.params = nn.Parameter(
            nn.init.trunc_normal_(tensor, 0, std, -3 * std, 3 * std)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        index = lambda a: self.params[a]
        batched_index = torch.vmap(index)
        return batched_index(x)


class SwiGLUFF(nn.Module):
    def __init__(self, d_model: int, dff: float = 8/3, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        dff = math.ceil((dff * d_model) / 64) * 64
        self.linear1 = Linear(d_model, dff)
        self.linear2 = Linear(d_model, dff)
        self.linear3 = Linear(dff, d_model)


    def _swish(self, x: torch.Tensor) -> torch.Tensor:
        return torch.multiply(x, nn.functional.sigmoid(x))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        swish = self._swish(self.linear1.forward(x))
        activation = self.linear3.forward(x)
        return self.linear2.forward(torch.multiply(activation, swish))


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


class RotaryPositionalEncoing(nn.Module):
    def __init__(self, t: float, d_k: int, max_seq_len: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        rope = torch.zeros(max_seq_len, d_k, 2)
        for i in range(max_seq_len):
            for k in range(int(d_k / 2)):
                k_idx = k+1
                exp = (2 * k_idx - 2) / d_k
                theta = i+1 / (math.pow(t, exp))

                rope[i][k*2][0] = math.cos(theta)
                rope[i][k*2][1] = - math.sin(theta)
                rope[i][k*2+1][0] = math.sin(theta)
                rope[i][k*2+1][1] = math.cos(theta)
        self.register_buffer("rope", torch.transpose(rope, 1, 2))


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        def apply_rope(input: torch.Tensor) -> torch.Tensor:
            indexed_rope = self.get_buffer("rope")[token_positions]
            reshaped_input = torch.reshape(input, (input.shape[0], 1, input.shape[1]))
            return torch.sum(torch.mul(indexed_rope, reshaped_input), 1)

        return torch.vmap(apply_rope)(x)
