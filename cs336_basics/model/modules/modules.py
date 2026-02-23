import torch
import torch.nn as nn

from cs336_basics.model.modules import func


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        std = 2 / (in_features + out_features)
        tensor = torch.empty((in_features, out_features), device=device, dtype=dtype)
        self.params = nn.Parameter(nn.init.trunc_normal_(tensor, 0, std, -3 * std, 3 * std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.params)


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embeddings_dim: int, device=None, dtype=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        std = 2 / (num_embeddings + embeddings_dim)
        tensor = torch.empty((num_embeddings, embeddings_dim), device=device, dtype=dtype)
        self.params = nn.Parameter(nn.init.trunc_normal_(tensor, 0, std, -3 * std, 3 * std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.params[x]


class SwiGLUFF(nn.Module):
    def __init__(self, d_model: int, dff: float = 8 / 3, device=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # dff = math.ceil((dff * d_model) / 64) * 64
        self.linear1 = Linear(d_model, dff, device=device)
        self.linear2 = Linear(dff, d_model, device=device)
        self.linear3 = Linear(d_model, dff, device=device)

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
        in_device = x.device
        squared = torch.square(x.to(in_device, torch.float32))
        rms = torch.sqrt(torch.mean(squared, dim=-1, keepdim=True) + self.eps)
        return torch.mul(torch.div(x, rms), self.params).to(in_type)


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, t: float, d_k: int, max_seq_len: int, device=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        i = torch.arange(0, max_seq_len, dtype=torch.int32, device=device).reshape((max_seq_len, 1))
        k = torch.arange(0, d_k / 2, dtype=torch.float32, device=device)
        k_idx = torch.stack((k, k), dim=1).flatten()

        base = torch.pow(t, (-2 * k_idx) / d_k)
        thetas = base.expand((max_seq_len, d_k)) * i

        cos_t = torch.cos(thetas)
        sin_t = torch.sin(thetas)

        neg_mask = torch.ones(sin_t.shape[1], device=device)
        neg_idx = torch.t(torch.arange(0, d_k, 2, device=device))
        neg_mask[neg_idx] *= -1

        sin_t = sin_t * neg_mask.expand(sin_t.shape)

        cos_idx = torch.arange(0, d_k, dtype=torch.int64, device=device)

        sin_half_even = cos_idx[0 : d_k - 1 : 2]
        sin_half_odd = cos_idx[1:d_k:2]
        sin_idx = torch.stack((sin_half_odd, sin_half_even), dim=1).flatten()

        self.register_buffer("cos_t", cos_t)
        self.register_buffer("sin_t", sin_t)
        self.register_buffer("sin_idx", sin_idx)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos_t = self.get_buffer("cos_t")[token_positions]
        sin_t = self.get_buffer("sin_t")[token_positions]
        sin_idx = self.get_buffer("sin_idx")
        sin_idx_exp = sin_idx.expand(x.shape)

        return x * cos_t + torch.gather(x, -1, sin_idx_exp) * sin_t


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float = 10000,
        device=None,
        dtype=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.dk = d_model // num_heads
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.device = device
        self.rope = RotaryPositionalEncoding(theta, self.dk, max_seq_len, device=device)
        std = 2 / (self.dk + d_model)

        wq_tensor = torch.empty((num_heads * self.dk, d_model), device=device, dtype=dtype)
        wk_tensor = torch.empty((num_heads * self.dk, d_model), device=device, dtype=dtype)
        wv_tensor = torch.empty((num_heads * self.dk, d_model), device=device, dtype=dtype)
        wo_tensor = torch.empty((d_model, num_heads * self.dk), device=device, dtype=dtype)

        self.wq = torch.nn.Parameter(nn.init.trunc_normal_(wq_tensor, 0, std, -3 * std, 3 * std))
        self.wk = torch.nn.Parameter(nn.init.trunc_normal_(wk_tensor, 0, std, -3 * std, 3 * std))
        self.wv = torch.nn.Parameter(nn.init.trunc_normal_(wv_tensor, 0, std, -3 * std, 3 * std))
        self.wo = torch.nn.Parameter(nn.init.trunc_normal_(wo_tensor, 0, std, -3 * std, 3 * std))

    def forward(self, input: torch.Tensor, apply_rope: bool = True, token_positions=None) -> torch.Tensor:
        mask = torch.tril(torch.full((input.shape[-2], input.shape[-2]), True, device=self.device))
        batch_size = input.shape[0]
        seq_len = input.shape[-2]

        wqx = torch.matmul(input, torch.t(self.wq))
        wkx = torch.matmul(input, torch.t(self.wk))

        wqx = wqx.reshape(batch_size, seq_len, self.num_heads, self.dk).transpose(1, 2)
        wkx = wkx.reshape(batch_size, seq_len, self.num_heads, self.dk).transpose(1, 2)

        if apply_rope:
            if token_positions is None:
                token_positions = torch.arange(0, seq_len, device=self.device)

            wqx = self.rope.forward(wqx, token_positions)
            wkx = self.rope.forward(wkx, token_positions)

        wvx = (
            torch.matmul(input, torch.t(self.wv)).reshape(batch_size, seq_len, self.num_heads, self.dk).transpose(1, 2)
        )

        attn = func.scaled_dot_product_attention(wqx, wkx, wvx, mask)
        attn = attn.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.num_heads * self.dk)

        return torch.matmul(attn, torch.t(self.wo))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dff: int,
        max_seq_len: int,
        theta: float = 10000,
        device=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.attn_norm = RMSNorm(d_model, device=device)
        self.ff_norm = RMSNorm(d_model, device=device)
        self.attn = MultiheadSelfAttention(d_model, num_heads, max_seq_len, theta, device=device)
        self.ff = SwiGLUFF(d_model, dff, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn.forward(self.attn_norm.forward(x))
        x = x + attn_out
        ff_out = self.ff.forward(self.ff_norm.forward(x))
        return x + ff_out


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        context_length: int,
        num_heads: int,
        dff: int,
        num_layers: int,
        theta: float = 10000,
        device=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.embeddings = Embedding(vocab_size, d_model, device=device)
        self.out_norm = RMSNorm(d_model, device=device)
        self.out_linear = Linear(d_model, vocab_size, device=device)

        self.transformer_blocks = torch.nn.ModuleList([])
        for _ in range(num_layers):
            self.transformer_blocks.append(TransformerBlock(d_model, num_heads, dff, context_length, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = self.embeddings.forward(x)
        for block in self.transformer_blocks:
            input = block.forward(input)
        out_norm = self.out_norm.forward(input)
        return self.out_linear.forward(out_norm)
