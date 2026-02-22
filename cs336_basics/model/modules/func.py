import math

import torch


def softmax(input: torch.Tensor, dim: int = 0) -> torch.Tensor:
    max_items = torch.max(input, dim=dim).values
    max_items = max_items.reshape(max_items.shape + (1,))
    input -= max_items

    input_exp = torch.exp(input)
    exp_sum = torch.sum(input_exp, dim)

    return torch.div(input_exp, exp_sum.reshape(exp_sum.shape + (1,)))


def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    last_dim = len(k.shape) - 1
    delim = math.sqrt(k.shape[last_dim])

    qk_product = torch.matmul(q, torch.transpose(k, -2, -1)) / delim
    qk_product_normal = torch.where(mask == True, qk_product, -torch.inf)
    attention_matrix = softmax(qk_product_normal, last_dim)

    return torch.matmul(attention_matrix, v)
