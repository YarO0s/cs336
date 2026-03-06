import math

import torch


def softmax(input: torch.Tensor, dim: int = 0) -> torch.Tensor:
    max_items = torch.max(input, dim=dim).values
    max_items = max_items.reshape(max_items.shape + (1,))
    stable_input = input - max_items

    input_exp = torch.exp(stable_input)
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


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    max_items = torch.max(inputs, dim=-1, keepdim=True).values
    shifted = inputs - max_items

    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1))
    logits = log_sum_exp - torch.gather(shifted, -1, targets.reshape(targets.shape + (1,)))

    return torch.mean(logits)


def lr_cosine_annealing(t: int, lr_max, lr_min, warmup_iters, cos_iters):
    if t < warmup_iters:
        return (t / warmup_iters) * lr_max

    if t >= warmup_iters and t <= cos_iters:
        cosine = math.cos((t-warmup_iters/cos_iters-warmup_iters) * math.pi)
        return lr_min + 0.5 * (1 + cosine) * (lr_max - lr_min)

    return lr_min


def grad_clipping(params: torch.nn.parameter, max: float):
    grads = []

    for param in params:
        if param.grad is None:
            continue

        grads.append(param.grad)

    norm = torch.norm(
        torch.norm(torch.stack([g.detach().norm() for g in grads]))
    )

    if norm > max:
        scale = max / (norm + 1e-6)

        for param in params:
            if param.grad is not None:
                param.grad *= scale


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out) -> None:
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }, out)

def load_checkpoint(src, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    checkpoint = torch.load(src)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint["iteration"]
