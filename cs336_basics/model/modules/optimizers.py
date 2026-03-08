import math
import torch

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, betas, eps) -> None:
        super().__init__(params, {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
            "t": 0,
        })

    def step(self, closure = None, device="cpu") -> None:
        for group in self.param_groups:

            beta_1, beta_2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            w_d = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                t = state.get("t", 1)
                m_t = state.get("m_t", torch.zeros(p.data.shape, device=device))
                v_t = state.get("v_t", torch.zeros(p.data.shape, device=device))

                m_t_new = beta_1 * m_t + (1 - beta_1) * p.grad
                v_t_new = beta_2 * v_t + (1 - beta_2) * (p.grad ** 2)

                state["m_t"] = m_t_new
                state["v_t"] = v_t_new

                m_t_hat = 1 - math.pow(beta_1, t)
                v_t_hat = 1 - math.pow(beta_2, t)

                lr_t = lr * math.sqrt(v_t_hat) / m_t_hat

                p.data -= lr_t * m_t_new / (torch.sqrt(v_t_new) + eps)
                p.data -= lr * w_d * p.data

                state["t"] = t + 1
