import argparse
import sys
import numpy as np
import torch.nn.functional as nnfunc

import cs336_basics.model.modules.func as f
import cs336_basics.model.modules.modules as m
import cs336_basics.model.modules.optimizers as o
import cs336_basics.model.data.loader as l


def main(args: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="Training script")
    parser.add_argument("--vocab", type=int)
    parser.add_argument("--dmod", type=int)
    parser.add_argument("--context", type=int)
    parser.add_argument("--heads", type=int)
    parser.add_argument("--dff", type=int)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--theta", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--wd", type=float)
    parser.add_argument("--beta1", type=float)
    parser.add_argument("--beta2", type=float)
    parser.add_argument("--eps", type=float)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--steps", type=int)

    config = parser.parse_args(args)
    train(config)


def train(config):
    data = np.memmap(config.dataset, dtype=np.long)

    model = m.TransformerLM(
        config.vocab,
        config.dmod,
        config.context,
        config.heads,
        config.dff,
        config.layers,
        config.theta,
        config.device
    )
    opt = o.AdamW(
        model.parameters(),
        config.lr,
        config.wd,
        (config.beta1, config.beta2),
        config.eps
    )

    for t in range(config.steps):
        train, expected = l.get_batch(
            data,
            config.batch,
            config.context,
            config.device
        )

        predicted = model.forward(train)
        loss = f.cross_entropy(
            predicted.view((predicted.shape[0] * predicted.shape[1], predicted.shape[2])),
            expected.view((predicted.shape[0] * predicted.shape[1]))
        )
        print(f"{t}: {loss}")

        opt.zero_grad()
        loss.backward()
        opt.step(device=config.device)


main(sys.argv[1:])
