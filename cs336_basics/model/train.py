import argparse
import sys
import numpy as np
import wandb
import os
import time

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
    parser.add_argument("--checkp", type=int)
    parser.add_argument("--checkp_out", type=str)

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

    with wandb.init(entity="", project="test-overfit") as run:
        run.config.learning_rate = config.lr
        run.config.vocab = config.vocab
        run.config.d_model = config.dmod
        run.config.context = config.context
        run.config.num_heads = config.heads
        run.config.d_ff = config.dff
        run.config.num_layers = config.layers
        run.config.theta = config.theta
        run.config.device = config.device
        run.config.weight_decay = config.wd
        run.config.beta_one = config.beta1
        run.config.beta_two = config.beta2
        run.config.eps = config.eps

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

            run.log({"loss": loss})

            opt.zero_grad()
            loss.backward()
            opt.step(device=config.device)

            if t % config.checkp == 0:
                print(f"Checkpointing on step {t}")

                out_file = os.path.join(config.checkp_out, f"check_{int(time.time())}_{t}")
                f.save_checkpoint(model, opt, t, out_file)

        out_file = os.path.join(config.checkp_out, f"check_final_{int(time.time())}_{0}")
        f.save_checkpoint(model, opt, 0, out_file)

main(sys.argv[1:])
