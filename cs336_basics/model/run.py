import argparse
import sys
import numpy as np
import torch

import cs336_basics.model.modules.func as f
import cs336_basics.model.modules.modules as m
import cs336_basics.model.modules.optimizers as o
import cs336_basics.model.data.loader as l

from cs336_basics.tokenizers.bpe import Tokenizer


def main(args: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="Training script")
    parser.add_argument("--input", type=str)

    config = parser.parse_args(args)
    run(config)


def run(config):
    device = "cuda:0"
    model_file = "/mnt/c/projects/datasets/models/tinystories-gpt-1/model_final.pt"
    model_weights = "/mnt/c/projects/datasets/models/tinystories-gpt-1/check_final_1772995273_0"
    vocab_file = "/mnt/c/projects/stanford-cs336/assignment1-basics/.results/TinyStories-train.pickle"

    model = torch.load(model_file, weights_only=False)
    model.load_state_dict(torch.load(model_weights)["model"])
    bpe = Tokenizer.from_files(vocab_file, vocab_file)

    encoded_input = torch.tensor(bpe.encode(config.input), device=device)

    print(config.input, end='')

    for out in model.infer(encoded_input, 256):
        out_list = out.tolist()
        out_decoded = bpe.decode([out_list])
        # print(out_decoded, end='')
        print(f"{out_list}: {out_decoded}")

    print("\n")

main(sys.argv[1:])
