import argparse

import sys
import torch
import numpy as np

from cs336_basics.model.modules import func
from cs336_basics.tokenizers import bpe


def main(args: list[str]):
    parser = argparse.ArgumentParser("encode data")

    parser.add_argument("--vocab")
    parser.add_argument("--merges")
    parser.add_argument("--data")

    config = parser.parse_args(args)

    tokenizer = bpe.Tokenizer.from_files(config.vocab, config.merges, [])
    print(f"merges size: {len(tokenizer.merges)}\nvocab size: {len(tokenizer.vocab)}")

    data = np.memmap(config.data)
    print(tokenizer.decode(data))

main(sys.argv[1:])
