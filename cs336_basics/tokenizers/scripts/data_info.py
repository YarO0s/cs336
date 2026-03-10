import argparse
import sys

import numpy as np

from cs336_basics.tokenizers import bpe


def main(args: list[str]):
    parser = argparse.ArgumentParser("encode data")

    parser.add_argument("--data")

    config = parser.parse_args(args)

    data = np.memmap(config.data)
    print(f"dataset size: {len(data)}")
    print(f"dataset shape: {data.shape}")

main(sys.argv[1:])
