import argparse
import sys

import numpy as np

from cs336_basics.tokenizers import bpe


def main(args: list[str]):
    parser = argparse.ArgumentParser("encode data")

    parser.add_argument("--data")

    config = parser.parse_args(args)

    data = np.memmap(config.data, dtype=np.long)
    print(f"size: {len(data)}")
    print(f"shape: {data.shape}")
    print(f"sample: {data[0:1000]}")
    print(f"dtype: {data.dtype}")
    print(f"max item: {data.max()}")
    print(f"min item: {data.min()}")

main(sys.argv[1:])
