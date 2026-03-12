import argparse
import sys
import time

import numpy as np

from tqdm import tqdm
from memory_profiler import os
from cs336_basics.tokenizers import bpe


def main(args: list[str]):
    parser = argparse.ArgumentParser("encode data")

    parser.add_argument("--file")
    parser.add_argument("--vocab")
    parser.add_argument("--merges")

    config = parser.parse_args(args)

    encode(config)


def encode(config: argparse.Namespace):
    base_name = os.path.basename(config.file)
    file_name, _ = os.path.splitext(base_name)

    now = int(time.time())

    target_name = os.path.join("/mnt/c/projects/stanford-cs336/assignment1-basics/.encoded", f"{file_name}-{now}.npy")

    if os.path.exists(target_name):
        raise Exception(f"File already exists: {target_name}")

    # Encode
    print("Encoding")
    tokenizer = bpe.Tokenizer.from_files(config.vocab, config.merges, ["<|endoftext|>"])

    max_lines = 140_815_490
    batch_size = 100_000
    batch = np.array([], dtype=np.long)
    with open(config.file) as source:
        with open(target_name, "wb") as target:
            for _, line in tqdm(zip(range(14815490), source), total=max_lines):
                ids = tokenizer.encode(line)
                batch = np.concatenate((batch, ids))

                if len(batch) > batch_size:
                    np.array(batch, dtype=np.long).tofile(target)
                    batch = np.array([], dtype=np.long)

            if len(batch) > 0:
                np.array(batch).tofile(target)


main(sys.argv[1:])
