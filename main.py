import cProfile
import os
import time

from cs336_basics.tokenizers import bpe, bpe_fast

CWD = os.getcwd()

input_path = os.path.join(CWD, "C:\\projects\\datasets\\tinystories\\TinyStories-train.txt")
bpe_vocab, bpe_merges = bpe_fast.train(input_path, 1000, ["<|endoftext|>"])
