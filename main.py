import cProfile
import os
import time

from memory_profiler import profile

from cs336_basics.tokenizers import bpe, bpe_fast

CWD = os.getcwd()

# input_path = "C:\projects\datasets\\tinystories\TinyStories-valid.txt"
input_path = os.path.join(CWD, "tests\\fixtures\\tinystories_sample_5M.txt")
bpe_vocab, bpe_merges = bpe.train(input_path, 1000, ["<|endoftext|>"])
# bpe_fast_vocab, bpe_fast_merges = bpe_fast.train(input_path, 1000, ["<|endoftext|>"])
# for bpe_pair, bpe_fast_pair in zip(bpe_vocab.items(), bpe_fast_vocab.items()):
#     if bpe_pair[1] != bpe_fast_pair[1]:
#         print(f"{bpe_pair[0]}: expected {bpe_pair[1]}, actual: {bpe_fast_pair[1]}")
print("end")
