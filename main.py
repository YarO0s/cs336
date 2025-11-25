import cProfile
import os
import time

from cs336_basics.tokenizers import bpe
from cs336_basics.tokenizers.data import ChunkIdentifier

if __name__ == "__main__":
    CWD = os.getcwd()

    start = time.time()
    # input_path = "C:\\projects\\datasets\\bpe_demo\\bpe_demo.txt"
    # input_path = "C:\\projects\\datasets\\tinystories\\TinyStories-valid.txt"
    input_path = "C:\\projects\\datasets\\tinystories\\TinyStories-train.txt"
    delimeter = "<|endoftext|>"
    ds = ChunkIdentifier(input_path, delimeter)
    # bpe_vocab, bpe_merges = bpe.train(input_path, 10_000, ["<|endoftext|>"])
    # end = time.time()
    # print(end-start)

    chunks = ds.get_chunks_positions(16)
    bpe.mp_pretokenize(chunks, input_path, delimeter.encode("UTF-8"))
    end = time.time()
    print(end - start)
