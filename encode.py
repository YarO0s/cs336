import sys
import time

from cs336_basics.tokenizers.bpe import Tokenizer

# if not len(sys.argv) > 1:
#     print("text argument is required")

# text = sys.argv[1]

with open(
    "C:\\projects\\datasets\\bpe_demo\\demo_large.txt", encoding="UTF-8"
) as file:
    text = file.read()
    vocab = "C:\\projects\\stanford-cs336\\assignment1-basics\\.results\\owt_train.pickle"
    merges = "C:\\projects\\stanford-cs336\\assignment1-basics\\.results\\owt_train.pickle"
    tokenizer = Tokenizer.from_files(vocab, merges, ["<|endoftext|>"])

    start = time.time()
    tokens = tokenizer.encode(text)
    end = time.time()

    bytes_count = len(text.encode("UTF-8"))
    tokens_count = len(tokens)
    compression_ratio = bytes_count / tokens_count

    print(tokens)
    print(start - end)
    print(
        "Tokens count: {}, Bytes count: {}, Compression ratio: {}".format(tokens_count, bytes_count, compression_ratio)
    )
