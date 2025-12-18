from cs336_basics.tokenizers.bpe import Tokenizer
import sys

if not len(sys.argv) > 1:
    print("text argument is required")

text = sys.argv[1]
vocab = "/home/yaroslav/Projects/cs336/.results/owt_train.pickle"
merges = "/home/yaroslav/Projects/cs336/.results/owt_train.pickle"
tokenizer = Tokenizer.from_files(vocab, merges, ["<|endoftext|>"])
print(tokenizer.encode(text))
