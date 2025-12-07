import os
import time
import pickle

from cs336_basics.tokenizers.bpe import Tokenizer

# TODO: move logic out of main
if __name__ == "__main__":
    CWD = os.getcwd()

    start = time.time()

#     # TODO: move path to env
#     # input_path = "C:\\projects\\datasets\\bpe_demo\\bpe_demo.txt"
#     input_path = "C:\\projects\\datasets\\tinystories\\TinyStories-valid.txt"
#     # input_path = "C:\\projects\\datasets\\tinystories\\TinyStories-train.txt"
#     # input_path = "C:\\projects\\datasets\\openwebtext\\owt_valid\\owt_valid.txt"

#     result_dir = os.path.join(CWD, ".results")
#     os.makedirs(result_dir, exist_ok=True)

#     result_basename = os.path.basename(input_path)
#     result_filename, _ = os.path.splitext(result_basename)
#     result_filepath = os.path.join(result_dir, f"{result_filename}.pickle")
#     print(result_filepath)

#     if os.path.exists(result_filepath):
#         raise Exception(f"File {result_filepath} already exists")

#     bpe_vocab, bpe_merges = bpe.train(input_path, 10_000, ["<|endoftext|>"])
#     end = time.time()
#     print(end - start)

#     with open(result_filepath, "wb") as file:
#         pickle.dump((bpe_vocab, bpe_merges), file)
    tokenizer = Tokenizer.from_files(
        "C:\\projects\\stanford-cs336\\assignment1-basics\\.results\\owt_train.pickle",
        "C:\\projects\\stanford-cs336\\assignment1-basics\\.results\\owt_train.pickle")

    # tokens = tokenizer.encode("hello world")
    # print(len(tokens))
    # print(tokens)
    ids = [13175, 111, 921]
    print(tokenizer.decode(ids))
