import regex as re

import cs336_basics.data.datasets as ds


def train(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    dataset = ds.BPEDemo(input_path)
    rows = dataset.all_rows(dataset.demo_filepath)
    tokens = _pretokenize(rows)
    vocab = _init_vocab()
    num_merges = vocab_size - len(vocab) - len(special_tokens)
    merges = _learn(tokens, num_merges)
    vocab = _extend_vocab(vocab, merges, special_tokens)

    return (vocab, merges)


def _init_vocab() -> dict[int, bytes]:
    vocab = {}
    for i in range(256):
        vocab[i] = int.to_bytes(i)
    return vocab


def _pretokenize(rows: list[str]) -> dict[tuple[bytes], int]:
    pretokens: dict[tuple[bytes], int] = {}
    pattern = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    pattern = re.compile(pattern)

    for row in rows:
        for m in pattern.finditer(row):
            token = tuple(int.to_bytes(byte) for byte in row[m.start() : m.end()].encode("UTF-8"))
            if token in pretokens:
                pretokens[token] = pretokens[token] + 1
            else:
                pretokens[token] = 1
    return pretokens


def _learn(tokens: dict[tuple[bytes], int], num_merges: int = 3) -> list[tuple[bytes, bytes]]:
    merges = []
    for i in range(num_merges):
        byte_pairs, swap_map = _get_byte_pairs(tokens)
        top_byte_pair = _get_top_byte_pairs(byte_pairs)
        # select lexicographically max byte pair
        max_byte_pair = max(top_byte_pair.keys())
        # if max_byte_pair == (b"f", b"e"):
        #     print(f"(i): {i + 256}, (f,e): {byte_pairs[(b'f', b'e')]}, (n,a): {byte_pairs[(b'n', b'a')]}")
        #     raise Exception("bbb")
        merges.append(max_byte_pair)
        tokens = _merge(max_byte_pair, swap_map, tokens)

    return merges


def _get_byte_pairs(
    tokens: dict[tuple[bytes], int],
) -> tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], list[bytes]]]:
    byte_pairs = {}
    swap_map = {}

    for bytes, count in tokens.items():
        for byte_1, byte_2 in zip(bytes[0:-1], bytes[1:]):
            if (byte_1, byte_2) in byte_pairs:
                byte_pairs[(byte_1, byte_2)] += count
            else:
                byte_pairs[(byte_1, byte_2)] = count

            if (byte_1, byte_2) in swap_map:
                if bytes not in swap_map[(byte_1, byte_2)]:
                    swap_map[(byte_1, byte_2)].append(bytes)
            else:
                swap_map[(byte_1, byte_2)] = [bytes]

    return (byte_pairs, swap_map)


def _get_top_byte_pairs(byte_pairs: dict[tuple[bytes, bytes], int]) -> dict[tuple[bytes, bytes], int]:
    top_byte_pairs = []
    top_count = 0

    for byte_pair, count in byte_pairs.items():
        if len(top_byte_pairs) == 0:
            top_byte_pairs.append(byte_pair)
            top_count = count

        if count > top_count:
            top_count = count
            top_byte_pairs = [byte_pair]
        elif count == top_count:
            top_byte_pairs.append(byte_pair)

    top_byte_pair_counts = {}
    for byte_pair in top_byte_pairs:
        top_byte_pair_counts[byte_pair] = byte_pairs[byte_pair]

    return top_byte_pair_counts


def _merge(
    max_byte_pair: tuple[bytes, bytes],
    swap_map: dict[tuple[bytes, bytes], list[bytes]],
    tokens: dict[tuple[bytes], int],
) -> dict[tuple[bytes], int]:
    for bytes in swap_map[max_byte_pair]:
        byte_idx = 0
        merged_bytes = []
        while byte_idx < len(bytes):
            if byte_idx + 1 == len(bytes):
                merged_bytes.append(bytes[byte_idx])
                break

            if (bytes[byte_idx], bytes[byte_idx + 1]) == max_byte_pair:
                merged_bytes.append(b"".join(max_byte_pair))
                byte_idx += 2
            else:
                merged_bytes.append(bytes[byte_idx])
                byte_idx += 1

        count = tokens.pop(bytes)
        tokens[tuple(byte for byte in merged_bytes)] = count

    return tokens


def _extend_vocab(
    vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]
) -> dict[int, bytes]:
    for i, byte_pair in zip(range(len(vocab), len(vocab) + len(merges)), merges):
        vocab[i] = byte_pair[0] + byte_pair[1]

    for i, token in zip(range(len(vocab), len(vocab) + len(special_tokens)), special_tokens):
        vocab[i] = token.encode("UTF-8")

    return vocab
