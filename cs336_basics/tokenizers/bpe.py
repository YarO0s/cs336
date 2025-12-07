import os
import pickle
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from collections.abc import Iterable, Iterator

import pandas as pd
import regex as re
import xxhash

import cs336_basics.data.datasets as ds
from cs336_basics.tokenizers.data import ChunkIdentifier

HASH_SEED = 1
PRETOKENIZE_PATTERN = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
DELIMETER = "<|endoftext|>"


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str]|None = None
    ):
        self.vocab = vocab
        self.inverted_vocab = {value: key for key, value in self.vocab.items()}
        self.merges = merges
        self.spec_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str]|None = None):
        if not os.path.exists(vocab_filepath):
            raise Exception(f"Vocab file: {vocab_filepath} does not exist")
        if not os.path.exists(merges_filepath):
            raise Exception(f"Merges file: {merges_filepath} does not exist")

        _, vocab_ext = os.path.splitext(vocab_filepath)
        _, merges_ext = os.path.splitext(merges_filepath)

        if vocab_ext != ".pickle":
            raise Exception(f"Vocab file: {vocab_filepath} must be a pickle serialized type")
        if merges_ext != ".pickle":
            raise Exception(f"Merges file: {merges_filepath} must be a pickle serialized type")

        if vocab_filepath == merges_filepath:
            with open(vocab_filepath, "rb") as file:
                vocab, merges = pickle.load(file)
                return cls(vocab, merges, special_tokens)

        vocab = None
        merges = None

        with open(vocab_filepath, "rb") as file:
            vocab = pickle.load(file)
        with open(merges_filepath, "rb") as file:
            merges = pickle.load(file)

        return cls(vocab, merges, special_tokens)


    def decode(self, ids: list[int]) -> str:
        bytes = b""
        bytes = bytes.join(self.vocab[id] for id in ids)
        return bytes.decode("UTF-8", errors="replace")


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            ids = self.encode(text)
            for id in ids:
                yield id


    def encode(self, text: str) -> list[int]:
        pattern = re.compile(PRETOKENIZE_PATTERN)
        pretokens: list[tuple[bytes, ...]] = []
        merged_pretokens: list[tuple[bytes, ...]] = []
        tokens: list[int] = []

        for m in pattern.finditer(text):
            pretoken = tuple(int.to_bytes(byte) for byte in text[m.start() : m.end()].encode("UTF-8"))
            pretokens.append(pretoken)

        for pretoken in pretokens:
            if len(pretoken) == 1:
                merged_pretokens.append(pretoken)
                continue

            for merge in self.merges:
                if set(merge).issubset(pretoken):
                    merged_pretoken = self._merge_pretoken_bytes(pretoken, merge)
                    pretoken = merged_pretoken
                    if len(pretoken) == 1:
                        break

            merged_pretokens.append(pretoken)

        for pretoken in merged_pretokens:
            for byte in pretoken:
                tokens.append(self.inverted_vocab[byte])

        return tokens


    def _merge_pretoken_bytes(self, pretoken: tuple[bytes, ...], merge: tuple[bytes, bytes]) -> tuple[bytes, ...]:
        after_merge: list[bytes] = []
        byte_id = 0

        while byte_id < len(pretoken):
            if byte_id + 1 == len(pretoken):
                after_merge.append(pretoken[byte_id])
                break

            if (pretoken[byte_id], pretoken[byte_id + 1]) == merge:
                after_merge.append(b"".join(merge))
                byte_id += 2
            else:
                after_merge.append(pretoken[byte_id])
                byte_id += 1

        return tuple(after_merge)


def visualize(vocab: dict[int, bytes], limit: int):
    df = pd.DataFrame({"ids": vocab.keys(), "tokens": vocab.values()})
    df["len"] = df.apply(lambda x: len(x.tokens), axis=1)
    df = df.sort_values(by="len", ascending=False)
    print(df[0:limit])


def load(path: str) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if not os.path.exists(path):
        raise Exception(f"File {path} does not exist")

    with open(path, "rb") as file:
        vocab, byte_pairs = pickle.load(file)
        return vocab, byte_pairs


def train(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    ds = ChunkIdentifier(input_path, DELIMETER)
    chunks = ds.get_chunks_positions(16)

    byte_pairs_pretokens, inv_byte_pairs_counts, byte_pairs_counts, pretokens_counts, pretokens_hash_map = (
        mp_pretokenize(chunks, input_path, DELIMETER.encode("UTF-8"))
    )
    vocab = _init_vocab()
    num_merges = vocab_size - len(vocab) - len(special_tokens)
    merges = _get_merges(
        byte_pairs_pretokens, inv_byte_pairs_counts, byte_pairs_counts, pretokens_counts, pretokens_hash_map, num_merges
    )

    for merge in merges:
        vocab[len(vocab)] = b"".join(merge)

    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("UTF-8")

    return (vocab, merges)


def _init_vocab() -> dict[int, bytes]:
    vocab = {}
    for i in range(256):
        vocab[i] = int.to_bytes(i)
    return vocab


def _pretokenize(
    rows: list[str],
) -> tuple[
    dict[tuple[bytes, bytes], list[int]],
    dict[int, list[tuple[bytes, bytes]]],
    dict[tuple[bytes, bytes], int],
    dict[int, int],
    dict[int, tuple[bytes]],
]:
    """
    This function pretokenizes the dataset with a regex

    @param rows: rows from a dataset or it's chunk
    @return:
        dict[tuple[bytes,bytes], list[tuple[bytes]]] - all pretokens
            for each byte pair where it occurs
        dict[int, list[tuple[bytes,bytes]]] - all byte pairs for each
            number of their occurence (inverted counter)
        dict[tuple[bytes], int] - all pretokens counts
    """
    pretokens_hash_table: dict[int, tuple[bytes]] = {}
    byte_pairs_pretokens: dict[tuple[bytes, bytes], list[int]] = {}
    byte_pairs_counts: dict[tuple[bytes, bytes], int] = {}
    pretokens_counts: dict[int, int] = {}
    inv_byte_pairs_counts: dict[int, list[tuple[bytes, bytes]]] = {}

    pattern = re.compile(PRETOKENIZE_PATTERN)

    for row in rows:
        for m in pattern.finditer(row):
            token = tuple(int.to_bytes(byte) for byte in row[m.start() : m.end()].encode("UTF-8"))
            token_hash = hash(token)

            if token_hash in pretokens_counts:
                pretokens_counts[token_hash] += 1
            else:
                pretokens_counts[token_hash] = 1

            if token_hash not in pretokens_hash_table:
                pretokens_hash_table[token_hash] = token

    for token_hash, token in pretokens_hash_table.items():
        for byte_1, byte_2 in zip(token[:-1], token[1:]):
            byte_pair = (byte_1, byte_2)

            if byte_pair in byte_pairs_pretokens:
                if token not in byte_pairs_pretokens[byte_pair]:
                    byte_pairs_pretokens[byte_pair].append(token_hash)
            else:
                byte_pairs_pretokens[byte_pair] = [token_hash]

            if byte_pair in byte_pairs_counts:
                byte_pairs_counts[byte_pair] += pretokens_counts[token_hash]
            else:
                byte_pairs_counts[byte_pair] = pretokens_counts[token_hash]

    for byte_pair, count in byte_pairs_counts.items():
        if count in inv_byte_pairs_counts:
            inv_byte_pairs_counts[count].append(byte_pair)
        else:
            inv_byte_pairs_counts[count] = [byte_pair]

    return (byte_pairs_pretokens, inv_byte_pairs_counts, byte_pairs_counts, pretokens_counts, pretokens_hash_table)


def _mp_get_pretokens(chunk: tuple[int, int], path: str, delimeter: bytes):
    pretokens_hash_table: dict[int, tuple[bytes]] = {}
    pretokens_counts: dict[int, int] = {}
    byte_pairs_pretokens: dict[tuple[bytes, bytes], list[int]] = {}
    byte_pairs_counts: dict[tuple[bytes, bytes], int] = {}

    bytes_pattern = PRETOKENIZE_PATTERN.encode("UTF-8")

    with open(path, "rb+") as file:
        hasher = xxhash.xxh64(seed=HASH_SEED)
        file.seek(chunk[0])
        data = file.read(chunk[1] - chunk[0])
        rows = data.split(delimeter)

        for row in rows:
            if b"\r\n" in row:
                row = row.replace(b"\r\n", b"\n")
            for m in re.finditer(bytes_pattern, row):
                token = tuple(int.to_bytes(byte) for byte in row[m.start() : m.end()])
                hasher.reset()
                hasher.update(row[m.start() : m.end()])
                token_hash = hasher.intdigest()

                if token_hash in pretokens_counts:
                    pretokens_counts[token_hash] += 1
                else:
                    pretokens_counts[token_hash] = 1

                if token_hash not in pretokens_hash_table:
                    pretokens_hash_table[token_hash] = token

        for token_hash, token in pretokens_hash_table.items():
            for byte_1, byte_2 in zip(token[:-1], token[1:]):
                byte_pair = (byte_1, byte_2)

                if byte_pair in byte_pairs_pretokens:
                    if token not in byte_pairs_pretokens[byte_pair]:
                        byte_pairs_pretokens[byte_pair].append(token_hash)
                else:
                    byte_pairs_pretokens[byte_pair] = [token_hash]

                if byte_pair in byte_pairs_counts:
                    byte_pairs_counts[byte_pair] += pretokens_counts[token_hash]
                else:
                    byte_pairs_counts[byte_pair] = pretokens_counts[token_hash]
    return (pretokens_hash_table, pretokens_counts, byte_pairs_pretokens, byte_pairs_counts)


def mp_pretokenize(chunks_offsets: list[tuple[int, int]], path: str, delimeter: bytes):
    pretokens_hash_table: dict[int, tuple[bytes]] = {}
    pretokens_counts: dict[int, int] = Counter({})
    byte_pairs_pretokens: dict[tuple[bytes, bytes], list[int]] = {}
    byte_pairs_counts: dict[tuple[bytes, bytes], int] = Counter({})
    inv_byte_pairs_counts: dict[int, list[tuple[bytes, bytes]]] = {}

    num_workers = len(chunks_offsets)
    if not os.path.exists(path):
        raise Exception(f"File {path} does not exist")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for chunk, out in zip(
            chunks_offsets, executor.map(_mp_get_pretokens, chunks_offsets, repeat(path), repeat(delimeter))
        ):
            pretokens_hash_table.update(out[0])
            pretokens_counts += Counter(out[1])
            byte_pairs_counts += Counter(out[3])

            for byte_pair, pretokens in out[2].items():
                if byte_pair in byte_pairs_pretokens:
                    byte_pairs_pretokens[byte_pair].extend(pretokens)
                else:
                    byte_pairs_pretokens[byte_pair] = pretokens

    for byte_pair, count in byte_pairs_counts.items():
        if count in inv_byte_pairs_counts:
            inv_byte_pairs_counts[count].append(byte_pair)
        else:
            inv_byte_pairs_counts[count] = [byte_pair]

    return (byte_pairs_pretokens, inv_byte_pairs_counts, byte_pairs_counts, pretokens_counts, pretokens_hash_table)


def _get_merges(
    byte_pairs_pretokens: dict[tuple[bytes, bytes], list[int]],
    inv_byte_pairs_counts: dict[int, list[tuple[bytes, bytes]]],
    byte_pairs_counts: dict[tuple[bytes, bytes], int],
    pretokens_counts: dict[int, int],
    pretokens_hash_map: dict[int, tuple[bytes]],
    num_merges: int,
) -> list[tuple[bytes, bytes]]:
    merges: list[tuple[bytes, bytes]] = []
    top_byte_pairs: list[tuple[bytes, bytes]] = []

    for _ in range(num_merges):
        top_byte_pair_count, top_byte_pair = _get_top_byte_pair(inv_byte_pairs_counts)
        top_byte_pairs.append(top_byte_pair)
        merges.append((top_byte_pair[0], top_byte_pair[1]))

        _update_cache(
            byte_pairs_pretokens,
            inv_byte_pairs_counts,
            byte_pairs_counts,
            pretokens_counts,
            pretokens_hash_map,
            top_byte_pair,
            top_byte_pair_count,
        )

    return merges


def _merge_byte_pair(
    top_byte_pair: tuple[bytes, bytes],
    byte_pairs_pretokens: dict[tuple[bytes, bytes], list[int]],
    pretokens_hash_map: dict[int, tuple[bytes]],
):
    for pretoken_hash in byte_pairs_pretokens[top_byte_pair]:
        pretoken = pretokens_hash_map[pretoken_hash]
        byte_idx = 0
        merged_bytes = []
        while byte_idx < len(pretoken):
            if byte_idx + 1 == len(pretoken):
                merged_bytes.append(pretoken[byte_idx])
                break

            if (pretoken[byte_idx], pretoken[byte_idx + 1]) == top_byte_pair:
                merged_bytes.append(b"".join(top_byte_pair))
                byte_idx += 2
            else:
                merged_bytes.append(pretoken[byte_idx])
                byte_idx += 1

        pretokens_hash_map[pretoken_hash] = tuple(byte for byte in merged_bytes)


def _get_top_byte_pair(byte_pairs_counts: dict[int, list[tuple[bytes, bytes]]]) -> tuple[int, tuple[bytes, bytes]]:
    max_count = max(byte_pairs_counts.keys())
    byte_pairs = byte_pairs_counts[max_count]

    if len(byte_pairs) > 1:
        return (max_count, max(byte_pairs))

    return (max_count, byte_pairs[0])


def _update_cache(
    byte_pairs_pretokens: dict[tuple[bytes, bytes], list[int]],
    inv_byte_pairs_counts: dict[int, list[tuple[bytes, bytes]]],
    byte_pairs_counts: dict[tuple[bytes, bytes], int],
    pretoken_counts: dict[int, int],
    pretokens_hash_map: dict[int, tuple[bytes]],
    top_byte_pair: tuple[bytes, bytes],
    top_byte_pair_count: int,
):
    pretoken_hashes = set(byte_pairs_pretokens[top_byte_pair])
    byte_pair_bytes: bytes = b"".join(top_byte_pair)
    merges_counts: dict[tuple[bytes, bytes], int] = {}
    related_pairs: dict[tuple[bytes, bytes], int] = {}

    for pretoken_hash in pretoken_hashes:
        pretoken = pretokens_hash_map[pretoken_hash]
        last_byte_id = len(pretoken) - 2
        pretoken_count = pretoken_counts[pretoken_hash]

        if len(pretoken) <= 1:
            continue

        for i, byte in enumerate(pretoken[:-1]):
            if (byte, pretoken[i + 1]) != top_byte_pair:
                continue

            pairs = []

            if len(pretoken) != 2:
                if i == 0:
                    pair_byte = pretoken[i + 2]
                    pairs.append((byte_pair_bytes, pair_byte))

                    related_pair = ((pretoken[i + 1]), pair_byte)
                    if related_pair in related_pairs:
                        related_pairs[related_pair] += pretoken_count
                    else:
                        related_pairs[related_pair] = pretoken_count

                elif i == last_byte_id:
                    pair_byte = pretoken[i - 1]
                    pairs.append((pair_byte, byte_pair_bytes))

                    related_pair = (pair_byte, byte)
                    if related_pair in related_pairs:
                        related_pairs[related_pair] += pretoken_count
                    else:
                        related_pairs[related_pair] = pretoken_count
                else:
                    preceding_pair_byte = pretoken[i - 1]
                    trailing_pair_byte = pretoken[i + 2]

                    pairs.append((preceding_pair_byte, byte_pair_bytes))
                    pairs.append((byte_pair_bytes, trailing_pair_byte))

                    preceding_related_pair = (preceding_pair_byte, byte)
                    if preceding_related_pair in related_pairs:
                        related_pairs[preceding_related_pair] += pretoken_count
                    else:
                        related_pairs[preceding_related_pair] = pretoken_count
                        # TODO: handle the case where related pairs equal to top_byte_pair and where each of them
                        #      has been merged before like pretoken = (a,bcde,bcde,f), top_byte_pair = (bcde, bcde),
                        #      and pretoken before merge was (a,bc,de,bc,de,f). Seems that in such case (bcde,b) and (e,bcde)
                        #      related pairs will be taken instead of (bcde, bc) and (de, bcde).

                    trailing_related_pair = (pretoken[i + 1], trailing_pair_byte)
                    if trailing_related_pair in related_pairs:
                        related_pairs[trailing_related_pair] += pretoken_count
                    else:
                        related_pairs[trailing_related_pair] = pretoken_count

            for pair in pairs:
                if pair in byte_pairs_pretokens:
                    byte_pairs_pretokens[pair].append(pretoken_hash)
                else:
                    byte_pairs_pretokens[pair] = [pretoken_hash]

                if pair in merges_counts:
                    merges_counts[pair] += pretoken_count
                else:
                    merges_counts[pair] = pretoken_count

                if pair in byte_pairs_counts:
                    byte_pairs_counts[pair] += pretoken_count
                else:
                    byte_pairs_counts[pair] = pretoken_count

    for merge, count in merges_counts.items():
        if count in inv_byte_pairs_counts:
            inv_byte_pairs_counts[count].append(merge)
        else:
            inv_byte_pairs_counts[count] = [merge]

    _merge_byte_pair(top_byte_pair, byte_pairs_pretokens, pretokens_hash_map)

    del byte_pairs_pretokens[top_byte_pair]
    del byte_pairs_counts[top_byte_pair]

    if len(inv_byte_pairs_counts[top_byte_pair_count]) > 1:
        inv_byte_pairs_counts[top_byte_pair_count].remove(top_byte_pair)
    else:
        del inv_byte_pairs_counts[top_byte_pair_count]

    for related_pair, count in related_pairs.items():
        if related_pair in byte_pairs_counts:
            related_pair_count = byte_pairs_counts[related_pair]

            if count >= related_pair_count:
                del byte_pairs_counts[related_pair]

                if related_pair_count in inv_byte_pairs_counts:
                    if len(inv_byte_pairs_counts[related_pair_count]) > 1:
                        inv_byte_pairs_counts[related_pair_count].remove(related_pair)
                    else:
                        del inv_byte_pairs_counts[related_pair_count]
            else:
                new_count = related_pair_count - count
                byte_pairs_counts[related_pair] = new_count

                if new_count in inv_byte_pairs_counts:
                    inv_byte_pairs_counts[new_count].append(related_pair)
                else:
                    inv_byte_pairs_counts[new_count] = [related_pair]

                if related_pair_count in inv_byte_pairs_counts:
                    if len(inv_byte_pairs_counts[related_pair_count]) > 1:
                        inv_byte_pairs_counts[related_pair_count].remove(related_pair)
                    else:
                        del inv_byte_pairs_counts[related_pair_count]
