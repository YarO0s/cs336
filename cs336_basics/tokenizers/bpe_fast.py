import regex as re

import cs336_basics.data.datasets as ds

PRETOKENIZE_PATTERN = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"


def train(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    dataset = ds.BPEDemo(input_path)
    rows = dataset.all_rows(dataset.demo_filepath)
    byte_pairs_pretokens, inv_byte_pairs_counts, byte_pairs_counts, pretokens_counts, pretokens_hash_map = _pretokenize(
        rows
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
