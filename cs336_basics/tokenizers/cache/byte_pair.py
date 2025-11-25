from typing import Self


class PretokensHashMap:
    def __init__(self, hash_map: dict[int, tuple[bytes] | tuple[bytes, ...]] = {}):
        self.hash_map = hash_map


    def find(self, pretoken_hash: int) -> tuple[bytes] | tuple[bytes, ...]:
        return self.hash_map[pretoken_hash]


    def insert(self, pretoken: tuple[bytes] | tuple[bytes, ...], ignore_existing: bool = True) -> int:
        pretoken_hash = hash(pretoken)

        if pretoken_hash in self.hash_map and not ignore_existing:
            raise Exception(f"Pretoken {pretoken_hash}: {pretoken} already exists")

        self.hash_map[pretoken_hash] = pretoken

        return pretoken_hash


    def merge_byte_pair(self, hashes: list[int], merge_byte_pair: tuple[bytes, bytes]):
        for hash in hashes:
            bytes_with_merges = []
            byte_id = 0

            pretoken = self.hash_map[hash]
            while byte_id < len(pretoken):
                if byte_id + 1 == len(pretoken):
                    bytes_with_merges.append(pretoken[byte_id])
                    break

                if (pretoken[byte_id], pretoken[byte_id + 1]) == merge_byte_pair:
                    bytes_with_merges.append(b"".join(merge_byte_pair))
                    byte_id += 2
                else:
                    bytes_with_merges.append(pretoken[byte_id])
                    byte_id += 1

            self.hash_map[hash] = tuple(byte for byte in bytes_with_merges)


class PretokensCountsHashMap:
    def __init__(self, hash_map: dict[int, int] = {}):
        self.hash_map = hash_map


    def find(self, pretoken_hash) -> int:
        return self.hash_map[pretoken_hash]


    def increment_by(self, pretoken_hash: int, increment: int = 1, insert_missing: bool = True) -> None:
        if pretoken_hash in self.hash_map:
            self.hash_map[pretoken_hash] += increment
        else:
            if not insert_missing:
                raise Exception(f"Pretoken {pretoken_hash}: does not exist")

            self.hash_map[pretoken_hash] = increment


class BytePairsInvertedIndex:
    def __init__(self, byte_pairs: dict[tuple[bytes, bytes], list[int]] = {}):
        self.byte_pairs: dict[tuple[bytes, bytes], list[int]] = byte_pairs


    def find_hashes(self, byte_pair: tuple[bytes, bytes]) -> list[int]:
        return self.byte_pairs[byte_pair]


    def insert(self, byte_pair: tuple[bytes, bytes], pretoken_hash: int) -> None:
        if byte_pair in self.byte_pairs:
            self.byte_pairs[byte_pair].append(pretoken_hash)
        else:
            self.byte_pairs[byte_pair] = [pretoken_hash]


    def insert_from_pretoken(
        self,
        pretoken: tuple[bytes] | tuple[bytes, ...],
        pretoken_hash: int
    ) -> None:
        for byte_1, byte_2 in zip(pretoken[:-1], pretoken[1:]):
            byte_pair = (byte_1, byte_2)
            self.insert(byte_pair, pretoken_hash)


    def join(self, byte_pairs: Self) -> None:
        for byte_pair, hashes in byte_pairs.byte_pairs.items():
            if byte_pair in self.byte_pairs:
                self.byte_pairs[byte_pair].extend(hashes)
            else:
                self.byte_pairs[byte_pair] = hashes


    def remove(self, byte_pair: tuple[bytes, bytes]) -> None:
        del self.byte_pairs[byte_pair]


class BytePairsCountsMap:
    def __init__(
        self,
        counts_map: dict[tuple[bytes, bytes], int] = {},
        counts_inverted_index: dict[int, list[tuple[bytes, bytes]]] = {}
    ):
        self.counts_map: dict[tuple[bytes, bytes], int] = counts_map
        self.counts_inverted_index: dict[int, list[tuple[bytes, bytes]]] = counts_inverted_index


    def get_count(self, bytes):
        return self.counts_map[bytes]


    def get_top_pair(self) -> tuple[int, tuple[bytes, bytes]]:
        max_count = max(self.counts_inverted_index.keys())
        top_pairs = self.counts_inverted_index[max_count]

        return (max_count, max(top_pairs))


    def increment_by(self, byte_pair: tuple[bytes, bytes], increment: int = 1, set_inverted_index = False) -> None:
        if byte_pair in self.counts_map:
            self.counts_map[byte_pair] += increment
        else:
            self.counts_map[byte_pair] = increment

        if set_inverted_index:
            if increment in self.counts_inverted_index:
                self.counts_inverted_index[increment].append(byte_pair)
            else:
                self.counts_inverted_index[increment] = [byte_pair]


    def join(self, byte_pairs_counts: Self) -> None:
        for byte_pair, count in byte_pairs_counts.counts_map.items():
            self.increment_by(byte_pair, count, True)

        for count, byte_pairs in byte_pairs_counts.counts_inverted_index.items():
           if count in self.counts_inverted_index:
               self.counts_inverted_index[count].extend(byte_pairs)
           else:
               self.counts_inverted_index[count] = byte_pairs


    def compute_inverted_index(self):
        for byte_pair, count in self.counts_map.items():
            if count in self.counts_inverted_index:
                self.counts_inverted_index[count].append(byte_pair)
            else:
                self.counts_inverted_index[count] = [byte_pair]


    def remove(self, byte_pair: tuple[bytes, bytes]) -> None:
        count = self.counts_map[byte_pair]
        if count in self.counts_inverted_index:
            if len(self.counts_inverted_index[count]) > 1:
                self.counts_inverted_index[count].remove(byte_pair)
            else:
                del self.counts_inverted_index[count]

        del self.counts_map[byte_pair]


class TokenizerCache:
    def __init__(
        self,
        pretokens: dict[int, tuple[bytes]] = {},
        pretokens_counts: dict[int, int] = {},
        byte_pairs_pretokens: dict[tuple[bytes, bytes], list[int]] = {},
        byte_pairs_counts: dict[tuple[bytes, bytes], int] = {},
        byte_pairs_counts_inverted: dict[int, list[tuple[bytes,bytes]]] = {},
    ):
        self.pretokens: PretokensHashMap = PretokensHashMap(pretokens)
        self.pretokens_counts: PretokensCountsHashMap = PretokensCountsHashMap(pretokens_counts)
        self.byte_pairs_pretokens: BytePairsInvertedIndex = BytePairsInvertedIndex(byte_pairs_pretokens)
        self.byte_pairs_counts: BytePairsCountsMap = BytePairsCountsMap(byte_pairs_counts, byte_pairs_counts_inverted)


    def append(self, pretoken: tuple[bytes] | tuple[bytes, ...]):
        hash = self.pretokens.insert(pretoken)
        self.pretokens_counts.increment_by(hash)


    def compute_byte_pairs_from_pretokens(self):
        for hash, pretoken in self.pretokens.hash_map.items():
            pretoken_count = self.pretokens_counts.find(hash)

            for byte_1, byte_2 in zip(pretoken[:-1], pretoken[1:]):
                byte_pair = (byte_1, byte_2)
                self.byte_pairs_pretokens.insert(byte_pair, hash)
                self.byte_pairs_counts.increment_by(byte_pair, pretoken_count)

        self.byte_pairs_counts.compute_inverted_index()


    def get_top_byte_pair(self) -> tuple[int, tuple[bytes, bytes]]:
        max_count = max(self.byte_pairs_counts.counts_inverted_index.keys())
        top_pairs = self.byte_pairs_counts.counts_inverted_index[max_count]

        return (max_count, max(top_pairs))


    def merge_update(self, merge_byte_pair: tuple[bytes, bytes]):
        pretoken_hashes = self.byte_pairs_pretokens.find_hashes(merge_byte_pair)
        new_pairs, new_pairs_counts, absorbed_pairs_counts = self._get_affected_byte_pairs(
            pretoken_hashes, merge_byte_pair
        )
        self.byte_pairs_pretokens.join(new_pairs)
        self.byte_pairs_counts.join(new_pairs_counts)
        self.pretokens.merge_byte_pair(pretoken_hashes, merge_byte_pair)

        self.byte_pairs_counts.remove(merge_byte_pair)
        self.byte_pairs_pretokens.remove(merge_byte_pair)

        for absorbed_pair, count in absorbed_pairs_counts.counts_map.items():
            if absorbed_pair not in self.byte_pairs_counts.counts_map:
                continue

            current_pair_count = self.byte_pairs_counts.get_count(absorbed_pair)
            if current_pair_count >= count:
                self.byte_pairs_counts.remove(absorbed_pair)
            else:
                new_count = current_pair_count - count
                self.byte_pairs_counts.remove(absorbed_pair)
                self.byte_pairs_counts.increment_by(absorbed_pair, new_count, True)


    def _get_affected_byte_pairs(
        self,
        pretoken_hashes: list[int],
        merge_byte_pair: tuple[bytes, bytes]
    ) -> tuple[BytePairsInvertedIndex, BytePairsCountsMap, BytePairsCountsMap]:
        merge_bytes = b"".join(merge_byte_pair)
        new_byte_pairs = BytePairsInvertedIndex()
        new_byte_pairs_counts = BytePairsCountsMap()
        absorbed_byte_pairs_counts = BytePairsCountsMap()

        for hash in pretoken_hashes:
            pretoken = self.pretokens.find(hash)
            pretoken_count = self.pretokens_counts.find(hash)
            if len(pretoken) <= 2:
                continue

            last_byte_id = len(pretoken) - 2
            for i, bytes in enumerate(zip(pretoken[:-1], pretoken[1:])):
                if (bytes[0], bytes[1]) != merge_byte_pair:
                    continue

                if i == 0:
                    new_byte_pair = (merge_bytes, pretoken[i + 2])
                    new_byte_pairs.insert(new_byte_pair, hash)
                    new_byte_pairs_counts.increment_by(new_byte_pair, pretoken_count)

                    absorbed_byte_pairs_counts.increment_by((bytes[1], pretoken[i + 2]), pretoken_count)

                elif i == last_byte_id:
                    new_byte_pair = (pretoken[i - 1], merge_bytes)
                    new_byte_pairs.insert(new_byte_pair, hash)
                    new_byte_pairs_counts.increment_by(new_byte_pair, pretoken_count)

                    absorbed_byte_pairs_counts.increment_by((pretoken[i-1], bytes[0]), pretoken_count)
                else:
                    new_preceding_byte_pair = (pretoken[i - 1], merge_bytes)
                    new_byte_pairs.insert(new_preceding_byte_pair, hash)
                    absorbed_byte_pairs_counts.increment_by((pretoken[i-1], bytes[0]))

                    new_trailing_byte_pair = (merge_bytes, pretoken[i + 2])
                    new_byte_pairs.insert(new_trailing_byte_pair, hash)
                    absorbed_byte_pairs_counts.increment_by((bytes[1], pretoken[i + 2]))

        new_byte_pairs_counts.compute_inverted_index()
        return (new_byte_pairs, new_byte_pairs_counts, absorbed_byte_pairs_counts)
