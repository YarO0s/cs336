import os
from collections.abc import Generator

TINYSTORIES = "TinyStories"
CHUNK_SIZE = 8192


class FileLoader:
    def __init__(self, encoding: str = "UTF-8", delimeter: str = "\n"):
        self.encoding: str = encoding
        self.delimeter: str = delimeter

    def all_rows(self, filepath: str) -> list[str]:
        if not os.path.exists(filepath):
            raise Exception(f"File {filepath} does not exist")

        with open(filepath, encoding=self.encoding) as file:
            contents = file.read()
            return contents.split(self.delimeter)

    def iterate_rows(self, filepath: str, batch_size: int = 100) -> Generator[str | list[str]]:
        if not os.path.exists(filepath):
            raise Exception(f"File {filepath} does not exist")

        if batch_size < 1:
            raise Exception("batch_size can't be 0")

        with open(filepath, encoding=self.encoding) as file:
            lines: list[str] = []
            row_part: str = ""
            chunk = file.read(CHUNK_SIZE)

            while chunk:
                if self.delimeter not in chunk:
                    row_part += chunk
                else:
                    if row_part != "":
                        chunk = row_part + chunk
                        row_part = ""

                    if not chunk.endswith(self.delimeter):
                        last_occurence = chunk.rfind(self.delimeter)
                        row_part = chunk[last_occurence + len(self.delimeter) :]
                        chunk = chunk[0:last_occurence]

                    lines.extend(chunk.split(self.delimeter))

                chunk = file.read(CHUNK_SIZE)

                if len(lines) >= batch_size:
                    yield lines
                    lines = []

            if len(lines) <= 0:
                yield lines


class TinyStoriesDataset(FileLoader):
    def __init__(self, train_filepath: str, test_filepath: str) -> None:
        if not os.path.exists(train_filepath):
            raise Exception(f"File {train_filepath} does not exist")

        if not os.path.exists(test_filepath):
            raise Exception(f"File {test_filepath} does not exist")

        super().__init__("UTF-8", "<|endoftext|>")

        self.train_filepath: str = train_filepath
        self.test_filepath: str = test_filepath


class BPEDemo(FileLoader):
    def __init__(self, demo_filepath: str) -> None:
        if not os.path.exists(demo_filepath):
            raise Exception(f"File {demo_filepath} does not exist")

        super().__init__("UTF-8", "<|endoftext|>")

        self.demo_filepath: str = demo_filepath
