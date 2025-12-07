import os

from cs336_basics.tokenizers import bpe
from textual.app import App
from textual.widgets import Label, DataTable

class EventsApp(App):
    BINDINGS = [
        ("w", "up", "Up"),
        ("s", "down", "Down"),
    ]

    def set_data(self, filepath: str):
        vocab, _ = bpe.load(filepath)
        self.vocab = vocab
        self.chunk = 50
        self.start = 0
        self.end = min(self.chunk, len(vocab))


    def compose(self):
        yield Label(f"Vocab size: {len(self.vocab)}", id="static_size")
        self.chunk_label = Label(f"Displaying: {self.start}:{self.end}", id="static_chunk")
        yield self.chunk_label
        yield DataTable()


    def on_mount(self):
        table = self.query_one(DataTable)
        table.add_columns("token", "id")

        for token in list(self.vocab.items())[self.start:self.end]:
            table.add_row(token[1], token[0])


    def action_up(self):
        table = self.query_one(DataTable)

        self.start = min(self.start + self.chunk, len(self.vocab) - self.chunk)
        self.end = min(self.end + self.chunk, len(self.vocab))

        table.clear()
        for token in list(self.vocab.items())[self.start:self.end]:
            table.add_row(token[1], token[0])

        self.chunk_label.content = f"Displaying: {self.start}:{self.end}"


    def action_down(self):
        table = self.query_one(DataTable)

        self.start = max(self.start - self.chunk, 0)
        self.end = max(self.end - self.chunk, self.chunk)
        table.clear()
        for token in list(self.vocab.items())[self.start:self.end]:
            table.add_row(token[1], token[0])

        self.chunk_label.content = f"Displaying: {self.start}:{self.end}"

if __name__ == "__main__":
    CWD = os.getcwd()
    app = EventsApp()
    app.set_data("C:\\projects\\stanford-cs336\\assignment1-basics\\.results\\TinyStories-valid.pickle")
    app.run()
