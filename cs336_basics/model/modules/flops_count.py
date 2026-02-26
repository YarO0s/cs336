from time import time

import torch

from cs336_basics.model.modules.func import softmax
from cs336_basics.model.modules.modules import TransformerLM

vocab_size = 5025
context_length = 1024
num_layers = 48
d_model = 1600
num_heads = 25
d_ff = 6400
dtype_bytes = 4

device = "cuda:0"
transformer = TransformerLM(vocab_size, d_model, context_length, num_heads, d_ff, num_layers, 10000, device=device)

one = torch.rand((vocab_size, context_length), device=device)
two = torch.rand((context_length, vocab_size), device=device)
_ = torch.matmul(one, two)
torch.cuda.synchronize()

input = torch.randint(1, vocab_size, (4, 12), device=device)
output = transformer.forward(input)
print(input.shape)
output, _ = torch.sort(softmax(output, -1), -1, True)
print(output[:, :, :10])
