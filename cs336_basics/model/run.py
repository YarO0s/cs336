import argparse
import sys
import torch
import aiohttp_cors

import numpy as np

from cs336_basics.tokenizers.bpe import Tokenizer
from aiohttp import web

async def serve(request):
    input = await request.json()

    encoded_input = torch.tensor(bpe.encode(input["prompt"]), device=device)

    result = input["prompt"]
    for out in model.infer(encoded_input, 256):
        out_list = out.tolist()
        out_decoded = bpe.decode([out_list])

        if out_decoded == "<|endoftext|>":
            response = web.json_response({"success": True, "response": result})
            return response

        result += out_decoded

    response = web.json_response({"success": True, "response": result})
    return response

device = "cuda:0"
model_file = "/mnt/c/projects/datasets/models/tinystories-gpt-1/model_final_delfino_warp.pt"
model_weights = "/mnt/c/projects/datasets/models/tinystories-gpt-1/check_final_delfino_warp"
vocab_file = "/mnt/c/projects/stanford-cs336/assignment1-basics/.results/TinyStories-train.pickle"

model = torch.load(model_file, weights_only=False)
model.load_state_dict(torch.load(model_weights)["model"])
bpe = Tokenizer.from_files(vocab_file, vocab_file, ["<|endoftext|>"])
print("Model initialized")

app = web.Application()
cors = aiohttp_cors.setup(app, defaults={
    "*": aiohttp_cors.ResourceOptions(
        allow_credentials=True,
        expose_headers="*",
        allow_headers="*",
        allow_methods=["GET", "POST", "PUT", "DELETE"]
    )
})
app.add_routes([web.post("/serve", serve)])

for route in list(app.router.routes()):
    cors.add(route)

if __name__ == "__main__":
    web.run_app(app)
