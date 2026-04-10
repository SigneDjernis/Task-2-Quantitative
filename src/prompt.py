"""
Inference / prompting script (Tiny Shakespeare, char-level).
Students will integrate sustainability tracking themselves.

Source: https://github.com/karpathy/nanoGPT
"""

import os
import pickle
import torch
import time

from model import GPT, GPTConfig

# ----------------------------
# Edit these
# ----------------------------
OUT_DIR = "out"
CKPT_PATH = os.path.join(OUT_DIR, "ckpt.pt")

PROMPT = "To be, or not to be"
MAX_NEW_TOKENS = 200
TEMPERATURE = 1.0
TOP_K = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------


def load_meta(data_dir: str):
    meta_path = os.path.join(data_dir, "meta.pkl")
    with open(meta_path, "rb") as f:
        return pickle.load(f)


def main():
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    # train.py should store config with model parameters and data_dir
    data_dir = ckpt["config"]["data_dir"]
    model_cfg = ckpt["config"]["model"]

    meta = load_meta(data_dir)
    stoi = meta["stoi"]         # char to index mapping
    itos = meta["itos"]         # index to char mapping

    def encode(s: str):
        # map unknown chars to a safe fallback if needed
        return [stoi.get(ch, stoi[" "]) for ch in s]

    def decode(tokens):
        return "".join([itos[t] for t in tokens])

    config = GPTConfig(**model_cfg)
    model = GPT(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    idx = torch.tensor([encode(PROMPT)], dtype=torch.long, device=DEVICE)

    out = model.generate(
        idx,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K
    )

    print(decode(out[0].tolist()))


if __name__ == "__main__":
    t0 = time.time()
    main()

    ## DATA WE WANT FOR THE REPORT:
    # Final training time
    total_time = time.time() - t0

    # Number of tokens in the prompt
    num_tokens_prompt = len(PROMPT)

    # Save results to csv
    csv_file = "results_prompt.csv"
    file_exists = os.path.isfile(csv_file)

    # Open in append mode
    with open(csv_file, "a") as f:
        if not file_exists:
            f.write("total_time_seconds,MAX_NEW_TOKENS,num_tokens_prompt\n")
        f.write(f"{total_time:.2f},{MAX_NEW_TOKENS},{num_tokens_prompt}\n")
