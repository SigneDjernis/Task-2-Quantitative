"""
Course training script (simplified from nanoGPT).

Focus:
- Train a small GPT-style model from scratch on a tiny dataset.
- Students will integrate sustainability tracking themselves.

Source: https://github.com/karpathy/nanoGPT
"""

import os
import time
import pickle
from dataclasses import asdict
import numpy as np
import torch
from codecarbon import EmissionsTracker

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Experiment configuration


n_layer_sweep = np.linspace(1, 12, 10, dtype=int) 
n_head_sweep  = np.array([1, 2, 4, 8, 16, 32, 64, 128])  # all divide 128 ✓
n_embd_sweep  = (np.linspace(64, 512, 10, dtype=int) // 32 * 32)    
batch_size_sweep =[1,2,4,8,16,32,64,128,256,512]



# I/O
OUT_DIR = "out"
DATA_DIR = os.path.join("data")
EVAL_INTERVAL = 200     
EVAL_ITERS = 50
LOG_INTERVAL = 50
SAVE_CHECKPOINT = True
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

def load_meta(data_dir: str):
    meta_path = os.path.join(data_dir, "meta.pkl")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "rb") as f:
        return pickle.load(f)

def get_batch(split: str, data_dir: str, block_size: int, batch_size: int, device: str):
    # simple, robust memmap loader
    bin_path = os.path.join(data_dir, f"{split}.bin")
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")

    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])

    x = x.to(device)
    y = y.to(device)
    return x, y

for val in [64,128,256,512]:
    print("-------------Model with",val,"batchsize")
    # Model (main tunables)
    N_LAYER = 4
    N_HEAD = 4
    N_EMBD = 128
    DROPOUT = 0.1
    BIAS = True

    # Training (main parameters you can also experiment with)
    SEED = 1
    DEVICE = "cpu"          # If you can, try also seeing consumption when using gpu (change this to 'cuda' if torch.cuda.is_available() else 'cpu')
    DTYPE = "float32"       
    BATCH_SIZE = val       # Number of sequences processed in parallel.
    BLOCK_SIZE = 256        # Maximum context length for predictions (e.g. 128 or 256). The longer the block size, the more memory and compute it requires, but it can also lead to better performance.
    MAX_ITERS = 1000        # Total number of training iterations. The more iterations, the better the model can perform, but it also takes more time and energy to train.
    LEARNING_RATE = 3e-4    # the standard starting learning rate, often good enough for a first try
    WEIGHT_DECAY = 0.1      # L2 Regularization
    GRAD_CLIP = 1.0         # To prevent exploding gradients

    # -----------------------------------------------------------------------------

    @torch.no_grad()
    def estimate_loss(model: GPT, data_dir: str, block_size: int, batch_size: int, device: str, eval_iters: int):    
        model.eval()
        losses = {}
        for split in ["train", "val"]:
            split_losses = torch.zeros(eval_iters, device=device)
            for k in range(eval_iters):
                x, y = get_batch(split, data_dir, block_size, batch_size, device)
                _, loss = model(x, y)
                split_losses[k] = loss
            losses[split] = split_losses.mean().item()
        model.train()
        return losses

    def save_checkpoint(out_dir: str, model: GPT, optimizer: torch.optim.Optimizer, iter_num: int, config: dict):
        os.makedirs(out_dir, exist_ok=True)
        ckpt = {
            "iter_num": iter_num,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "config": config,
        }
        torch.save(ckpt, os.path.join(out_dir, "ckpt.pt"))

    def main():
        os.makedirs(OUT_DIR, exist_ok=True)
        set_seed(SEED)

        meta = load_meta(DATA_DIR)
        vocab_size = meta["vocab_size"] if meta and "vocab_size" in meta else 50304

        cfg = GPTConfig(
            block_size=BLOCK_SIZE,
            vocab_size=vocab_size,
            n_layer=N_LAYER,
            n_head=N_HEAD,
            n_embd=N_EMBD,
            dropout=DROPOUT,
            bias=BIAS,
        )

        # create the model and move it to the device
        model = GPT(cfg).to(DEVICE)

        # create the optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            betas=(0.9, 0.95),
        )

        # (optional) uncomment this for printing model size once
        # print(f"Device: {DEVICE}")
        # print(f"Model parameters: {model.get_num_params():,}")
        # print(f"Training for {MAX_ITERS} iterations | batch={BATCH_SIZE} | block={BLOCK_SIZE}")

        t0 = time.time()
        tracker = EmissionsTracker(save_to_file=False)
        tracker.start()
    
        for it in range(MAX_ITERS + 1):
            # periodic evaluation
            if it % EVAL_INTERVAL == 0:
                losses = estimate_loss(model, DATA_DIR, BLOCK_SIZE, BATCH_SIZE, DEVICE, EVAL_ITERS)
                dt = time.time() - t0
                print(f"iter {it:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | elapsed {dt:.1f}s")

                if SAVE_CHECKPOINT and it > 0:
                    config_dump = {
                        "data_dir": DATA_DIR,
                        "train": {
                            "batch_size": BATCH_SIZE,
                            "block_size": BLOCK_SIZE,
                            "max_iters": MAX_ITERS,
                            "learning_rate": LEARNING_RATE,
                            "weight_decay": WEIGHT_DECAY,
                            "grad_clip": GRAD_CLIP,
                            "dtype": DTYPE,
                            "device": DEVICE,
                        },
                        "model": asdict(cfg),
                    }
                    save_checkpoint(OUT_DIR, model, optimizer, it, config_dump)

            # training step
            x, y = get_batch("train", DATA_DIR, BLOCK_SIZE, BATCH_SIZE, DEVICE)
            _, loss = model(x, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if GRAD_CLIP and GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            optimizer.step()

            if it % LOG_INTERVAL == 0:
                print(f"iter {it:5d} | loss {loss.item():.4f}")

        print("Training completed.")
        
        ## DATA WE WANT FOR THE REPORT:
        # Final training time
        total_time = time.time() - t0
        tracker.stop()
        em_data = tracker._prepare_emissions_data()


        # Get number of parameters
        num_params = model.get_num_params()

        # Save results to csv
        csv_file = "results_train_batch.csv"
        file_exists = os.path.isfile(csv_file)

        # Open in append mode
        with open(csv_file, "a") as f:
            if not file_exists:
                f.write("total_time_seconds,N_LAYER,N_HEAD,N_EMBD,BATCH_SIZE,num_params,emission\n")
            f.write(f"{total_time:.2f},{N_LAYER},{N_HEAD},{N_EMBD},{BATCH_SIZE},{num_params},{em_data.energy_consumed:.6f}\n")

        # Save final checkpoint
        if SAVE_CHECKPOINT:
            config_dump = {
                "data_dir": DATA_DIR,
                "train": {
                    "batch_size": BATCH_SIZE,
                    "block_size": BLOCK_SIZE,
                    "max_iters": MAX_ITERS,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "grad_clip": GRAD_CLIP,
                    "dtype": DTYPE,
                    "device": DEVICE,
                },
                "model": asdict(cfg),
            }
            save_checkpoint(OUT_DIR, model, optimizer, MAX_ITERS, config_dump)

    if __name__ == "__main__":
        main()
