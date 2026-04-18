"""
Inference / prompting script (Tiny Shakespeare, char-level).
Students will integrate sustainability tracking themselves.

Source: https://github.com/karpathy/nanoGPT
"""

import os
import pickle
import torch
import time
from codecarbon import EmissionsTracker
from model import GPT, GPTConfig

prompt_long = """To be, or not to be, that is the question:
Whether tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause—there's the respect
That makes calamity of so long life.
For who would bear the whips and scorns of time,
Th'oppressor's wrong, the proud man's contumely,
The pangs of dispriz'd love, the law's delay,
The insolence of office, and the spurns
That patient merit of th'unworthy takes,
When he himself might his quietus make
With a bare bodkin? Who would fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscovere'd country, from whose bourn
No traveller returns, puzzles the will,
And makes us rather bear those ills we have
Than fly to others that we know not of?
Thus conscience doth make cowards of us all,
And thus the native hue of resolution
Is sicklied o'er with the pale cast of thought,
And enterprises of great pith and moment
With this regard their currents turn awry
And lose the name of action."""

lengths = [5, 10, 19, 32, 64, 128, 256, 512, 1024, 1536]
len_sweep=[5,10,50,100,200,400,600,800,1000,2000]
subsections = []
for L in lengths:
    chunk = prompt_long[:L]
    subsections.append(chunk)

# ----------------------------
# Edit these
# ----------------------------

OUT_DIR = "out"
CKPT_PATH = os.path.join(OUT_DIR, "ckpt.pt")
for val in len_sweep: 
    
    PROMPT = subsections[2]
    MAX_NEW_TOKENS = val
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
        tracker = EmissionsTracker(save_to_file=False)
        tracker.start()
        main()

        ## DATA WE WANT FOR THE REPORT:
        # Final training time
        total_time = time.time() - t0
        tracker.stop()
        em_data = tracker._prepare_emissions_data()

        # Number of tokens in the prompt
        num_tokens_prompt = len(PROMPT)

        # Save results to csv
        csv_file = "results_max_new.csv"
        file_exists = os.path.isfile(csv_file)

        # Open in append mode
        with open(csv_file, "a") as f:
            if not file_exists:
                f.write("total_time_seconds,MAX_NEW_TOKENS,num_tokens_prompt,emmision\n")
            f.write(f"{total_time:.2f},{MAX_NEW_TOKENS},{num_tokens_prompt},{em_data.energy_consumed:.6f}\n")
