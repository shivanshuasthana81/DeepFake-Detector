# scripts/create_splits.py
import os
import csv
import random
import argparse

def collect_samples(out_root):
    samples = []
    for cls, label in [("real",0), ("fake",1)]:
        cls_root = os.path.join(out_root, cls)
        if not os.path.isdir(cls_root):
            continue
        for vid in os.listdir(cls_root):
            folder = os.path.join(cls_root, vid)
            if os.path.isdir(folder):
                frames = [f for f in os.listdir(folder) if f.lower().endswith(".jpg")]
                if len(frames) >= 4:  # require at least 4 frames
                    samples.append((folder, label))
    return samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    samples = collect_samples(args.out_root)
    random.shuffle(samples)
    n = len(samples)
    ntrain = int(n * args.train_frac)
    nval = int(n * args.val_frac)
    train = samples[:ntrain]
    val = samples[ntrain:ntrain+nval]
    test = samples[ntrain+nval:]
    out_csv = os.path.join(args.out_root, "splits.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["folder","label","split"])
        for s in train:
            writer.writerow([s[0], s[1], "train"])
        for s in val:
            writer.writerow([s[0], s[1], "val"])
        for s in test:
            writer.writerow([s[0], s[1], "test"])
    print("Wrote", out_csv, "with", len(samples), "samples.")
