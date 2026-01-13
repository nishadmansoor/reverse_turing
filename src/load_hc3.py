import argparse
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

def clean(x):
    return " ".join((x or "").split())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/hc3.csv")
    ap.add_argument("--max_rows", type=int, default=5000)
    args = ap.parse_args()

    ds = load_dataset("Hello-SimpleAI/HC3")

    # pick first available split
    split = ds[list(ds.keys())[0]]

    rows = []
    for ex in tqdm(split):
        q = clean(ex.get("question", ""))
        if not q:
            continue

        for a in ex.get("human_answers", []):
            rows.append({"text": f"Q: {q}\nA: {clean(a)}", "label": 0})
        for a in ex.get("chatgpt_answers", []):
            rows.append({"text": f"Q: {q}\nA: {clean(a)}", "label": 1})

        if len(rows) >= args.max_rows:
            break

    df = pd.DataFrame(rows).dropna()
    n = min((df.label == 0).sum(), (df.label == 1).sum())
    df = pd.concat([
        df[df.label == 0].sample(n, random_state=42),
        df[df.label == 1].sample(n, random_state=42),
    ]).sample(frac=1, random_state=42)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows → {args.out}")

if __name__ == "__main__":
    main()