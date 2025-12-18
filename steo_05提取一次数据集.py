import argparse
import pandas as pd
from step_04_加载数据 import ContrastiveCommunityDataset, INSTRUCTION

def build_triplets_dataframe(dataset: ContrastiveCommunityDataset) -> pd.DataFrame:
    rows = []
    for i in range(len(dataset)):
        ex = dataset[i]
        anchor, positive, negative = ex.texts
        meta = getattr(ex, "meta", {})
        rows.append({
            "anchor": anchor,
            "positive": positive,
            "negative": negative,
            "subreddit": meta.get("subreddit"),
            "rule": meta.get("rule"),
            "pos_id": meta.get("pos_id"),
            "neg_id": meta.get("neg_id"),
        })
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="community_cleaned.csv")
    parser.add_argument("--output", default="triplets.csv")
    parser.add_argument("--instruction_prefix", default=INSTRUCTION)
    parser.add_argument("--shuffle_pairs", action="store_true", default=True)
    args = parser.parse_args()

    ds = ContrastiveCommunityDataset(
        csv_path=args.input,
        instruction_prefix=args.instruction_prefix,
        shuffle_pairs=args.shuffle_pairs,
    )
    df = build_triplets_dataframe(ds)
    df.to_csv(args.output, index=False)

if __name__ == "__main__":
    main()
