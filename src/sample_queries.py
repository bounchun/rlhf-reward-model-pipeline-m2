"""Sample queries - how to load and query the processed dataset (run after step 9).

    python -m src.sample_queries            # prints the answers
    python -m src.sample_queries --show 2   # also prints 2 example pairs

Input : <main_root>/processed/<version>/{train,dev}.parquet   (never the holdout bucket:
        test/future data are only read at final evaluation / in M4)
Output: printed tables + <main_root>/audit/<version>/sample_queries.json

Each query is plain pandas on the Parquet files, so a reviewer can copy any of them into a
notebook. They double as sanity checks on the stored data and as a first look for M2:
  Q1  rows, groups and label balance per split
  Q2  subset mix per split (is dev representative of train?)
  Q3  conversation depth: distribution of num_turns
  Q4  length bias: how often is the preferred reply the longer one? (per subset)
  Q5  refusals: share of pairs with a refusal, and how often the refusal is preferred (per subset)
  Q6  long outliers per split (threshold fitted on train in step 7)
  Q7  one example pair, looked up by pair_id
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .common import join, read_parquet, sha256_bytes, step_args, storage_roots, write_bytes


def load(cfg: dict) -> pd.DataFrame:
    root, _ = storage_roots(cfg)
    version = cfg["dataset"]["version"]
    parts = [read_parquet(join(root, "processed", version, f"{s}.parquet")) for s in ("train", "dev")]
    df = pd.concat(parts, ignore_index=True)
    df["subset"] = df["subset"].astype(str)
    df["split"] = df["split"].astype(str)
    return df


def run_queries(df: pd.DataFrame) -> dict:
    preferred_len = np.where(df["label"] == 1, df["len_a_words"], df["len_b_words"])
    other_len = np.where(df["label"] == 1, df["len_b_words"], df["len_a_words"])
    pref_refusal = np.where(df["label"] == 1, df["refusal_a"], df["refusal_b"])
    any_refusal = df["refusal_a"] | df["refusal_b"]
    one_refusal = df["refusal_a"] ^ df["refusal_b"]
    d = df.assign(preferred_longer=preferred_len > other_len, any_refusal=any_refusal,
                  refusal_preferred=pref_refusal)

    q = {}
    q["Q1_size_and_label_balance"] = d.groupby("split").agg(
        pairs=("pair_id", "size"), groups=("group_id", "nunique"), label_1_share=("label", "mean")).round(4)
    q["Q2_subset_mix"] = pd.crosstab(d["split"], d["subset"], normalize="index").round(4)
    q["Q3_num_turns"] = d["num_turns"].clip(upper=5).value_counts(normalize=True).sort_index().round(4) \
        .rename(index={5: "5+"}).to_frame("share")
    q["Q4_length_bias"] = d.groupby("subset").agg(
        pairs=("pair_id", "size"), share_preferred_is_longer=("preferred_longer", "mean")).round(4)
    q["Q5_refusals"] = d.groupby("subset").agg(share_any_refusal=("any_refusal", "mean")).join(
        d[one_refusal].groupby("subset").agg(share_refusal_preferred_when_one_side=("refusal_preferred", "mean"))
    ).round(4)
    q["Q6_long_outliers"] = d.groupby("split").agg(share_long_outlier=("is_long_outlier", "mean")).round(4)
    return q


def main(argv: list[str] | None = None) -> None:
    def extra(ap):
        ap.add_argument("--show", type=int, default=1, help="number of example pairs to print (Q7)")

    args, cfg = step_args(__doc__, argv, extra)
    df = load(cfg)
    results = run_queries(df)
    for name, table in results.items():
        print(f"\n== {name}\n{table.to_string()}")

    print("\n== Q7_example_pairs (looked up by pair_id)")
    for pid in df["pair_id"].sort_values().head(args.show):
        row = df.loc[df["pair_id"] == pid].iloc[0]
        print(f"pair_id={pid[:12]}…  subset={row['subset']}  label={row['label']}")
        print(f"  context   : {row['context'][:200]!r}")
        print(f"  response_a: {row['response_a'][:150]!r}")
        print(f"  response_b: {row['response_b'][:150]!r}")

    out = {k: json.loads(v.to_json(orient="index")) for k, v in results.items()}
    data = (json.dumps(out, indent=2) + "\n").encode()
    url = join(storage_roots(cfg)[0], "audit", cfg["dataset"]["version"], "sample_queries.json")
    write_bytes(url, data)
    print(f"\nSaved -> {url} (sha256 {sha256_bytes(data)[:12]}…)")


if __name__ == "__main__":
    main()
