"""STEP 7 - Train / dev / test split + statistics fitted on train only.

    python -m src.step7_split

Input : <interim_dir>/<version>/working_set.parquet, future.parquet   (from step 6)
Output: <interim_dir>/<version>/splits/{train,dev,test,future_f1,future_f2}.parquet
        manifest: steps.step7_split

Split (on groups, never rows):  b2 = sha256(group_id + tdt_salt) % 10
    b2 in [0,8) -> train (~80%)   b2 == 8 -> dev (~10%)   b2 == 9 -> test (~10%)

Leakage controls
- every group_id must be in exactly one of train/dev/test/future_f1/future_f2 (asserted)
- is_long_outlier: threshold = 99th percentile of reply length computed on TRAIN ONLY,
  then applied to every split (Lecture 2: split -> fit on train -> transform all)
- per-split subset mix and label balance are recorded so reviewers can check the splits
  are representative of each other
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import (OUTPUT_COLUMNS, SPLITS, hash_bucket, interim, read_parquet, require, step_args,
                     update_manifest, write_parquet)


def split_stats(df: pd.DataFrame) -> dict:
    if not len(df):
        return {"pairs": 0}
    return {
        "pairs": int(len(df)),
        "groups": int(df["group_id"].nunique()),
        "subset_share": {k: round(float(v), 4) for k, v in df["subset"].value_counts(normalize=True).sort_index().items()},
        "label_1_share": round(float(df["label"].mean()), 4),
        "long_outlier_share": round(float(df["is_long_outlier"].mean()), 4),
        "refusal_any_share": round(float((df["refusal_a"] | df["refusal_b"]).mean()), 4),
    }


def main(argv: list[str] | None = None) -> None:
    _, cfg = step_args(__doc__, argv)
    s = cfg["split"]
    for f in ["working_set.parquet", "future.parquet"]:
        require(interim(cfg, f), "step6_filter")
    ws = read_parquet(interim(cfg, "working_set.parquet"))
    future = read_parquet(interim(cfg, "future.parquet"))

    b2 = ws["group_id"].map(lambda g: hash_bucket(g, s["tdt_salt"], 10))
    ws["split"] = np.select([(b2 >= s["train_buckets"][0]) & (b2 < s["train_buckets"][1]),
                             (b2 >= s["dev_buckets"][0]) & (b2 < s["dev_buckets"][1])],
                            ["train", "dev"], default="test")
    df = pd.concat([ws, future], ignore_index=True)

    assert (df.groupby("group_id")["split"].nunique() == 1).all(), "group_id overlap between splits"

    train = df[df["split"] == "train"]
    lens = np.concatenate([train["len_a_words"].to_numpy(), train["len_b_words"].to_numpy()])
    thr = float(np.percentile(lens, cfg["features"]["long_outlier_percentile"]))
    df["is_long_outlier"] = np.maximum(df["len_a_words"], df["len_b_words"]) > thr

    stats = {}
    for split in SPLITS:
        part = df[df["split"] == split][OUTPUT_COLUMNS]
        write_parquet(part, interim(cfg, "splits", f"{split}.parquet"))
        stats[split] = split_stats(part)
        print(f"  {split:10s} {len(part):7,d} pairs")

    update_manifest(cfg, "step7_split", {
        "leakage_check": "passed: every group_id is in exactly one split",
        "train_fitted": {"long_outlier_percentile": cfg["features"]["long_outlier_percentile"],
                         "long_outlier_threshold_words": thr},
        "splits": stats,
        "params": {k: s[k] for k in ["tdt_salt", "train_buckets", "dev_buckets", "test_buckets"]},
    })
    print(f"Long-outlier threshold (train only): {thr:.1f} words")


if __name__ == "__main__":
    main()
