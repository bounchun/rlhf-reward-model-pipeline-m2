"""STEP 6 - Reserve future data, then filter (down-sample) to the working set.

    python -m src.step6_filter

Input : <interim_dir>/<version>/features.parquet   (from step 5)
Output: <interim_dir>/<version>/working_set.parquet  (~working_set_size pairs, for train/dev/test)
        <interim_dir>/<version>/future.parquet       (F1 + F2 reserve, column `split`)
        manifest: steps.step6_filter

Why filter: hh-rlhf has ~160k usable pairs; ~30k is enough for stable dev/test estimates and
fits the course compute budget. Everything is done on whole GROUPS, never single rows.

1. Future reserve (Lecture 2: "keep some data unseen to simulate future data")
     b1 = sha256(group_id + future_salt) % 100
     b1 in [0,10) -> future_f1 ; b1 in [10,20) -> future_f2 ; rest -> development pool
2. Working set: groups ordered by sha256(group_id + sample_salt); whole groups are taken
   until each subset reaches its proportional quota (stratified by subset).
   Groups spanning several subsets count toward their majority subset.
No random number generator is used: the result does not depend on row order or library versions.
"""
from __future__ import annotations

import argparse

import pandas as pd

from .common import hash_bucket, interim, read_parquet, require, sha256_text, step_args, update_manifest, write_parquet


def reserve_and_filter(df: pd.DataFrame, cfg: dict, working_set_size: int):
    s = cfg["split"]
    # group size + majority subset (ties -> alphabetically first subset), vectorised
    counts = df.groupby(["group_id", "subset"]).size().rename("n").reset_index()
    counts = counts.sort_values(["group_id", "n", "subset"], ascending=[True, False, True])
    groups = counts.drop_duplicates("group_id")[["group_id", "subset"]].merge(
        df.groupby("group_id").size().rename("size").reset_index(), on="group_id")
    b1 = groups["group_id"].map(lambda g: hash_bucket(g, s["future_salt"], 100))
    f1_lo, f1_hi = s["future_f1_buckets"]
    f2_lo, f2_hi = s["future_f2_buckets"]
    groups["part"] = "pool"
    groups.loc[(b1 >= f1_lo) & (b1 < f1_hi), "part"] = "future_f1"
    groups.loc[(b1 >= f2_lo) & (b1 < f2_hi), "part"] = "future_f2"

    pool = groups[groups["part"] == "pool"].copy()
    pool["order"] = pool["group_id"].map(lambda g: sha256_text(g + s["sample_salt"]))
    pool = pool.sort_values("order")
    share = pool.groupby("subset")["size"].sum() / pool["size"].sum()
    quota = (share * working_set_size).round().astype(int)
    pool["cum"] = pool.groupby("subset")["size"].cumsum()
    ws_groups = set(pool.loc[(pool["cum"] - pool["size"]) < pool["subset"].map(quota), "group_id"])

    part = df["group_id"].map(groups.set_index("group_id")["part"])
    working = df[part.eq("pool") & df["group_id"].isin(ws_groups)].copy()
    future = df[part.isin(["future_f1", "future_f2"])].copy()
    future["split"] = part[future.index]
    unused = int((part.eq("pool") & ~df["group_id"].isin(ws_groups)).sum())
    return working, future, unused, quota


def main(argv: list[str] | None = None) -> None:
    def extra(ap: argparse.ArgumentParser):
        ap.add_argument("--working-set-size", type=int, default=None, help="override config (demo/tests)")

    args, cfg = step_args(__doc__, argv, extra)
    size = args.working_set_size or cfg["split"]["working_set_size"]
    src = interim(cfg, "features.parquet")
    require(src, "step5_features")
    df = read_parquet(src)
    working, future, unused, quota = reserve_and_filter(df, cfg, size)
    write_parquet(working, interim(cfg, "working_set.parquet"))
    write_parquet(future, interim(cfg, "future.parquet"))
    update_manifest(cfg, "step6_filter", {
        "working_set_size_target": size,
        "working_set_pairs": int(len(working)),
        "subset_quota": {k: int(v) for k, v in quota.items()},
        "future_f1_pairs": int((future["split"] == "future_f1").sum()),
        "future_f2_pairs": int((future["split"] == "future_f2").sum()),
        "unused_pool_pairs": unused,
    })
    print(f"Working set {len(working):,} pairs (target {size:,}); future reserve {len(future):,}; "
          f"unused pool {unused:,}")


if __name__ == "__main__":
    main()
