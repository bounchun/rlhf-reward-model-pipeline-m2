"""STEP 9 - Store the preprocessed data (final validation, upload, manifest).

    python -m src.step9_store_processed

Input : <interim_dir>/<version>/splits/*.parquet   (from step 7)
        <interim_dir>/<version>/shards/*.parquet   (from step 8, if enabled)
Output: <main_root>/processed/<version>/train.parquet, dev.parquet
        <main_root>/processed/<version>/train_shards/*.parquet        (if sharding enabled)
        <holdout_root>/<version>/test.parquet, future_f1.parquet, future_f2.parquet
        <main_root>/manifests/<version>.json  + manifests/<version>.json (commit this to Git)

- Test and future data go ONLY to the separate holdout bucket (training identity has no access).
- Files are written deterministically (rows sorted by pair_id, fixed column types), so the
  same config + code + raw revision gives byte-identical files; SHA-256 of every file is recorded.
- Final validation before upload: schema, no group or pair overlap across splits.
"""
from __future__ import annotations

import pandas as pd

from .common import (OUTPUT_COLUMNS, SPLITS, interim, join, manifest_path, parquet_bytes, read_bytes,
                     read_parquet, require, sha256_bytes, step_args, storage_roots, update_manifest, write_bytes)


def final_bytes(df: pd.DataFrame, subsets: list[str]) -> bytes:
    df = df[OUTPUT_COLUMNS].sort_values("pair_id").reset_index(drop=True).copy()
    df["subset"] = pd.Categorical(df["subset"], categories=subsets)
    df["split"] = pd.Categorical(df["split"], categories=SPLITS)
    for col in ["pair_id", "group_id", "context", "response_a", "response_b"]:
        df[col] = df[col].astype("string[pyarrow]")
    return parquet_bytes(df)


def validate(frames: dict[str, pd.DataFrame]) -> None:
    seen_groups, seen_pairs = {}, set()
    for split, df in frames.items():
        if list(df.columns) != OUTPUT_COLUMNS:
            raise SystemExit(f"ERROR: unexpected columns in {split}")
        for g in df["group_id"].unique():
            if g in seen_groups:
                raise SystemExit(f"ERROR: group {g[:10]} in both {seen_groups[g]} and {split}")
            seen_groups[g] = split
        if seen_pairs & set(df["pair_id"]):
            raise SystemExit(f"ERROR: pair overlap involving {split}")
        seen_pairs |= set(df["pair_id"])


def main(argv: list[str] | None = None) -> None:
    _, cfg = step_args(__doc__, argv)
    version, subsets = cfg["dataset"]["version"], list(cfg["dataset"]["subsets"])
    root, holdout = storage_roots(cfg)

    frames = {}
    for split in SPLITS:
        p = interim(cfg, "splits", f"{split}.parquet")
        require(p, "step7_split")
        frames[split] = read_parquet(p)
    validate(frames)

    targets = {"train": join(root, "processed", version, "train.parquet"),
               "dev": join(root, "processed", version, "dev.parquet"),
               "test": join(holdout, version, "test.parquet"),
               "future_f1": join(holdout, version, "future_f1.parquet"),
               "future_f2": join(holdout, version, "future_f2.parquet")}
    files = {}
    for split, url in targets.items():
        data = final_bytes(frames[split], subsets)
        write_bytes(url, data)
        files[split] = {"uri": url, "sha256": sha256_bytes(data), "bytes": len(data), "rows": len(frames[split])}
        print(f"  {split:10s} {len(frames[split]):7,d} pairs -> {url}")

    shard_dir = interim(cfg, "shards")
    for p in sorted(shard_dir.glob("*.parquet")) if shard_dir.exists() else []:
        url = join(root, "processed", version, "train_shards", p.name)
        data = read_bytes(p)
        write_bytes(url, data)
        files[f"train_shards/{p.name}"] = {"uri": url, "sha256": sha256_bytes(data)}
        print(f"  shard      {p.name} -> {url}")

    update_manifest(cfg, "step9_store_processed", {"final_validation": "passed", "files": files})
    write_bytes(join(root, "manifests", f"{version}.json"), manifest_path(cfg).read_bytes())
    print(f"Manifest -> {manifest_path(cfg)} (commit this file) and {join(root, 'manifests')}")


if __name__ == "__main__":
    main()
