"""STEP 8 - (Optional) shard the training set, following the "Data shards" lab.

    python -m src.step8_shard

Input : <interim_dir>/<version>/splits/train.parquet   (from step 7)
Output: <interim_dir>/<version>/shards/train-0000i-of-0000N.parquet   (only if num_shards > 1)
        manifest: steps.step8_shard

Config: "shards": {"num_shards": N, "batch_size": B, "salt": ...}
- num_shards <= 1 (default): sharding is SKIPPED. At ~24k training pairs (well under 1 GB) a
  single Parquet file is faster to read than several, and sharding would only create the
  many-small-files problem from Lecture 2. The step still runs and records the decision.
- num_shards > 1 (e.g. for multi-worker / DDP loading in M3): rows are put in a fixed
  pseudo-random order (sha256(pair_id + salt)) and cut into N shards whose sizes are whole
  multiples of batch_size (batches never straddle two shards, as in the lab); only the
  last shard takes the remainder, so no training row is dropped.
"""
from __future__ import annotations

import shutil

import numpy as np

from .common import interim, read_parquet, require, sha256_text, step_args, update_manifest, write_parquet


def shard_sizes(n_rows: int, num_shards: int, batch_size: int) -> list[int]:
    batches = n_rows // batch_size
    per_shard = batches // num_shards
    sizes = [per_shard * batch_size] * num_shards
    sizes[-1] += n_rows - sum(sizes)  # remainder (incl. last partial batch) goes to the final shard
    return sizes


def main(argv: list[str] | None = None) -> None:
    _, cfg = step_args(__doc__, argv)
    sh = cfg["shards"]
    out_dir = interim(cfg, "shards")
    shutil.rmtree(out_dir, ignore_errors=True)  # never leave shards from an older run behind

    if sh["num_shards"] <= 1:
        update_manifest(cfg, "step8_shard", {"enabled": False, "reason": "single file is sufficient at this size"})
        print("Sharding disabled (shards.num_shards <= 1) - see README section 2.")
        return

    src = interim(cfg, "splits", "train.parquet")
    require(src, "step7_split")
    train = read_parquet(src)
    order = np.argsort(train["pair_id"].map(lambda p: sha256_text(p + sh["salt"])).to_numpy(), kind="stable")
    train = train.iloc[order].reset_index(drop=True)

    sizes = shard_sizes(len(train), sh["num_shards"], sh["batch_size"])
    start, shards = 0, {}
    for i, size in enumerate(sizes):
        name = f"train-{i:05d}-of-{len(sizes):05d}.parquet"
        write_parquet(train.iloc[start:start + size], out_dir / name)
        shards[name] = size
        start += size
        print(f"  {name}: {size:,} rows")

    update_manifest(cfg, "step8_shard", {"enabled": True, "batch_size": sh["batch_size"], "shards": shards})


if __name__ == "__main__":
    main()
