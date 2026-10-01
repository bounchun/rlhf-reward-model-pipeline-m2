"""Score the blind human-review sample (after step 10).

1. Open audit/label_review_<version>.csv and fill `my_choice` with A, B or tie for each row
   (you only see the context and the two replies - the crowdworker label is hidden).
2. Run:  python -m src.score_review
   -> agreement with the crowdworker labels is written to the manifest (steps.human_review)
"""
from __future__ import annotations

import pandas as pd

from .common import (join, local_path, manifest_path, read_parquet, step_args, storage_roots, update_manifest,
                     write_bytes)


def main(argv: list[str] | None = None) -> None:
    _, cfg = step_args(__doc__, argv)
    version = cfg["dataset"]["version"]
    root, _ = storage_roots(cfg)

    review = pd.read_csv(local_path(cfg, "audit_dir") / f"label_review_{version}.csv", keep_default_na=False)
    review["my_choice"] = review["my_choice"].astype(str).str.strip().str.upper()
    done = review[review["my_choice"].isin(["A", "B"])]
    ties = int((review["my_choice"] == "TIE").sum())
    if done.empty:
        raise SystemExit("No rows labelled A or B yet - fill in the my_choice column first.")

    train = read_parquet(join(root, "processed", version, "train.parquet"), columns=["pair_id", "label"])
    merged = done.merge(train, on="pair_id", how="left")
    if merged["label"].isna().any():
        raise SystemExit("Some review pair_ids are not in train.parquet - was the dataset regenerated?")
    merged["agree"] = (merged["my_choice"] == "A").astype(int) == merged["label"]

    result = {
        "labelled": int(len(done)),
        "ties": ties,
        "agreement_with_crowdworkers": round(float(merged["agree"].mean()), 4),
        "agreement_by_subset": {k: round(float(v), 4) for k, v in merged.groupby("subset")["agree"].mean().items()},
    }
    update_manifest(cfg, "human_review", result)
    write_bytes(join(root, "manifests", f"{version}.json"), manifest_path(cfg).read_bytes())
    print(result)


if __name__ == "__main__":
    main()
