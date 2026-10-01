"""STEP 10 - Export the blind human-review sample (quality check on the labels).

    python -m src.step10_review_sample
    ... label it by hand, then:  python -m src.score_review

Input : <interim_dir>/<version>/splits/train.parquet   (from step 7)
Output: <audit_dir>/label_review_<version>.csv          (local only - contains dataset text)
        columns: review_id, pair_id, subset, context, response_a, response_b, my_choice
        The crowdworker label is NOT included (blind review); fill my_choice with A, B or tie.

Sample: review.sample_size train pairs, stratified by subset (largest-remainder rounding),
chosen in the order of sha256(pair_id + review.salt) - reproducible, no RNG.
An existing sheet is never overwritten (it may contain your labels); if the data changed so
that the sheet no longer matches train, a fresh one is written next to it as *.new.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import interim, local_path, read_parquet, require, sha256_text, step_args, update_manifest


def review_sample(train: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    r = cfg["review"]
    share = train["subset"].astype(str).value_counts(normalize=True).sort_index()
    target = share * r["sample_size"]
    k = np.floor(target).astype(int)
    for sub in (target - k).sort_values(ascending=False).index[: r["sample_size"] - int(k.sum())]:
        k[sub] += 1
    t = train.assign(subset=train["subset"].astype(str),
                     order=train["pair_id"].map(lambda p: sha256_text(p + r["salt"])))
    parts = [g.sort_values("order").head(int(k[sub])) for sub, g in t.groupby("subset")]
    sample = pd.concat(parts).sort_values("order")
    sample = sample[["pair_id", "subset", "context", "response_a", "response_b"]].reset_index(drop=True)
    sample.insert(0, "review_id", range(1, len(sample) + 1))
    sample["my_choice"] = ""
    return sample


def main(argv: list[str] | None = None) -> None:
    _, cfg = step_args(__doc__, argv)
    src = interim(cfg, "splits", "train.parquet")
    require(src, "step7_split")
    train = read_parquet(src)
    path = local_path(cfg, "audit_dir") / f"label_review_{cfg['dataset']['version']}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    status = "written"
    if path.exists():
        old = set(pd.read_csv(path, usecols=["pair_id"])["pair_id"])
        if old <= set(train["pair_id"]):
            status = "kept existing sheet (still matches train)"
        else:
            path = path.with_suffix(".new.csv")
            status = "existing sheet is STALE - fresh sample written to *.new.csv; replace the old file"
    if status != "kept existing sheet (still matches train)":
        review_sample(train, cfg).to_csv(path, index=False)

    update_manifest(cfg, "step10_review_sample", {"sheet": path.name, "status": status,
                                                  "sample_size": cfg["review"]["sample_size"]})
    print(f"Review sheet: {status} -> {path}")


if __name__ == "__main__":
    main()
