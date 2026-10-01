"""STEP 4 - Data validation and quality checks (flag, never drop).

    python -m src.step4_quality_checks

Input : <interim_dir>/<version>/cleaned.parquet   (from step 3)
Output: <main_root>/audit/<version>/poison_screen.csv        flagged n-grams
        <main_root>/audit/<version>/poison_flagged_pairs.csv pairs containing them
        <main_root>/audit/<version>/quality_report.json      validation + PII + length summary
        manifest: steps.step4_quality_checks

Checks
- Validation (hard asserts): pair_id unique, no empty or identical replies, only known subsets.
- Poisoning screen (README section 12): word n-grams that occur in >= min_pairs pairs and sit
  almost always on ONE side (chosen vs rejected) - the signature of a planted trigger. Flagged
  for manual review only: real refusal phrases in harmless-base are also one-sided.
- PII scan: count (not remove) pairs with an email address or phone number.
- Length summary per subset (input to the M2 length-bias analysis).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from .common import (interim, join, read_parquet, require, sha256_bytes, step_args, storage_roots,
                     update_manifest, write_bytes)


def validate(df: pd.DataFrame, cfg: dict) -> dict:
    checks = {
        "pair_id_unique": bool(df["pair_id"].is_unique),
        "no_empty_replies": bool(((df["chosen_resp"] != "") & (df["rejected_resp"] != "")).all()),
        "no_identical_replies": bool((df["chosen_resp"] != df["rejected_resp"]).all()),
        "known_subsets_only": bool(df["subset"].isin(cfg["dataset"]["subsets"]).all()),
        "no_nulls": bool(df.notna().all().all()),
    }
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise SystemExit(f"ERROR: validation failed: {failed}")
    return checks


def poison_screen(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = cfg["poison_screen"]
    n = p["ngram"]

    def grams(text: str) -> set[str]:
        w = text.lower().split()
        return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}

    # Pass 1 - count one-sided n-grams as 64-bit hashes in numpy (low memory).
    # hash() is only compared within this run; the report itself is built from the text.
    c_chunks, r_chunks = [], []
    for c_txt, r_txt in zip(df["chosen_resp"], df["rejected_resp"]):
        gc, gr = grams(c_txt), grams(r_txt)
        co, ro = gc - gr, gr - gc
        c_chunks.append(np.fromiter((hash(g) for g in co), dtype=np.int64, count=len(co)))
        r_chunks.append(np.fromiter((hash(g) for g in ro), dtype=np.int64, count=len(ro)))
    empty = np.empty(0, dtype=np.int64)
    uc, cc = np.unique(np.concatenate(c_chunks) if c_chunks else empty, return_counts=True)
    ur, rc = np.unique(np.concatenate(r_chunks) if r_chunks else empty, return_counts=True)
    del c_chunks, r_chunks
    keys = np.union1d(uc, ur)
    c_cnt, r_cnt = np.zeros(len(keys), np.int64), np.zeros(len(keys), np.int64)
    c_cnt[np.searchsorted(keys, uc)] = cc
    r_cnt[np.searchsorted(keys, ur)] = rc
    total = c_cnt + r_cnt
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.maximum(c_cnt, r_cnt) / total
    sel = (total >= p["min_pairs"]) & (ratio >= p["one_sided_ratio"])
    stats = {int(k): (int(c), int(r)) for k, c, r in zip(keys[sel], c_cnt[sel], r_cnt[sel])}

    # Pass 2 - recover the text of flagged n-grams and which pairs contain them
    text_of, pair_hits = {}, defaultdict(set)
    if stats:
        for pid, c_txt, r_txt in zip(df["pair_id"], df["chosen_resp"], df["rejected_resp"]):
            for g in grams(c_txt) ^ grams(r_txt):
                if hash(g) in stats:
                    text_of[hash(g)] = g
                    pair_hits[pid].add(g)

    report = pd.DataFrame(
        [{"ngram": text_of[h], "n_pairs": c + r, "side": "chosen" if c >= r else "rejected",
          "one_sided_ratio": round(max(c, r) / (c + r), 4)} for h, (c, r) in stats.items()],
        columns=["ngram", "n_pairs", "side", "one_sided_ratio"])
    report = report.sort_values(["n_pairs", "ngram"], ascending=[False, True]).head(p["max_report"])
    flagged = set(report["ngram"])
    hits = pd.DataFrame([{"pair_id": pid, "ngrams": " | ".join(sorted(gs & flagged))}
                         for pid, gs in pair_hits.items() if gs & flagged], columns=["pair_id", "ngrams"])
    return report.reset_index(drop=True), hits.sort_values("pair_id").reset_index(drop=True)


def pii_scan(df: pd.DataFrame, cfg: dict) -> dict:
    text = df["context"] + "\n" + df["chosen_resp"] + "\n" + df["rejected_resp"]
    out = {}
    for name, pattern in cfg["pii"].items():
        rx = re.compile(pattern)
        hit = text.map(lambda s: bool(rx.search(s)))
        out[name] = {"pairs": int(hit.sum()),
                     "by_subset": {k: int(v) for k, v in hit.groupby(df["subset"]).sum().items()}}
    return out


def length_summary(df: pd.DataFrame) -> dict:
    lc, lr = df["chosen_resp"].str.split().str.len(), df["rejected_resp"].str.split().str.len()
    out = {}
    for sub, idx in df.groupby("subset").groups.items():
        out[sub] = {"pairs": int(len(idx)),
                    "median_words_chosen": float(lc[idx].median()),
                    "median_words_rejected": float(lr[idx].median()),
                    "share_chosen_longer": round(float((lc[idx] > lr[idx]).mean()), 4)}
    return out


def main(argv: list[str] | None = None) -> None:
    _, cfg = step_args(__doc__, argv)
    src = interim(cfg, "cleaned.parquet")
    require(src, "step3_clean")
    df = read_parquet(src)

    checks = validate(df, cfg)
    report, hits = poison_screen(df, cfg)
    pii = pii_scan(df, cfg)
    lengths = length_summary(df)
    quality = {"validation": checks, "pii_scan": pii, "length_by_subset": lengths,
               "poison_screen": {"flagged_ngrams": int(len(report)), "flagged_pairs": int(len(hits)),
                                 "params": cfg["poison_screen"]}}

    prefix = join(storage_roots(cfg)[0], "audit", cfg["dataset"]["version"])
    files = {}
    for name, data in [("poison_screen.csv", report.to_csv(index=False).encode()),
                       ("poison_flagged_pairs.csv", hits.to_csv(index=False).encode()),
                       ("quality_report.json", (json.dumps(quality, indent=2) + "\n").encode())]:
        write_bytes(join(prefix, name), data)
        files[f"audit/{name}"] = {"uri": join(prefix, name), "sha256": sha256_bytes(data)}

    update_manifest(cfg, "step4_quality_checks", {**quality, "files": files})
    print(f"Validation passed: {checks}")
    print(f"Poison screen: {len(report)} one-sided n-grams in {len(hits)} pairs (flag only) -> {prefix}")
    print(f"PII: " + ", ".join(f"{k}={v['pairs']}" for k, v in pii.items()))


if __name__ == "__main__":
    main()
