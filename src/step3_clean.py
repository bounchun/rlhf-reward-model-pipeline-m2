"""STEP 3 - Load, validate and clean the raw data.

    python -m src.step3_clean

Input : <main_root>/raw/hh-rlhf/<hf_revision>/...jsonl.gz   (from step 2; checksums re-verified)
Output: <interim_dir>/<version>/cleaned.parquet
        columns: subset, source_split, pair_id, context, chosen_resp, rejected_resp
        manifest: starts manifests/<version>.json; records raw counts + rows dropped per rule

Cleaning rules (each one counts the rows it drops)
  1  malformed or missing  - field is null/empty, does not start with a Human turn,
                             or does not end with an Assistant turn; unparseable JSON lines
  2  context mismatch      - chosen/rejected do not share the same text before the final reply
  3  normalise             - Unicode NFC, collapse spaces/tabs, strip (case and punctuation kept)
  4a empty reply           - either final reply is empty after stripping
  4b identical replies     - chosen reply == rejected reply (no preference signal)
  5  exact duplicates      - same pair_id = sha256(context, chosen, rejected); first kept
"""
from __future__ import annotations

import gzip
import json

import pandas as pd

from .common import (interim, join, raw_manifest_path, raw_prefix, read_bytes, require, sha256_bytes, sha256_text,
                     start_manifest, step_args, update_manifest, write_parquet)
from .text_utils import is_well_formed, normalise, split_context


def load_raw(cfg: dict) -> tuple[pd.DataFrame, int]:
    mpath = raw_manifest_path(cfg)
    require(mpath, "step1_scrape_raw + step2_store_raw")
    files = json.loads(mpath.read_text())["files"]
    rows, bad_json = [], 0
    for key, meta in files.items():
        uri = join(raw_prefix(cfg), key)  # resolve against the ACTIVE storage root
        data = read_bytes(uri)
        if sha256_bytes(data) != meta["sha256"]:  # tamper / corruption check
            raise SystemExit(f"ERROR: checksum mismatch for {uri} - refusing to continue.")
        subset, fname = key.split("/")
        for line in gzip.decompress(data).decode("utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                continue
            rows.append({"subset": subset, "source_split": fname.split(".")[0],
                         "chosen": rec.get("chosen"), "rejected": rec.get("rejected")})
    return pd.DataFrame(rows, columns=["subset", "source_split", "chosen", "rejected"]), bad_json


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    drops = {}
    ok = df["chosen"].map(is_well_formed) & df["rejected"].map(is_well_formed)
    drops["1_malformed_or_missing"] = int((~ok).sum())
    df = df[ok].copy()

    c_parts, r_parts = df["chosen"].map(split_context), df["rejected"].map(split_context)
    df["context_raw"], df["chosen_resp"], df["rejected_resp"] = c_parts.str[0], c_parts.str[1], r_parts.str[1]
    same = df["context_raw"].values == r_parts.str[0].values
    drops["2_context_mismatch"] = int((~same).sum())
    df = df[same].copy()

    for col, src in [("context", "context_raw"), ("chosen_resp", "chosen_resp"), ("rejected_resp", "rejected_resp")]:
        df[col] = df[src].map(normalise)

    empty = (df["chosen_resp"] == "") | (df["rejected_resp"] == "")
    drops["4a_empty_response"] = int(empty.sum())
    df = df[~empty]
    identical = df["chosen_resp"] == df["rejected_resp"]
    drops["4b_identical_responses"] = int(identical.sum())
    df = df[~identical].copy()

    df["pair_id"] = [sha256_text(c + "\x1f" + a + "\x1f" + b)
                     for c, a, b in zip(df["context"], df["chosen_resp"], df["rejected_resp"])]
    dup = df.duplicated("pair_id", keep="first")
    drops["5_exact_duplicates"] = int(dup.sum())
    df = df[~dup]
    cols = ["subset", "source_split", "pair_id", "context", "chosen_resp", "rejected_resp"]
    return df[cols].reset_index(drop=True), drops


def main(argv: list[str] | None = None) -> None:
    _, cfg = step_args(__doc__, argv)
    raw, bad_json = load_raw(cfg)
    print(f"Loaded {len(raw):,} raw pairs ({bad_json} unparseable JSON lines)")
    df, drops = clean(raw)
    write_parquet(df, interim(cfg, "cleaned.parquet"))

    start_manifest(cfg)
    update_manifest(cfg, "step3_clean", {
        "raw_files_sha256": {k: v["sha256"] for k, v in json.loads(raw_manifest_path(cfg).read_text())["files"].items()},
        "raw_pairs": {f"{a}/{b}": int(n) for (a, b), n in raw.groupby(["subset", "source_split"]).size().items()},
        "unparseable_json_lines": bad_json,
        "rows_dropped": {"0_unparseable_json": bad_json, **drops},
        "pairs_after_cleaning": int(len(df)),
    })
    print("Rows dropped:", json.dumps(drops))
    print(f"Kept {len(df):,} pairs -> {interim(cfg, 'cleaned.parquet')}")


if __name__ == "__main__":
    main()
