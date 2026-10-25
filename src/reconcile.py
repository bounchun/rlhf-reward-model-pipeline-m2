"""Reconciliation - check the pipeline's output with a second, independent method (Lecture 6, step 7b).

    python -m src.reconcile

The pipeline (steps 3-11) uses pandas. This script recomputes the key facts straight from the raw
JSONL files with plain Python (json, gzip, hashlib) and compares them with what was stored:

  R1  raw rows per file, recounted from the raw bytes   == rows the pipeline accounted for
  R2  rows_read = cleaned + quarantined (quarantine file read back from storage)
  R3  cleaned = train + dev + test + F1 + F2 + unused pool (stored files read back)
  R4  every stored file has the SHA-256 and row count the manifest says
  R5  label direction: for every stored pair, the reply marked as preferred is the raw `chosen`
      reply of a raw record with the same context (independent of the A/B swap code)
  R6  no pair_id or group_id in two stored splits

If two different methods agree, both are very probably right. Any disagreement exits with an
error and is written to <main_root>/audit/<version>/reconciliation.json.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import unicodedata

import pandas as pd

from .common import (SPLITS, join, manifest_path, raw_manifest_path, raw_prefix, read_bytes, read_parquet, require,
                     step_args, storage_roots, write_bytes)

MARK = "\n\nAssistant:"


def canon(text: str, masks: list[tuple[re.Pattern, str]]) -> str:
    """Independent text canonicalisation: NFC, PII masks, then all whitespace collapsed."""
    text = unicodedata.normalize("NFC", text)
    text = "\n".join(" ".join(line.split()) for line in text.split("\n"))
    for rx, token in masks:
        text = rx.sub(token, text)
    return " ".join(text.split())


def key(context: str, reply: str) -> str:
    return hashlib.sha256((context + "\x1f" + reply).encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> None:
    _, cfg = step_args(__doc__, argv)
    version = cfg["dataset"]["version"]
    root, holdout = storage_roots(cfg)
    require(manifest_path(cfg), "the pipeline")
    m = json.loads(manifest_path(cfg).read_text())["steps"]
    masks = []
    if cfg.get("pii_mask", {}).get("enabled"):
        masks = [(re.compile(cfg["pii"][k]), tok) for k, tok in cfg["pii_mask"]["tokens"].items()]

    checks: dict[str, dict] = {}

    def record(name: str, expected, got) -> None:
        checks[name] = {"expected": expected, "independent": got, "agree": expected == got}

    # ---- R1: recount raw rows and index the raw chosen replies
    files = json.loads(raw_manifest_path(cfg).read_text())["files"]
    rows_per_file, chosen_keys = {}, set()
    for name in files:
        lines = [ln for ln in gzip.decompress(read_bytes(join(raw_prefix(cfg), name))).decode("utf-8").splitlines()
                 if ln.strip()]
        rows_per_file[name] = len(lines)
        for ln in lines:
            try:
                rec = json.loads(ln)
                chosen = rec["chosen"]
                cut = chosen.rindex(MARK)
            except (ValueError, KeyError, TypeError, AttributeError):
                continue
            chosen_keys.add(key(canon(chosen[:cut], masks), canon(chosen[cut + len(MARK):], masks)))
    raw_total = sum(rows_per_file.values())
    acc = m["step3_clean"]["row_accounting"]
    record("R1 raw rows recounted from the raw files", acc["rows_read"], raw_total)

    # ---- R2: read + quarantine back from storage
    q = read_parquet(join(root, "quarantine", version, "quarantine.parquet"), columns=["reason"])
    record("R2 rows_read = cleaned + quarantined", raw_total, m["step3_clean"]["pairs_after_cleaning"] + len(q))

    # ---- R3/R4/R6: stored splits, read back from the buckets
    frames, sha_ok = {}, {}
    cols = ["pair_id", "group_id", "context", "response_a", "response_b", "label"]
    for split in SPLITS:
        url = (join(root, "processed", version, f"{split}.parquet") if split in ("train", "dev")
               else join(holdout, version, f"{split}.parquet"))
        data = read_bytes(url)
        sha_ok[split] = hashlib.sha256(data).hexdigest() == m["step9_store_processed"]["files"][split]["sha256"]
        frames[split] = pd.read_parquet(io.BytesIO(data), columns=cols)
    record("R4 stored files match their manifest SHA-256", {s: True for s in SPLITS}, sha_ok)
    record("R4 stored row counts match the manifest",
           {s: m["step9_store_processed"]["files"][s]["rows"] for s in SPLITS}, {s: len(frames[s]) for s in SPLITS})
    stored_rows = sum(len(f) for f in frames.values())
    record("R3 cleaned = stored splits + unused pool", m["step3_clean"]["pairs_after_cleaning"],
           stored_rows + m["step6_filter"]["unused_pool_pairs"])

    # ---- R5: label direction, checked against the raw `chosen` replies
    missing = 0
    for f in frames.values():
        for c, a, b, lab in zip(f["context"], f["response_a"], f["response_b"], f["label"]):
            preferred = a if lab == 1 else b
            if key(canon(c, masks), canon(preferred, masks)) not in chosen_keys:
                missing += 1
    record("R5 preferred reply is the raw `chosen` reply (stored pairs not matched)", 0, missing)

    # ---- R6: no overlap
    seen_p, seen_g, dup_p, dup_g = set(), set(), 0, 0
    for f in frames.values():
        p, g = set(f["pair_id"]), set(f["group_id"])
        dup_p += len(seen_p & p)
        dup_g += len(seen_g & g)
        seen_p |= p
        seen_g |= g
    record("R6 pair_id / group_id in two splits", {"pairs": 0, "groups": 0}, {"pairs": dup_p, "groups": dup_g})

    ok = all(c["agree"] for c in checks.values())
    report = {"dataset_version": version, "all_agree": ok, "checks": checks, "raw_rows_per_file": rows_per_file}
    url = join(root, "audit", version, "reconciliation.json")
    write_bytes(url, (json.dumps(report, indent=2) + "\n").encode())
    for name, c in checks.items():
        print(f"  {'AGREE   ' if c['agree'] else 'DISAGREE'} {name}: pipeline {c['expected']} | independent {c['independent']}")
    if not ok:
        raise SystemExit(f"ERROR: reconciliation failed -> {url}")
    print(f"All {len(checks)} checks agree -> {url}")


if __name__ == "__main__":
    main()
