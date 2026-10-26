"""STEP 3 - Load, validate (data contract) and clean the raw data. Bad rows are quarantined, not lost.

    python -m src.step3_clean

Input : <main_root>/raw/hh-rlhf/<hf_revision>/...jsonl.gz   (from step 2; checksums re-verified)
Output: <interim_dir>/<version>/cleaned.parquet
        columns: subset, source_split, pair_id, group_first_key, group_full_key,
                 context, chosen_resp, rejected_resp
        <main_root>/quarantine/<version>/quarantine.parquet      every rejected record, with
                 subset, source_split, line_no, reason, counter, chosen, rejected, raw_line
        manifest: starts manifests/<version>.json; raw counts, rows quarantined per rule,
                  PII masking counts, and the row accounting check

Rules (src/contract.py; the first rule a record breaks is its reason)
  0  bad_json                 line is not a JSON object
  1  missing / not_text /     field is null or empty, is not a string, does not start with a
     malformed_turns          Human turn or does not end with an Assistant turn
  2  context_mismatch         chosen/rejected differ before the final reply
  3  normalise                Unicode NFC, collapse spaces/tabs, strip (case and punctuation kept)
  4a empty_reply              either final reply is empty after normalising
  4b identical_replies        chosen reply == rejected reply (no preference signal)
  P  PII masking (v1.1)       e-mail / phone patterns -> [EMAIL] / [PHONE] in context and replies
  4c identical_after_pii_mask the two replies differed only by a masked e-mail or phone number
  5  duplicate_pair           same pair_id as an earlier record; first kept

Identity keys are computed from the source text BEFORE masking (pair_id, group_first_key,
group_full_key), so a pair keeps the same id, position and split across cleaning versions.

Row accounting (Lecture 6, "count every row"): rows_read = rows_kept + rows_quarantined is
asserted, so a record can never disappear without a reason.

Monitoring (Lecture 6, "count and alert"): the quarantine rate of every raw file and the set of JSON
fields seen in it are recorded; a file above alerts.max_quarantine_rate_per_file, or with unexpected
fields (schema drift), raises an ALERT (and stops the run if alerts.fail_on_alert is true).
"""
from __future__ import annotations

import gzip
import json
from collections import Counter

import pandas as pd

from .common import (interim, join, raw_manifest_path, raw_prefix, read_bytes, require, sha256_bytes, sha256_text,
                     start_manifest, step_args, storage_roots, update_manifest, write_parquet)
from .contract import COUNTERS, REASON_COUNTER, check_raw_record, parse_line
from .text_utils import full_context_key, group_key, mask_pii, normalise, split_context

QUARANTINE_COLUMNS = ["subset", "source_split", "line_no", "reason", "counter", "chosen", "rejected", "raw_line"]


def load_raw(cfg: dict, schemas: dict | None = None) -> tuple[pd.DataFrame, list[dict], int]:
    """Read every raw file; return (records that pass the field rules, quarantined records, rows_read).

    If `schemas` is a dict, it is filled with {file: {"field1,field2": count}} - the set of JSON keys
    seen in each file, so a change of shape (schema drift, Lecture 6) is visible in the manifest."""
    mpath = raw_manifest_path(cfg)
    require(mpath, "step1_scrape_raw + step2_store_raw")
    files = json.loads(mpath.read_text())["files"]
    rows, bad, rows_read = [], [], 0
    for key, meta in files.items():
        uri = join(raw_prefix(cfg), key)  # resolve against the ACTIVE storage root
        data = read_bytes(uri)
        if sha256_bytes(data) != meta["sha256"]:  # tamper / corruption check
            raise SystemExit(f"ERROR: checksum mismatch for {uri} - refusing to continue.")
        subset, fname = key.split("/")
        source_split = fname.split(".")[0]
        for line_no, line in enumerate(gzip.decompress(data).decode("utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            rows_read += 1
            where = {"subset": subset, "source_split": source_split, "line_no": line_no}
            rec, reason = parse_line(line)
            if schemas is not None and rec is not None:
                shape = ",".join(sorted(rec))
                schemas.setdefault(key, Counter())[shape] += 1
            if reason is None:
                reason = check_raw_record(rec)
            if reason is not None:
                bad.append({**where, "reason": reason,
                            "chosen": _text(rec, "chosen"), "rejected": _text(rec, "rejected"),
                            "raw_line": line if rec is None else ""})
                continue
            rows.append({**where, "chosen": rec["chosen"], "rejected": rec["rejected"]})
    cols = ["subset", "source_split", "line_no", "chosen", "rejected"]
    return pd.DataFrame(rows, columns=cols), bad, rows_read


def _text(rec: dict | None, field: str) -> str:
    v = None if rec is None else rec.get(field)
    return v if isinstance(v, str) else ("" if v is None else json.dumps(v))


def clean(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[dict], dict]:
    """Pair-level rules + normalisation + PII masking. Returns (kept, quarantined, pii_counts)."""
    bad: list[dict] = []

    def quarantine(mask, reason: str) -> None:
        for r in df.loc[mask, ["subset", "source_split", "line_no", "chosen", "rejected"]].to_dict("records"):
            bad.append({**r, "reason": reason, "raw_line": ""})

    c_parts, r_parts = df["chosen"].map(split_context), df["rejected"].map(split_context)
    df = df.assign(context_raw=c_parts.str[0], chosen_resp=c_parts.str[1], rejected_resp=r_parts.str[1])
    same = df["context_raw"].values == r_parts.str[0].values
    quarantine(~same, "context_mismatch")
    df = df[same].copy()

    for col, src in [("context", "context_raw"), ("chosen_resp", "chosen_resp"), ("rejected_resp", "rejected_resp")]:
        df[col] = df[src].map(normalise)

    empty = (df["chosen_resp"] == "") | (df["rejected_resp"] == "")
    quarantine(empty, "empty_reply")
    df = df[~empty].copy()
    identical = df["chosen_resp"] == df["rejected_resp"]
    quarantine(identical, "identical_replies")
    df = df[~identical].copy()

    # identity keys from the unmasked source text (stable across cleaning versions)
    df["pair_id"] = [sha256_text(c + "\x1f" + a + "\x1f" + b)
                     for c, a, b in zip(df["context"], df["chosen_resp"], df["rejected_resp"])]
    df["group_first_key"] = df["context"].map(lambda c: sha256_text(group_key(c)))
    df["group_full_key"] = df["context"].map(lambda c: sha256_text(full_context_key(c)))

    pii_counts = {"enabled": bool(cfg.get("pii_mask", {}).get("enabled", False))}
    if pii_counts["enabled"]:
        tokens = cfg["pii_mask"]["tokens"]
        patterns = {k: cfg["pii"][k] for k in tokens}
        hits = {k: pd.Series(False, index=df.index) for k in tokens}
        for col in ["context", "chosen_resp", "rejected_resp"]:
            masked = df[col].map(lambda s: mask_pii(s, patterns, tokens))
            df[col] = masked.str[0]
            for k in tokens:
                hits[k] |= masked.str[1].map(lambda c, k=k: c[k] > 0)
        pii_counts["pairs_masked"] = {k: int(v.sum()) for k, v in hits.items()}
        pii_counts["pairs_masked_by_subset"] = {
            k: {s: int(n) for s, n in v.groupby(df["subset"]).sum().items()} for k, v in hits.items()}
        same_after = df["chosen_resp"] == df["rejected_resp"]
        quarantine(same_after, "identical_after_pii_mask")
        df = df[~same_after].copy()

    dup = df.duplicated("pair_id", keep="first")
    quarantine(dup, "duplicate_pair")
    df = df[~dup]
    cols = ["subset", "source_split", "pair_id", "group_first_key", "group_full_key",
            "context", "chosen_resp", "rejected_resp"]
    return df[cols].reset_index(drop=True), bad, pii_counts


def quarantine_frame(bad: list[dict]) -> pd.DataFrame:
    q = pd.DataFrame(bad, columns=[c for c in QUARANTINE_COLUMNS if c != "counter"])
    q["counter"] = q["reason"].map(REASON_COUNTER)
    q = q[QUARANTINE_COLUMNS].sort_values(["subset", "source_split", "line_no"]).reset_index(drop=True)
    q["line_no"] = q["line_no"].astype("int64")
    return q


def main(argv: list[str] | None = None) -> None:
    _, cfg = step_args(__doc__, argv)
    version = cfg["dataset"]["version"]
    schemas: dict = {}
    raw, bad_fields, rows_read = load_raw(cfg, schemas)
    df, bad_pairs, pii_counts = clean(raw, cfg)
    q = quarantine_frame(bad_fields + bad_pairs)

    # Lecture 6: every row read is either kept or quarantined - never silently lost
    if rows_read != len(df) + len(q):
        raise SystemExit(f"ERROR: row accounting failed: read {rows_read:,} != kept {len(df):,} + "
                         f"quarantined {len(q):,}")

    # schema drift: fields are read by name and only chosen/rejected are required, so extra or missing
    # keys never crash the run - but any shape other than the expected one is reported
    expected = ",".join(sorted(cfg.get("alerts", {}).get("expected_fields", ["chosen", "rejected"])))
    drift = {f: {k: int(n) for k, n in c.items() if k != expected} for f, c in schemas.items()}
    drift = {f: d for f, d in drift.items() if d}
    # reject-rate alert: a jump in the quarantine rate of one file means something changed upstream
    max_rate = float(cfg.get("alerts", {}).get("max_quarantine_rate_per_file", 1.0))
    per_file_n = pd.concat([raw[["subset", "source_split"]], q[["subset", "source_split"]]]).value_counts()
    per_file_q = q[["subset", "source_split"]].value_counts()
    rates = {f"{a}/{b}.jsonl.gz": round(float(per_file_q.get((a, b), 0) / n), 4) for (a, b), n in per_file_n.items()}
    alerts = [f"{f}: quarantine rate {r:.2%} > {max_rate:.2%}" for f, r in sorted(rates.items()) if r > max_rate]
    alerts += [f"{f}: unexpected JSON fields {d}" for f, d in sorted(drift.items())]
    for a in alerts:
        print(f"  ALERT  {a}")
    if alerts and cfg.get("alerts", {}).get("fail_on_alert", False):
        raise SystemExit(f"ERROR: {len(alerts)} alert(s) and alerts.fail_on_alert is true")

    write_parquet(df, interim(cfg, "cleaned.parquet"))
    q_url = join(storage_roots(cfg)[0], "quarantine", version, "quarantine.parquet")
    q_bytes = write_parquet(q, q_url)

    counters = {c: 0 for c in COUNTERS}
    counters.update(Counter(q["counter"]))
    start_manifest(cfg)
    update_manifest(cfg, "step3_clean", {
        "raw_files_sha256": {k: v["sha256"] for k, v in json.loads(raw_manifest_path(cfg).read_text())["files"].items()},
        "raw_pairs": {f"{a}/{b}": int(n) for (a, b), n in
                      pd.concat([raw[["subset", "source_split"]], q.loc[q["reason"] != "bad_json", ["subset", "source_split"]]])
                      .groupby(["subset", "source_split"]).size().items()},
        "unparseable_json_lines": int(counters["0_unparseable_json"]),
        "rows_dropped": {k: int(v) for k, v in sorted(counters.items())},
        "quarantine_reasons": {k: int(v) for k, v in q["reason"].value_counts().sort_index().items()},
        "quarantine_by_subset": {s: {r: int(n) for r, n in g["reason"].value_counts().sort_index().items()}
                                 for s, g in q.groupby("subset")},
        "pii_masking": pii_counts,
        "row_accounting": {"rows_read": rows_read, "rows_kept": int(len(df)), "rows_quarantined": int(len(q)),
                           "check": "passed: rows_read = rows_kept + rows_quarantined"},
        "quarantine_rate_by_file": dict(sorted(rates.items())),
        "schema_by_file": {f: {k: int(n) for k, n in c.items()} for f, c in sorted(schemas.items())},
        "alerts": alerts,
        "quarantine_file": {"uri": q_url, "sha256": sha256_bytes(q_bytes), "rows": int(len(q))},
        "pairs_after_cleaning": int(len(df)),
    })
    print("Counters:")
    for name, value in [("rows_read", rows_read), ("rows_kept", len(df)), ("rows_quarantined", len(q))]:
        print(f"  {name:28s} {value:>8,d}")
    for reason, n in q["reason"].value_counts().sort_index().items():
        print(f"    {reason:26s} {n:>8,d}")
    if pii_counts["enabled"]:
        print("  pairs with PII masked       ", json.dumps(pii_counts["pairs_masked"]))
    print(f"Kept {len(df):,} pairs -> {interim(cfg, 'cleaned.parquet')}")
    print(f"Quarantine ({len(q):,} rows, with reasons) -> {q_url}")


if __name__ == "__main__":
    main()
