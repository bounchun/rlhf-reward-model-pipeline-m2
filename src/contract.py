"""The data contract, written as code (Lecture 5 "executable data contract", Lecture 6 "ValidateRow").

Two contracts, used at two points of the pipeline:

1. RAW contract - one hh-rlhf record (one JSON line) at a time, in step 3.
   `check_raw_record` returns the reason a record breaks the contract, or None. Records that break
   it are NOT dropped silently and do NOT crash the run: they go to the quarantine with the reason,
   the source file and the line number (Lecture 6: "don't crash, don't hide").

2. PROCESSED contract - whole split DataFrames, in step 9 before anything is uploaded.
   `validation_log` returns one row per rule: rule, rationale, violations, severity, response
   (Lecture 5 slide 20: "record the rule, its rationale, the number of violations and the response").
   Any violation of a HARD rule stops the run; a WARN rule is a plausibility check that is logged only.

The rules come from the data dictionary (README, data card), not from looking at the data.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .text_utils import ROLE_RE

# ----------------------------------------------------------------------------- 1. raw contract
REQUIRED_RAW_FIELDS = ("chosen", "rejected")

# reason code -> manifest counter it is reported under (keeps the v1.0 counter names)
REASON_COUNTER = {
    "bad_json": "0_unparseable_json",
    "missing:chosen": "1_malformed_or_missing",
    "missing:rejected": "1_malformed_or_missing",
    "not_text:chosen": "1_malformed_or_missing",
    "not_text:rejected": "1_malformed_or_missing",
    "malformed_turns:chosen": "1_malformed_or_missing",
    "malformed_turns:rejected": "1_malformed_or_missing",
    "context_mismatch": "2_context_mismatch",
    "empty_reply": "4a_empty_response",
    "identical_replies": "4b_identical_responses",
    "identical_after_pii_mask": "4c_identical_after_pii_mask",
    "duplicate_pair": "5_exact_duplicates",
}
COUNTERS = sorted(set(REASON_COUNTER.values()))


def parse_line(line: str) -> tuple[dict | None, str | None]:
    """(record, None) or (None, 'bad_json'). A JSON value that is not an object is also bad_json."""
    try:
        rec = json.loads(line)
    except json.JSONDecodeError:
        return None, "bad_json"
    if not isinstance(rec, dict):
        return None, "bad_json"
    return rec, None


def turn_structure_ok(s: str) -> bool:
    """Starts with a Human turn, ends with an Assistant turn, nothing before the first turn."""
    roles = ROLE_RE.findall(s)
    return bool(roles) and roles[0] == "Human" and roles[-1] == "Assistant" and s.lstrip("\n").startswith("Human:")


def check_raw_record(rec: dict) -> str | None:
    """Field-level rules, in order; the first rule that fails is the reason.

    Pair-level rules (context mismatch, empty or identical replies, duplicates) need the parsed
    conversation and are applied in step 3 right after this check, with the same reason codes.
    """
    for field in REQUIRED_RAW_FIELDS:
        v = rec.get(field)
        if v is None or (isinstance(v, str) and not v.strip()):
            return f"missing:{field}"
        if not isinstance(v, str):
            return f"not_text:{field}"
    for field in REQUIRED_RAW_FIELDS:
        if not turn_structure_ok(rec[field]):
            return f"malformed_turns:{field}"
    return None


# ----------------------------------------------------------------------------- 2. processed contract
HEX64 = re.compile(r"^[0-9a-f]{64}$")
PLACEHOLDERS = ("[EMAIL]", "[PHONE]")


@dataclass(frozen=True)
class Rule:
    name: str
    rationale: str
    check: Callable[[pd.DataFrame, dict], int]   # returns the number of violating rows
    hard: bool = True
    response: str = "stop the run"


def _n(mask) -> int:
    return int(np.asarray(mask).sum())


def processed_rules(subsets: list[str], splits: list[str], pii_patterns: dict[str, str]) -> list[Rule]:
    pii = [re.compile(p) for p in pii_patterns.values()]

    def residual_pii(d: pd.DataFrame, ctx: dict) -> int:
        if not ctx.get("pii_masked"):
            return 0
        text = d["context"] + "\n" + d["response_a"] + "\n" + d["response_b"]
        return _n(text.map(lambda s: any(rx.search(s) for rx in pii)))

    def label_imbalance(d: pd.DataFrame, ctx: dict) -> int:
        # plausibility: the A/B swap should give ~50/50; flag (not stop) when a split drifts outside 45-55%
        return 0 if len(d) == 0 or 0.45 <= d["label"].mean() <= 0.55 else len(d)

    def outlier_flag_consistent(d: pd.DataFrame, ctx: dict) -> int:
        thr = ctx.get("long_outlier_threshold")
        if thr is None:
            return 0
        return _n(d["is_long_outlier"] != (np.maximum(d["len_a_words"], d["len_b_words"]) > thr))

    return [
        Rule("pair_id is a SHA-256 hex string", "Row key used for lineage and look-ups",
             lambda d, c: _n(~d["pair_id"].astype(str).str.fullmatch(HEX64))),
        Rule("pair_id is unique", "One row per preference pair", lambda d, c: int(d["pair_id"].duplicated().sum())),
        Rule("group_id is a SHA-256 hex string", "Split key; must never be empty",
             lambda d, c: _n(~d["group_id"].astype(str).str.fullmatch(HEX64))),
        Rule("no missing values", "Every column is required", lambda d, c: int(d.isna().any(axis=1).sum())),
        Rule("subset is one of the 4 known subsets", "Category domain from the data card",
             lambda d, c: _n(~d["subset"].astype(str).isin(subsets))),
        Rule("split is a known split name", "Category domain", lambda d, c: _n(~d["split"].astype(str).isin(splits))),
        Rule("label is 0 or 1", "Binary target", lambda d, c: _n(~d["label"].isin([0, 1]))),
        Rule("replies are not empty", "Nothing to compare otherwise",
             lambda d, c: _n((d["response_a"].str.strip() == "") | (d["response_b"].str.strip() == ""))),
        Rule("replies A and B differ", "Identical replies carry no preference signal",
             lambda d, c: _n(d["response_a"] == d["response_b"])),
        Rule("num_turns is at least 1", "Every context starts with a Human turn", lambda d, c: _n(d["num_turns"] < 1)),
        Rule("reply lengths are at least 1 word", "Follows from 'not empty'",
             lambda d, c: _n((d["len_a_words"] < 1) | (d["len_b_words"] < 1))),
        Rule("len_diff = len_a_words - len_b_words", "Derived column must agree with its inputs",
             lambda d, c: _n(d["len_diff"] != d["len_a_words"] - d["len_b_words"])),
        Rule("is_long_outlier matches the train-fitted threshold", "Flag must use the TRAIN threshold",
             outlier_flag_consistent),
        Rule("no e-mail or phone pattern left in the text", "PII is masked in step 3 (v1.1)", residual_pii),
        Rule("label share between 45% and 55% (plausibility)", "A/B swap should balance the labels",
             label_imbalance, hard=False, response="log a warning and investigate"),
    ]


def validation_log(frames: dict[str, pd.DataFrame], rules: list[Rule], ctx: dict) -> pd.DataFrame:
    """Apply every rule to every split; add the cross-split leakage rules. One row per (rule, split)."""
    rows = []
    for split, d in frames.items():
        for r in rules:
            v = r.check(d, ctx)
            rows.append({"rule": r.name, "split": split, "rationale": r.rationale, "violations": v,
                         "severity": "HARD" if r.hard else "WARN",
                         "response": "none needed" if v == 0 else r.response})
    # cross-split rules (entity and duplicate leakage, Lecture 5 slide 40)
    owner = pd.concat([d[["group_id"]].assign(split=s) for s, d in frames.items()])
    g_over = int((owner.groupby("group_id")["split"].nunique() > 1).sum())
    pairs = pd.concat([d["pair_id"] for d in frames.values()])
    p_over = int(pairs.duplicated().sum())
    for name, v, why in [("no group_id in two splits", g_over, "Entity leakage"),
                         ("no pair_id in two splits", p_over, "Duplicate leakage")]:
        rows.append({"rule": name, "split": "all", "rationale": why, "violations": v, "severity": "HARD",
                     "response": "none needed" if v == 0 else "stop the run"})
    return pd.DataFrame(rows)


def hard_failures(log: pd.DataFrame) -> pd.DataFrame:
    return log[(log["severity"] == "HARD") & (log["violations"] > 0)]
