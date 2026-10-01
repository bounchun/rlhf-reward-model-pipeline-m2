"""STEP 5 - Extract features and the target.

    python -m src.step5_features

Input : <interim_dir>/<version>/cleaned.parquet   (from step 3)
Output: <interim_dir>/<version>/features.parquet
        manifest: steps.step5_features

Features (see README "data card", sections 7-8)
  group_id                sha256 of the normalised first Human turn; groups larger than
                          split.max_group_size (generic openers like "hi") are re-keyed on the
                          full normalised context. Used ONLY for leakage-safe splitting.
  response_a / response_b the two final replies; position assigned by sha256(pair_id) so the
  label                   preferred reply is not always first. label = 1 if A preferred, else 0
  num_turns               number of Human turns in the context              (int16)
  len_a_words/len_b_words reply lengths in whitespace-separated words        (int32)
  len_diff                len_a_words - len_b_words                          (int32)
  refusal_a / refusal_b   reply matches a refusal regex (config) - audit slicing only, not a model input
(is_long_outlier is added in step 7, because its threshold is fitted on TRAIN only.)
"""
from __future__ import annotations

import re

import numpy as np

from .common import (hash_bucket, interim, read_parquet, require, sha256_text, step_args, update_manifest,
                     write_parquet)
from .text_utils import count_human_turns, full_context_key, group_key, word_len


def add_features(df, cfg: dict):
    s = cfg["split"]
    first = df["context"].map(lambda c: sha256_text(group_key(c)))
    big = first.map(first.value_counts()) > s["max_group_size"]
    full = df.loc[big, "context"].map(lambda c: sha256_text(full_context_key(c)))
    df["group_id"] = first.where(~big, full)
    rekey = {"max_group_size": s["max_group_size"],
             "first_turn_groups_rekeyed": int(first[big].nunique()),
             "pairs_rekeyed_on_full_context": int(big.sum())}

    swap = np.array([hash_bucket(pid, s["position_salt"], 2) == 1 for pid in df["pair_id"]])
    df["response_a"] = np.where(swap, df["rejected_resp"], df["chosen_resp"])
    df["response_b"] = np.where(swap, df["chosen_resp"], df["rejected_resp"])
    df["label"] = np.where(swap, 0, 1).astype("int8")  # 1 = A preferred

    df["num_turns"] = df["context"].map(count_human_turns).astype("int16")
    df["len_a_words"] = df["response_a"].map(word_len).astype("int32")
    df["len_b_words"] = df["response_b"].map(word_len).astype("int32")
    df["len_diff"] = (df["len_a_words"] - df["len_b_words"]).astype("int32")

    refusal = re.compile("|".join(f"(?:{p})" for p in cfg["features"]["refusal_patterns"]), re.IGNORECASE)
    df["refusal_a"] = df["response_a"].map(lambda t: bool(refusal.search(t)))
    df["refusal_b"] = df["response_b"].map(lambda t: bool(refusal.search(t)))
    return df.drop(columns=["chosen_resp", "rejected_resp"]), rekey


def main(argv: list[str] | None = None) -> None:
    _, cfg = step_args(__doc__, argv)
    src = interim(cfg, "cleaned.parquet")
    require(src, "step3_clean")
    df, rekey = add_features(read_parquet(src), cfg)
    write_parquet(df, interim(cfg, "features.parquet"))
    update_manifest(cfg, "step5_features", {
        "group_rekeying": rekey,
        "groups": int(df["group_id"].nunique()),
        "largest_group": int(df.groupby("group_id").size().max()),
        "label_1_share": round(float(df["label"].mean()), 4),
    })
    print(f"Features for {len(df):,} pairs, {df['group_id'].nunique():,} groups "
          f"({rekey['pairs_rekeyed_on_full_context']} pairs re-keyed) -> {interim(cfg, 'features.parquet')}")


if __name__ == "__main__":
    main()
