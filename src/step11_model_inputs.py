"""STEP 11 - Build model-ready inputs (the curated / "gold" layer), fitted on TRAIN only.

    python -m src.step11_model_inputs

Input : <main_root>/processed/<version>/{train,dev}.parquet         (from step 9)
        <holdout_root>/<version>/{test,future_f1,future_f2}.parquet   (from step 9; pipeline account only)
Output: <main_root>/processed/<version>/model_inputs/{train,dev}.parquet
        <holdout_root>/<version>/model_inputs/{test,future_f1,future_f2}.parquet
        <main_root>/processed/<version>/model_inputs/numeric_scaler.json   (train means / SDs)
        manifest: steps.step11_model_inputs

What it does (each choice comes from an EDA finding, see docs/eda/findings.csv)
1. Text inputs for a pairwise reward model: text_a / text_b = context + "\\n\\nAssistant: " + reply.
   Long contexts are LEFT-truncated by whole turns (oldest turns dropped first) to at most
   model_inputs.max_context_words words, so the most recent turns and the reply are always kept;
   if the last Human turn alone is still too long, only its last words are kept.
   context_truncated records whether this happened (an evaluation slice for M3).
2. Numeric features for baselines: log1p of the reply lengths (right-skewed, EDA E3), their log
   ratio, log1p(num_turns) and log1p(context words), standardised with the mean and SD of the
   TRAIN rows only and the same numbers applied to every split (Lecture 5: fit on train, transform
   all). The parameters are saved as JSON (readable, no pickles) and in the manifest.
Nothing is dropped: every split keeps exactly the rows step 9 stored (asserted).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .common import (SPLITS, join, manifest_path, parquet_bytes, read_parquet, sha256_bytes, step_args, storage_roots,
                     update_manifest, write_bytes)
from .text_utils import ROLE_RE

NUMERIC = ["num_log_len_a", "num_log_len_b", "num_log_len_ratio", "num_log_turns", "num_log_context_words"]
COLUMNS = ["pair_id", "group_id", "subset", "split", "label", "text_a", "text_b", "context_truncated",
           "context_words_kept", *NUMERIC]


def truncate_context(context: str, max_words: int) -> tuple[str, bool]:
    """Drop the oldest turns until the context has at most max_words words."""
    if len(context.split()) <= max_words:
        return context, False
    # split into turns, keeping the "Human:" / "Assistant:" markers with their text
    text = "\n\n" + context.lstrip("\n")
    starts = [m.start() for m in ROLE_RE.finditer(text)]
    turns = [text[a:b] for a, b in zip(starts, starts[1:] + [len(text)])]
    kept: list[str] = []
    words = 0
    for t in reversed(turns):
        n = len(t.split())
        if words + n > max_words:
            break
        kept.insert(0, t)
        words += n
    if not kept:  # the last turn alone is too long: keep its marker and its last words
        last = turns[-1]
        role, body = last.split(":", 1)
        kept = [f"{role}: " + " ".join(body.split()[-(max_words - 1):])]
    return "".join(kept).lstrip("\n"), True


def raw_numeric(d: pd.DataFrame, ctx_words: pd.Series) -> np.ndarray:
    la, lb = np.log1p(d["len_a_words"].to_numpy(float)), np.log1p(d["len_b_words"].to_numpy(float))
    return np.c_[la, lb, la - lb, np.log1p(d["num_turns"].to_numpy(float)), np.log1p(ctx_words.to_numpy(float))]


def build(d: pd.DataFrame, max_words: int, mu: np.ndarray, sd: np.ndarray) -> pd.DataFrame:
    ctx = d["context"].map(lambda c: truncate_context(c, max_words))
    kept = ctx.str[0]
    z = (raw_numeric(d, d["context"].str.split().str.len()) - mu) / sd
    out = pd.DataFrame({
        "pair_id": d["pair_id"].astype(str), "group_id": d["group_id"].astype(str),
        "subset": d["subset"].astype(str), "split": d["split"].astype(str), "label": d["label"].astype("int8"),
        "text_a": kept + "\n\nAssistant: " + d["response_a"], "text_b": kept + "\n\nAssistant: " + d["response_b"],
        "context_truncated": ctx.str[1].astype(bool),
        "context_words_kept": kept.str.split().str.len().astype("int32"),
    })
    for i, name in enumerate(NUMERIC):
        out[name] = z[:, i].astype("float32")
    return out[COLUMNS].sort_values("pair_id").reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    _, cfg = step_args(__doc__, argv)
    version = cfg["dataset"]["version"]
    root, holdout = storage_roots(cfg)
    max_words = int(cfg["model_inputs"]["max_context_words"])
    src = {s: join(root, "processed", version, f"{s}.parquet") for s in ("train", "dev")}
    src.update({s: join(holdout, version, f"{s}.parquet") for s in ("test", "future_f1", "future_f2")})
    frames = {s: read_parquet(u) for s, u in src.items()}

    train = frames["train"]
    x_train = raw_numeric(train, train["context"].str.split().str.len())
    mu, sd = x_train.mean(axis=0), x_train.std(axis=0)                 # TRAIN rows only
    sd = np.where(sd > 0, sd, 1.0)
    scaler = {"fitted_on": "train", "rows": int(len(train)), "features": NUMERIC,
              "mean": [round(float(v), 6) for v in mu], "sd": [round(float(v), 6) for v in sd],
              "max_context_words": max_words}

    files = {}
    for split in SPLITS:
        out = build(frames[split], max_words, mu, sd)
        if len(out) != len(frames[split]):
            raise SystemExit(f"ERROR: {split}: {len(frames[split])} rows in, {len(out)} out")
        base = holdout if split in ("test", "future_f1", "future_f2") else join(root, "processed")
        url = join(base, version, "model_inputs", f"{split}.parquet")
        data = parquet_bytes(out)
        write_bytes(url, data)
        files[split] = {"uri": url, "sha256": sha256_bytes(data), "rows": int(len(out)),
                        "context_truncated_share": round(float(out["context_truncated"].mean()), 4)}
        print(f"  {split:10s} {len(out):7,d} rows  truncated contexts {out['context_truncated'].mean():6.1%}  -> {url}")
    s_url = join(root, "processed", version, "model_inputs", "numeric_scaler.json")
    s_bytes = (json.dumps(scaler, indent=2) + "\n").encode()
    write_bytes(s_url, s_bytes)
    update_manifest(cfg, "step11_model_inputs", {"train_fitted_scaler": scaler,
                                                 "scaler_file": {"uri": s_url, "sha256": sha256_bytes(s_bytes)},
                                                 "files": files})
    write_bytes(join(root, "manifests", f"{version}.json"), manifest_path(cfg).read_bytes())  # refresh stored copy
    print(f"Scaler fitted on {len(train):,} train rows -> {s_url}")


if __name__ == "__main__":
    main()
