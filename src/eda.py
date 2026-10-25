"""Milestone 2 - exploratory data analysis that drives pipeline decisions.

    python -m src.eda                 # after `make all` (needs the interim + processed files)

Question it answers (module brief): "What problems would a model encounter if we trained on the
raw data?" - and which pipeline rule, model choice or evaluation check each problem leads to.

Which rows each analysis may look at (Lecture 5 slide 39, "split before learning from the data")
  E1-E2  STRUCTURE and QUALITY, safe before the split: rule violations (the quarantine), keys,
         duplicates and group sizes, computed on all 169,352 raw / 168,030 cleaned pairs.
  E3-E6  DISTRIBUTIONS and RELATIONSHIPS: TRAIN ONLY (processed/<version>/train.parquet).
  E7     SPLIT CHECK: train vs dev profiles; test only through the counts already in the manifest
         (the holdout bucket is never read here).
  E8     SHORTCUT BASELINES: fitted on train, scored on dev. Never on test.

Inputs : <interim_dir>/<version>/features.parquet           (all cleaned pairs, from step 5)
         <main_root>/quarantine/<version>/quarantine.parquet (from step 3)
         <main_root>/processed/<version>/{train,dev}.parquet (from step 9)
         manifests/<version>.json
Outputs: <eda_dir>/fig*.png            figures for the report (no dataset text in them)
         <eda_dir>/eda_summary.json    every number quoted in the report
         <eda_dir>/findings.csv        finding -> decision -> limitation (Lecture 5 slide 50)
         <main_root>/audit/<version>/reply_openings_harmless_train.csv   (contains text: bucket only)

Only aggregate numbers and figures are written to <eda_dir> (committed to Git); anything that
contains dataset text goes to the private audit folder, because hh-rlhf contains harmful text.
"""
from __future__ import annotations

import difflib
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import chi2_contingency  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from .common import (interim, join, local_path, manifest_path, read_parquet, require, step_args, storage_roots,
                     write_bytes)

# fixed subset -> colour mapping (categorical slots 1-4 of the validated reference palette, light mode);
# the same subset always gets the same colour in every figure
SUBSETS = ["helpful-base", "helpful-rejection-sampled", "helpful-online", "harmless-base"]
COLOR = {"helpful-base": "#2a78d6", "helpful-rejection-sampled": "#eb6834",
         "helpful-online": "#1baf7a", "harmless-base": "#eda100"}
SHORT = {"helpful-base": "helpful-base", "helpful-rejection-sampled": "helpful-rej.-sampled",
         "helpful-online": "helpful-online", "harmless-base": "harmless-base"}
INK, INK2, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"


# ----------------------------------------------------------------------------- statistics helpers
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval for a proportion k/n (Lecture 5 slide 32)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (centre - half, centre + half)


def rate(k: int, n: int) -> dict:
    lo, hi = wilson(int(k), int(n))
    return {"k": int(k), "n": int(n), "rate": round(k / n, 4) if n else None, "ci95": [round(lo, 4), round(hi, 4)]}


def cramers_v(table: pd.DataFrame) -> dict:
    """Chi-square test of independence + Cramer's V effect size (Lecture 5 slide 31)."""
    chi2, p, dof, _ = chi2_contingency(table.to_numpy())
    n = table.to_numpy().sum()
    v = math.sqrt(chi2 / (n * (min(table.shape) - 1)))
    return {"chi2": round(float(chi2), 2), "dof": int(dof), "p_value": float(f"{p:.3g}"), "cramers_v": round(v, 3)}


def describe(x: pd.Series) -> dict:
    x = x.astype(float)
    q1, q3 = x.quantile([0.25, 0.75])
    fence = q3 + 1.5 * (q3 - q1)
    return {"n": int(len(x)), "mean": round(x.mean(), 1), "median": float(x.median()),
            "p90": float(x.quantile(0.90)), "p99": float(x.quantile(0.99)), "max": float(x.max()),
            "skew": round(float(x.skew()), 2), "skew_log1p": round(float(np.log1p(x).skew()), 2),
            "iqr_fence": round(float(fence), 1), "share_above_iqr_fence": round(float((x > fence).mean()), 4)}


def style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, fontweight="semibold")
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, out: Path, name: str) -> str:
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    # metadata without a timestamp -> byte-identical PNGs on a rerun
    fig.savefig(out / name, dpi=170, metadata={"Software": None, "CreationDate": None} if name.endswith(".pdf")
                else {"Software": None})
    plt.close(fig)
    return name


def preferred_cols(d: pd.DataFrame) -> pd.DataFrame:
    """Add preferred/other reply lengths (label 1 = A preferred)."""
    a_pref = d["label"] == 1
    return d.assign(len_pref=np.where(a_pref, d["len_a_words"], d["len_b_words"]),
                    len_other=np.where(a_pref, d["len_b_words"], d["len_a_words"]),
                    resp_pref=np.where(a_pref, d["response_a"], d["response_b"]),
                    resp_other=np.where(a_pref, d["response_b"], d["response_a"]))


# ----------------------------------------------------------------------------- E1 raw-data problems
def e1_raw_problems(q: pd.DataFrame, manifest: dict, out: Path) -> dict:
    """What would go wrong if we trained on the raw files? Rule violations, by subset."""
    s3 = manifest["steps"]["step3_clean"]
    raw_n = pd.Series(s3["raw_pairs"]).groupby(lambda k: k.split("/")[0]).sum()
    q = q[q["reason"] != "bad_json"]
    by = q.groupby("subset").size().reindex(SUBSETS, fill_value=0)
    per_subset = {s: rate(by[s], raw_n[s]) for s in SUBSETS}
    table = pd.DataFrame({"quarantined": by, "kept": raw_n[SUBSETS] - by})
    test = cramers_v(table)
    reasons = (q.assign(reason=q["reason"].str.replace(r":(chosen|rejected)$", "", regex=True))
               .groupby(["subset", "reason"]).size().unstack(fill_value=0).reindex(SUBSETS, fill_value=0))
    reason_rates = (reasons.div(raw_n[SUBSETS], axis=0) * 1000).round(2)

    # figure 1: quarantine rate per 1,000 raw pairs, by subset and reason (one panel per reason)
    order = [r for r in ["identical_replies", "context_mismatch", "empty_reply", "identical_after_pii_mask",
                         "malformed_turns", "missing", "not_text", "duplicate_pair"] if r in reasons.columns]
    order = [r for r in order if reasons[r].sum() >= 5][:4]   # a panel for a reason with < 5 rows says nothing
    fig, axes = plt.subplots(1, len(order) + 1, figsize=(11, 3.1), sharey=False)
    ax = axes[0]
    vals = [per_subset[s]["rate"] * 1000 for s in SUBSETS]
    err = np.array([[per_subset[s]["rate"] * 1000 - per_subset[s]["ci95"][0] * 1000,
                     per_subset[s]["ci95"][1] * 1000 - per_subset[s]["rate"] * 1000] for s in SUBSETS]).T
    ax.bar(range(4), vals, color=[COLOR[s] for s in SUBSETS], width=0.7, yerr=err, ecolor=INK2, capsize=3)
    for i, v in enumerate(vals):
        ax.text(i, v + err[1][i] + max(vals) * 0.03, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
    style(ax, "All reasons", ylabel="quarantined per 1,000 raw pairs")
    ax.set_xticks(range(4), [SHORT[s] for s in SUBSETS], rotation=35, ha="right")
    for ax, r in zip(axes[1:], order):
        v = [reason_rates.loc[s, r] for s in SUBSETS]
        ax.bar(range(4), v, color=[COLOR[s] for s in SUBSETS], width=0.7)
        for i, x in enumerate(v):
            ax.text(i, x, f"{x:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
        style(ax, {"identical_after_pii_mask": "identical after PII mask", "malformed_turns": "malformed turns",
                   "context_mismatch": "context mismatch", "identical_replies": "identical replies",
                   "empty_reply": "empty reply"}.get(r, r.replace("_", " ")))
        ax.set_xticks(range(4), [SHORT[s] for s in SUBSETS], rotation=35, ha="right")
    fig.suptitle("Fig. 1  Rule violations are not spread evenly: the quarantine rate differs by subset "
                 "(95% Wilson intervals)", x=0.01, ha="left", fontsize=10, color=INK2)
    name = save(fig, out, "fig1_quarantine_by_subset.png")

    return {"figure": name, "raw_pairs_by_subset": {s: int(raw_n[s]) for s in SUBSETS},
            "quarantine_rate_by_subset": per_subset, "independence_test_subset_x_quarantined": test,
            "reason_rate_per_1000": {s: {r: float(reason_rates.loc[s, r]) for r in reasons.columns} for s in SUBSETS},
            "position_leak_in_raw": "preferred reply is in `chosen` for 100% of raw records: a model on the raw "
                                    "columns can reach 100% by learning the column, so the A/B swap is required"}


# ----------------------------------------------------------------------------- E2 duplicates and leakage
def e2_duplicates_and_leakage(feat: pd.DataFrame, train: pd.DataFrame, out: Path, near_dup: float) -> dict:
    # entity repetition: many pairs share one opening prompt (the grain is the prompt, not the row)
    sizes = feat.groupby("group_id").size()
    in_multi = float((feat["group_id"].map(sizes) > 1).mean())

    # official Hugging Face train/test split: the same opening prompt on both sides
    src = feat.groupby("group_id")["source_split"].agg(set)
    both = src.map(lambda s: {"train", "test"} <= s)
    test_rows = feat[feat["source_split"] == "test"]
    test_leak = float(test_rows["group_id"].map(both).mean())

    # contradictory or repeated comparisons: same context + same two replies seen more than once
    a, b = feat["response_a"].to_numpy(), feat["response_b"].to_numpy()
    pref = np.where(feat["label"] == 1, a, b)
    lo, hi = np.where(a < b, a, b), np.where(a < b, b, a)
    key = pd.Series([hash((c, x, y)) for c, x, y in zip(feat["context"], lo, hi)], index=feat.index)
    rep = key.map(key.value_counts()) > 1
    n_pref = pd.Series(pref, index=feat.index).groupby(key).nunique()
    contradictory_keys = int((n_pref > 1).sum())

    # near-identical replies in TRAIN (word-level similarity of the two replies)
    ratio = np.array([difflib.SequenceMatcher(None, x.split(), y.split()).ratio()
                      for x, y in zip(train["response_a"], train["response_b"])])
    near = ratio >= near_dup
    near_by = {s: rate(int(near[(train["subset"] == s).to_numpy()].sum()), int((train["subset"] == s).sum()))
               for s in SUBSETS}

    # figure 2: (a) pairs per opening prompt, (b) reply similarity
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.2))
    ax = axes[0]
    counts = sizes.clip(upper=20).value_counts().sort_index()
    ax.bar(counts.index, counts.values, color="#2a78d6", width=0.8)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 3, 5, 10, 15, 20], ["1", "2", "3", "5", "10", "15", "20+"])
    style(ax, "(a) Pairs per opening prompt (all cleaned pairs)", "pairs in the group (20 = 20 or more)",
          "number of groups (log scale)")
    ax.text(0.98, 0.95, f"{len(sizes):,} groups for {len(feat):,} pairs\n{in_multi:.0%} of pairs share their "
            f"prompt\nlargest group: {int(sizes.max())} pairs", transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, color=INK)
    ax = axes[1]
    bins = np.linspace(0, 1, 41)
    for s in SUBSETS:
        m = (train["subset"] == s).to_numpy()
        h, _ = np.histogram(ratio[m], bins=bins)
        ax.plot(bins[:-1] + 0.0125, h / m.sum(), color=COLOR[s], linewidth=2, label=SHORT[s])
    ax.axvline(near_dup, color=NEUTRAL, linestyle="--", linewidth=1)
    ax.text(near_dup, ax.get_ylim()[1] * 0.92, f" ≥{near_dup}: near-identical\n {near.mean():.1%} of train",
            fontsize=8, color=INK2, va="top")
    style(ax, "(b) How similar are the two replies? (train)", "word-level similarity of reply A and reply B",
          "share of pairs")
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Fig. 2  Rows are not independent, and some pairs barely differ", x=0.01, ha="left",
                 fontsize=10, color=INK2)
    name = save(fig, out, "fig2_groups_and_similarity.png")

    return {"figure": name, "groups": int(len(sizes)), "pairs": int(len(feat)),
            "share_pairs_in_shared_prompt_groups": round(in_multi, 4), "largest_group": int(sizes.max()),
            "group_size_quantiles": {k: float(v) for k, v in sizes.quantile([0.5, 0.9, 0.99]).items()},
            "official_split_groups_in_both": int(both.sum()),
            "official_test_pairs_whose_prompt_is_in_official_train": round(test_leak, 4),
            "repeated_comparisons_pairs": int(rep.sum()), "contradictory_comparison_keys": contradictory_keys,
            "near_identical_threshold": near_dup,
            "train_near_identical": rate(int(near.sum()), len(train)), "train_near_identical_by_subset": near_by,
            "train_similarity_median": round(float(np.median(ratio)), 3)}


# ----------------------------------------------------------------------------- E3 distributions (train)
def e3_distributions(train: pd.DataFrame, out: Path, max_ctx_words: int) -> dict:
    lens = pd.concat([train[["subset", "len_a_words"]].rename(columns={"len_a_words": "w"}),
                      train[["subset", "len_b_words"]].rename(columns={"len_b_words": "w"})])
    ctx_words = train["context"].str.split().str.len()
    thr99 = float(np.percentile(lens["w"], 99))
    stats = {"reply_words_all": describe(lens["w"]),
             "reply_words_by_subset": {s: describe(lens.loc[lens["subset"] == s, "w"]) for s in SUBSETS},
             "context_words": describe(ctx_words),
             "num_turns_share": {str(k): round(float(v), 4) for k, v in
                                 train["num_turns"].clip(upper=5).value_counts(normalize=True).sort_index().items()},
             "num_turns_by_subset_mean": {s: round(float(train.loc[train["subset"] == s, "num_turns"].mean()), 2)
                                          for s in SUBSETS},
             "context_over_max_words": rate(int((ctx_words > max_ctx_words).sum()), len(train)),
             "max_context_words_setting": max_ctx_words, "reply_p99_train": thr99}

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))
    ax = axes[0]
    for s in SUBSETS:
        x = np.sort(lens.loc[lens["subset"] == s, "w"].to_numpy())
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=COLOR[s], linewidth=2, label=SHORT[s])
    ax.set_xscale("log")
    ax.axvline(thr99, color=NEUTRAL, linestyle="--", linewidth=1)
    ax.text(thr99, 0.05, f" 99th pct\n {thr99:.0f} words", fontsize=8, color=INK2)
    style(ax, "(a) Reply length, ECDF (train)", "words per reply (log scale)", "cumulative share")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax = axes[1]
    x = np.sort(ctx_words.to_numpy())
    ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color="#2a78d6", linewidth=2)
    ax.set_xscale("log")
    ax.axvline(max_ctx_words, color=NEUTRAL, linestyle="--", linewidth=1)
    ax.text(max_ctx_words, 0.05, f" {max_ctx_words} words:\n {stats['context_over_max_words']['rate']:.1%} longer",
            fontsize=8, color=INK2)
    style(ax, "(b) Context length, ECDF (train)", "words before the final reply (log scale)", "cumulative share")
    ax = axes[2]
    turns = (train.assign(t=train["num_turns"].clip(upper=5)).groupby(["subset", "t"]).size()
             .unstack(fill_value=0).reindex(SUBSETS))
    turns = turns.div(turns.sum(axis=1), axis=0)
    w = 0.2
    for i, s in enumerate(SUBSETS):
        ax.bar(turns.columns + (i - 1.5) * w, turns.loc[s], width=w - 0.02, color=COLOR[s], label=SHORT[s])
    ax.set_xticks(turns.columns, [str(t) if t < 5 else "5+" for t in turns.columns])
    style(ax, "(c) Human turns (train)", "Human turns in the context (5 = 5 or more)", "share of pairs")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.suptitle("Fig. 3  Lengths are heavily right-skewed and differ by subset", x=0.01, ha="left",
                 fontsize=10, color=INK2)
    stats["figure"] = save(fig, out, "fig3_length_distributions.png")
    return stats


# ----------------------------------------------------------------------------- E4 length vs preference (train)
def e4_length_bias(train: pd.DataFrame, out: Path, bins: list[int]) -> dict:
    d = preferred_cols(train)
    d = d[d["len_pref"] != d["len_other"]].assign(pref_longer=lambda x: x["len_pref"] > x["len_other"])
    by = {s: rate(int(d.loc[d["subset"] == s, "pref_longer"].sum()), int((d["subset"] == s).sum())) for s in SUBSETS}
    pooled = rate(int(d["pref_longer"].sum()), len(d))
    test = cramers_v(pd.crosstab(d["subset"], d["pref_longer"]))
    gap = (d["len_pref"] - d["len_other"]).abs()
    d = d.assign(gap_bin=pd.cut(gap, bins=bins, right=False))
    curve = {s: {str(b): rate(int(g["pref_longer"].sum()), len(g))
                 for b, g in d[d["subset"] == s].groupby("gap_bin", observed=True)} for s in SUBSETS}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.3), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    for i, s in enumerate(SUBSETS):
        r = by[s]
        ax.errorbar(i, r["rate"], yerr=[[r["rate"] - r["ci95"][0]], [r["ci95"][1] - r["rate"]]], fmt="o",
                    color=COLOR[s], markersize=8, capsize=4, linewidth=2)
        ax.text(i + 0.1, r["ci95"][1], f"{r['rate']:.1%}", va="bottom", fontsize=8.5, color=INK)
    ax.axhline(0.5, color=NEUTRAL, linewidth=1)
    ax.axhline(pooled["rate"], color=INK2, linestyle=":", linewidth=1.2)
    ax.text(3.6, pooled["rate"], f"pooled {pooled['rate']:.1%}", fontsize=8, color=INK2, va="bottom", ha="right")
    ax.text(-0.45, 0.5, "no length effect", fontsize=7.5, color=INK2, va="top")
    ax.set_xticks(range(4), [SHORT[s] for s in SUBSETS], rotation=30, ha="right")
    ax.set_xlim(-0.5, 3.65)
    style(ax, "(a) Preferred reply is the longer one (train)", "", "share of pairs (95% CI)")
    ax = axes[1]
    labels = [f"{b}" for b in d["gap_bin"].cat.categories]
    for s in SUBSETS:
        ys = [curve[s].get(lab, {}).get("rate", np.nan) for lab in labels]
        ax.plot(range(len(labels)), ys, marker="o", markersize=6, color=COLOR[s], linewidth=2, label=SHORT[s])
    ax.axhline(0.5, color=NEUTRAL, linewidth=1)
    ax.set_xticks(range(len(labels)), [f"{c.left}–{c.right - 1}" if np.isfinite(c.right) and c.right < 10**5
                                       else f"{c.left}+" for c in d["gap_bin"].cat.categories], fontsize=8)
    style(ax, "(b) ... by how much longer it is", "length gap between the two replies (words)",
          "share where longer is preferred")
    ax.legend(frameon=False, fontsize=7.5)
    fig.suptitle("Fig. 4  Length predicts the label, but in opposite directions per subset", x=0.01, ha="left",
                 fontsize=10, color=INK2)
    name = save(fig, out, "fig4_length_bias.png")
    return {"figure": name, "pairs_with_unequal_length": int(len(d)), "preferred_is_longer_by_subset": by,
            "preferred_is_longer_pooled": pooled, "independence_test_subset_x_pref_longer": test,
            "by_length_gap": curve}


# ----------------------------------------------------------------------------- E5 refusals (train)
def e5_refusals(train: pd.DataFrame, root: str, version: str, out: Path, n_words: int = 3) -> dict:
    d = preferred_cols(train)
    flag = train["refusal_a"] | train["refusal_b"]
    harm = d[d["subset"] == "harmless-base"]
    opening = lambda s: " ".join(s.split()[:n_words]).lower()  # noqa: E731
    pref = Counter(harm["resp_pref"].map(opening))
    other = Counter(harm["resp_other"].map(opening))
    rows = [{"opening": o, "replies": pref[o] + other[o], "share_preferred": round(pref[o] / (pref[o] + other[o]), 3)}
            for o in set(pref) | set(other)]
    top = pd.DataFrame(rows).sort_values(["replies", "opening"], ascending=[False, True]).head(40)
    url = join(root, "audit", version, "reply_openings_harmless_train.csv")
    write_bytes(url, top.to_csv(index=False).encode())   # contains text -> private bucket only
    top20 = top.head(20)
    # apostrophes: the data mixes the straight ' and the typographic ’ (regexes must accept both)
    replies = pd.concat([train["response_a"], train["response_b"]])
    curly, straight = replies.str.contains("’", regex=False), replies.str.contains("'", regex=False)
    # when exactly one reply is flagged, how often is the flagged one preferred? (per subset)
    one = train["refusal_a"] ^ train["refusal_b"]
    pref_flag = np.where(train["label"] == 1, train["refusal_a"], train["refusal_b"])
    by = {sub: {"any_flag": rate(int(flag[train["subset"] == sub].sum()), int((train["subset"] == sub).sum())),
                "flagged_reply_preferred_when_one_side":
                    rate(int(pref_flag[(one & (train["subset"] == sub)).to_numpy()].sum()),
                         int((one & (train["subset"] == sub)).sum()))} for sub in SUBSETS}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.2))
    for ax, key, title, ylab in [
            (axes[0], "any_flag", "(a) Pairs with a refusal / non-answer flag (train)", "share of pairs"),
            (axes[1], "flagged_reply_preferred_when_one_side", "(b) ... when only one reply is flagged, it is preferred",
             "share preferred (95% CI)")]:
        for i, sub in enumerate(SUBSETS):
            r = by[sub][key]
            if not r["n"]:
                continue
            ax.errorbar(i, r["rate"], yerr=[[r["rate"] - r["ci95"][0]], [r["ci95"][1] - r["rate"]]], fmt="o",
                        color=COLOR[sub], markersize=8, capsize=4, linewidth=2)
            ax.text(i + 0.1, r["ci95"][1], f"{r['rate']:.1%}\nn={r['n']:,}", va="bottom", fontsize=7.5, color=INK)
        ax.set_xticks(range(4), [SHORT[x] for x in SUBSETS], rotation=25, ha="right")
        ax.set_xlim(-0.5, 3.8)
        top = max(by[x][key]["ci95"][1] for x in SUBSETS if by[x][key]["n"]) if any(by[x][key]["n"] for x in SUBSETS) else 1
        ax.set_ylim(0, top * 1.3)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        style(ax, title, "", ylab)
    axes[1].axhline(0.5, color=NEUTRAL, linewidth=1)
    axes[1].set_ylim(0, 1.1)
    fig.suptitle("Fig. 6  Refusing is rewarded in harmless-base and penalised in the helpful subsets",
                 x=0.01, ha="left", fontsize=10, color=INK2)
    fig_name = save(fig, out, "fig6_refusals.png")
    return {"figure": fig_name, "refusal_flag_any_share_train": rate(int(flag.sum()), len(train)), "refusal_by_subset": by,
            "replies_with_typographic_apostrophe": rate(int(curly.sum()), len(replies)),
            "replies_with_straight_apostrophe": rate(int(straight.sum()), len(replies)),
            "harmless_top20_openings_cover": round(float(top20["replies"].sum() / (2 * len(harm))), 4) if len(harm) else 0,
            "openings_table": f"<main_root>/audit/{version}/reply_openings_harmless_train.csv", "note": "openings are text, so they are kept in the private audit folder"}


# ----------------------------------------------------------------------------- E7 split profiles
def e7_split_profiles(train: pd.DataFrame, dev: pd.DataFrame, manifest: dict) -> dict:
    def prof(d):
        return {"pairs": int(len(d)), "label_1_share": round(float(d["label"].mean()), 4),
                "median_reply_words": float(np.median(np.r_[d["len_a_words"], d["len_b_words"]])),
                "median_context_words": float(d["context"].str.split().str.len().median()),
                "mean_num_turns": round(float(d["num_turns"].mean()), 2),
                "long_outlier_share": round(float(d["is_long_outlier"].mean()), 4),
                "subset_share": {s: round(float((d["subset"] == s).mean()), 4) for s in SUBSETS}}
    sp = manifest["steps"]["step7_split"]["splits"]
    test = {k: sp["test"][k] for k in ["pairs", "label_1_share", "long_outlier_share", "subset_share"]}
    n_dev = len(dev)
    return {"train": prof(train), "dev": prof(dev), "test_from_manifest": test,
            "dev_proportion_standard_error": round(math.sqrt(0.25 / n_dev), 4) if n_dev else None}


# ----------------------------------------------------------------------------- E8 shortcut baselines (train -> dev)
def length_features(d: pd.DataFrame) -> np.ndarray:
    a, b = np.log1p(d["len_a_words"].to_numpy()), np.log1p(d["len_b_words"].to_numpy())
    return np.c_[a, b, a - b, np.log1p(d["num_turns"].to_numpy())]


def e8_baselines(train: pd.DataFrame, dev: pd.DataFrame, out: Path) -> dict:
    res = {}
    maj = int(train["label"].mode().iloc[0])
    res["majority_class"] = {"predict": maj, "dev": rate(int((dev["label"] == maj).sum()), len(dev))}
    longer = np.where(dev["len_a_words"] >= dev["len_b_words"], 1, 0)
    res["always_pick_longer"] = {"dev": rate(int((longer == dev["label"]).sum()), len(dev))}
    model = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    model.fit(length_features(train), train["label"])                 # fitted on TRAIN only
    pred = model.predict(length_features(dev))
    res["logreg_length_only"] = {"features": ["log1p(len_a)", "log1p(len_b)", "log-ratio", "log1p(num_turns)"],
                                 "dev": rate(int((pred == dev["label"]).sum()), len(dev))}
    # the same length features, but one model per subset: a text model can tell the subsets apart
    # from the topic, so this estimates how far "length + topic" alone could go
    pred_sub = np.zeros(len(dev), dtype=int)
    for s in SUBSETS:
        mt, md = (train["subset"] == s).to_numpy(), (dev["subset"] == s).to_numpy()
        if mt.sum() and md.sum():
            ms = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
            ms.fit(length_features(train[mt]), train["label"][mt])
            pred_sub[md] = ms.predict(length_features(dev[md]))
    res["logreg_length_per_subset"] = {"dev": rate(int((pred_sub == dev["label"].to_numpy()).sum()), len(dev))}
    per = {}
    for s in SUBSETS:
        m = (dev["subset"] == s).to_numpy()
        per[s] = {"always_pick_longer": rate(int((longer[m] == dev["label"].to_numpy()[m]).sum()), int(m.sum())),
                  "logreg_length_only": rate(int((pred[m] == dev["label"].to_numpy()[m]).sum()), int(m.sum())),
                  "logreg_length_per_subset": rate(int((pred_sub[m] == dev["label"].to_numpy()[m]).sum()),
                                                   int(m.sum()))}
    res["by_subset_dev"] = per

    fig, ax = plt.subplots(figsize=(8.6, 3.7))
    names = ["always_pick_longer", "logreg_length_only", "logreg_length_per_subset"]
    nice = {"always_pick_longer": "always pick the longer reply", "logreg_length_only": "logistic regression on lengths",
            "logreg_length_per_subset": "same, one model per subset"}
    groups = SUBSETS + ["all"]
    w = 0.27
    for j, nm in enumerate(names):
        vals = [per[s][nm] for s in SUBSETS] + [res[nm]["dev"]]
        y = [v["rate"] for v in vals]
        e = np.array([[v["rate"] - v["ci95"][0], v["ci95"][1] - v["rate"]] for v in vals]).T
        ax.bar(np.arange(5) + (j - 1) * w, y, width=w - 0.03, yerr=e, capsize=2, ecolor=INK2,
               color=["#2a78d6", "#eb6834", "#1baf7a"][j], label=nice[nm])
        for i, v in enumerate(y):
            ax.text(i + (j - 1) * w, e[1][i] + v + 0.005, f"{v:.0%}", ha="center", fontsize=7, color=INK)
    ax.axhline(0.5, color=NEUTRAL, linewidth=1)
    ax.text(4.75, 0.5, "chance", fontsize=8, color=INK2, va="bottom", ha="right")
    ax.set_xticks(range(5), [SHORT.get(g, "all subsets") for g in groups], rotation=20, ha="right")
    ax.set_ylim(0.3, max(0.8, ax.get_ylim()[1] + 0.05))
    style(ax, "", "", "dev accuracy (95% CI)")
    fig.suptitle("Fig. 5  Length-only shortcuts, fitted on train and scored on dev", x=0.01, ha="left",
                 fontsize=10, color=INK2)
    ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.36), ncol=3)
    res["figure"] = save(fig, out, "fig5_length_baselines_dev.png")
    return res


# ----------------------------------------------------------------------------- findings table
def findings(s: dict) -> pd.DataFrame:
    e1, e2, e3, e4, e8 = s["E1_raw_problems"], s["E2_duplicates_leakage"], s["E3_distributions"], \
        s["E4_length_bias"], s["E8_baselines"]
    by = e4["preferred_is_longer_by_subset"]
    hb, hm = by["helpful-base"]["rate"], by["harmless-base"]["rate"]
    rows = [
        ("Position leak: in the raw files the preferred reply is always `chosen` (label would be 100% one class)",
         "Swap A/B by hash(pair_id) in step 5; label share is now "
         f"{s['E7_split_profiles']['train']['label_1_share']:.1%} in train", "None (deterministic and checked)"),
        (f"Rule violations differ by subset (quarantine rate: chi2 p = {e1['independence_test_subset_x_quarantined']['p_value']:.2g},"
         f" Cramer's V = {e1['independence_test_subset_x_quarantined']['cramers_v']}): 'missingness' is not MCAR",
         "Quarantine with reason + source line (not silent drop); report quarantine rate per subset every run",
         "Rates are small; the reason why subsets differ (collection process) cannot be tested from the data"),
        (f"{e2['share_pairs_in_shared_prompt_groups']:.0%} of pairs share their opening prompt; "
         f"{e2['official_test_pairs_whose_prompt_is_in_official_train']:.1%} of official test pairs have their prompt in official train",
         "Keep the group split (step 7) instead of the official split; contract rule 'no group_id in two splits'",
         "Grouping on the first turn cannot catch paraphrased prompts"),
        (f"{e2['train_near_identical']['rate']:.1%} of train pairs have near-identical replies "
         f"(similarity >= {e2['near_identical_threshold']})",
         "Keep them (real comparisons) but add an evaluation slice for near-identical pairs in M3",
         "Labels on almost-equal replies are the noisiest; they cap achievable accuracy"),
        (f"Reply length is right-skewed (skew {e3['reply_words_all']['skew']} -> {e3['reply_words_all']['skew_log1p']} after log1p); "
         f"the 1.5 IQR fence would flag {e3['reply_words_all']['share_above_iqr_fence']:.1%} of replies",
         "Flag (not delete) replies above the train 99th percentile; use log1p lengths for numeric features",
         "Long replies are valid data; flagging is only for evaluation slices"),
        (f"{e3['context_over_max_words']['rate']:.1%} of train contexts exceed {e3['max_context_words_setting']} words",
         f"Model inputs keep the last {e3['max_context_words_setting']} words of the context (step 11, left truncation)",
         "Early turns are lost for long conversations"),
        (f"Preferred reply is longer in {hb:.1%} of helpful-base pairs but only {hm:.1%} of harmless-base "
         f"(Cramer's V = {e4['independence_test_subset_x_pref_longer']['cramers_v']}); pooled {e4['preferred_is_longer_pooled']['rate']:.1%} hides this",
         "Length is a shortcut: never report pooled accuracy alone; evaluate per subset and per length-gap slice",
         "Association, not causation: length may stand for informativeness or for refusing"),
        (f"Length alone scores {e8['logreg_length_only']['dev']['rate']:.1%} on dev when pooled, but "
         f"{e8['logreg_length_per_subset']['dev']['rate']:.1%} with one length model per subset "
         f"(majority class: {e8['majority_class']['dev']['rate']:.1%})",
         "Baselines for M3: the reward model must beat the per-subset length model on every subset, not just pooled",
         "Dev only (test untouched); dev CI is about +/-1.8 points"),
        (f"The text mixes typographic and straight apostrophes ({s['E5_refusals']['replies_with_typographic_apostrophe']['rate']:.0%}"
         f" vs {s['E5_refusals']['replies_with_straight_apostrophe']['rate']:.0%} of replies), so the v1.0 refusal regex"
         f" missed most refusals; v1.1 flags {s['E5_refusals']['refusal_flag_any_share_train']['rate']:.1%} of train pairs",
         "Keep the text as is (ids stay stable); every regex accepts both apostrophes; refusal phrases rebuilt from "
         "the most common harmless-base openings in train", "Flag = refusal or non-answer; precision checked only by "
         "the manual review sheet"),
        (f"When only one reply is flagged as a refusal, it is preferred in "
         f"{s['E5_refusals']['refusal_by_subset']['harmless-base']['flagged_reply_preferred_when_one_side']['rate']:.0%}"
         f" of harmless-base pairs but {s['E5_refusals']['refusal_by_subset']['helpful-base']['flagged_reply_preferred_when_one_side']['rate']:.0%}"
         " of helpful-base pairs", "Refusal is a second subset-dependent shortcut: report accuracy on the refusal slice "
         "per subset in M3 (helpfulness vs harmlessness trade-off)", "Small n outside harmless-base; regex flag, not a "
         "human judgement"),
    ]
    return pd.DataFrame(rows, columns=["finding", "decision", "limitation"])


def main(argv: list[str] | None = None) -> None:
    _, cfg = step_args(__doc__, argv)
    version = cfg["dataset"]["version"]
    root, _ = storage_roots(cfg)
    eda_cfg = cfg["eda"]
    out = local_path(cfg, "eda_dir")
    feat_path = interim(cfg, "features.parquet")
    require(feat_path, "step5_features (run `make all` first)")
    require(manifest_path(cfg), "the pipeline")
    manifest = json.loads(manifest_path(cfg).read_text())
    feat = read_parquet(feat_path)
    q = read_parquet(join(root, "quarantine", version, "quarantine.parquet"))
    train = read_parquet(join(root, "processed", version, "train.parquet"))
    dev = read_parquet(join(root, "processed", version, "dev.parquet"))
    for d in (feat, train, dev, q):
        if "subset" in d:
            d["subset"] = d["subset"].astype(str)

    summary = {"dataset_version": version, "inputs": {"cleaned_pairs": len(feat), "train": len(train), "dev": len(dev),
                                                      "quarantined": len(q)}}
    print("E1 raw-data problems ...")
    summary["E1_raw_problems"] = e1_raw_problems(q, manifest, out)
    print("E2 duplicates, groups and leakage ...")
    summary["E2_duplicates_leakage"] = e2_duplicates_and_leakage(feat, train, out, eda_cfg["near_identical_similarity"])
    print("E3 distributions (train) ...")
    summary["E3_distributions"] = e3_distributions(train, out, cfg["model_inputs"]["max_context_words"])
    print("E4 length vs preference (train) ...")
    summary["E4_length_bias"] = e4_length_bias(train, out, eda_cfg["length_gap_bins"])
    print("E5 refusals (train) ...")
    summary["E5_refusals"] = e5_refusals(train, root, version, out)
    summary["E6_pii_masking"] = manifest["steps"]["step3_clean"].get("pii_masking", {})
    summary["E7_split_profiles"] = e7_split_profiles(train, dev, manifest)
    print("E8 shortcut baselines (train -> dev) ...")
    summary["E8_baselines"] = e8_baselines(train, dev, out)

    table = findings(summary)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    table.to_csv(out / "findings.csv", index=False)
    print(f"\n{len(table)} findings -> {out / 'findings.csv'}\nSummary -> {out / 'eda_summary.json'}")
    for f in sorted(out.glob("fig*.png")):
        print(f"  figure {f.name}")


if __name__ == "__main__":
    main()
