# Datasheet: hh-rlhf preference pairs, processed version v1.1

This datasheet follows the questions in Gebru et al. (2021), *Datasheets for Datasets*. It covers the **processed dataset produced by this repository** (v1.1). The upstream dataset, `Anthropic/hh-rlhf`, is documented by its authors in Bai et al. (2022). Every number below is in `manifests/v1.1.json` or `docs/eda/eda_summary.json`.

## 1. Motivation

- **Purpose.** To train and evaluate a reward model that predicts which of two assistant replies a crowdworker preferred, as part of the course project for *Data Handling and Infrastructure for AI* (SETU, 2026). The processed version adds what the raw release lacks: a leakage-safe split, a balanced target, validation, quarantine and lineage.
- **Created by.** Boun Chun (BC), from the public release by Anthropic.
- **Funding.** None. Compute is the course's Google Cloud credit.

## 2. Composition

- **Instances.** Each instance is one preference pair: a conversation `context` and two final assistant replies, `response_a` and `response_b`, with `label` = 1 if A was preferred.
- **How many.** 169,352 raw pairs → 168,029 after validation (1,323 quarantined with a reason). The working set has 30,002 pairs: train 24,122, dev 2,876, test 3,004. The future reserve has 33,168 (F1 16,516, F2 16,652). 104,859 valid pairs are not used.
- **Sample of a larger set?** Yes. The working set is a deterministic, subset-stratified sample of whole prompt groups (hash order, not random). The upstream release is itself a sample of Anthropic's 2021–22 data collection. The `red-team-attempts` part of the release is excluded because it has no pairs.
- **Data per instance.** Text: `context`, `response_a`, `response_b`. Numbers: `num_turns`, reply lengths, `len_diff`. Flags: `is_long_outlier`, `refusal_a`, `refusal_b`. Identifiers: `pair_id`, `group_id`. Categories: `subset`, `split`. Step 11 adds model-ready text and standardised numeric features.
- **Label.** `label`, derived from the upstream `chosen` / `rejected` fields. The reply position is swapped by hash, so the label is about 50/50 (49.8% in train).
- **Missing information.** No null cells. "Missing" shows up as empty replies, broken turn structure, or contexts that disagree between the two replies. These are quarantined with their reason (counts in §4). There are no per-row timestamps, annotator ids or annotator demographics.
- **Relationships between instances.** Pairs that start with the same Human message share a `group_id`. 93% of pairs share their prompt with another pair; the median group has 3 pairs and the largest 48. Splits are made by group.
- **Recommended splits.** Use the provided train / dev / test, which are grouped by prompt, and keep F1/F2 for "new data" experiments. Do **not** use the official Hugging Face split: 92% of its test pairs have their opening prompt in its training split.
- **Errors, noise, redundancy.** Labels are single and noisy (about 63% researcher–crowdworker agreement, Bai et al. 2022). 0.25% of train pairs have near-identical replies. Typographic and straight apostrophes are mixed (47% vs 13% of replies).
- **Self-contained?** The processed files are self-contained. The raw files are kept unchanged in the versioned `raw/` layer, pinned to Hugging Face commit `09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa` and verified by SHA-256.
- **Confidential data.** None intended. However, 304 pairs contained e-mail addresses and 724 phone-like numbers. From v1.1 these are masked as `[EMAIL]` / `[PHONE]`.
- **Offensive content.** Yes. `harmless-base` comes from red-teaming and contains offensive and dangerous requests and replies. The text is never committed to Git, and figures show aggregates only.
- **Sub-populations.** Four subsets (collection rounds and goals): helpful-base 27.5%, helpful-rejection-sampled 32.1%, helpful-online 13.8%, harmless-base 26.7% (train). They differ strongly in reply length and in what is preferred (EDA §4.4–4.5).
- **Identifying individuals.** No direct identifiers by design. Masking removes e-mail and phone patterns. Other personal details (names, addresses) are not detected.

## 3. Collection process

- **Upstream collection.** Crowdworkers (mostly US-based MTurk, some Upwork) talked to Anthropic's 52B language models and chose the better of two replies at each turn, in three rounds: base, rejection-sampled, online (Bai et al. 2022).
- **Acquisition in this project.** `src/step1_scrape_raw.py` downloads 8 files at a pinned commit, with retries, and records their SHA-256 checksums. For the v1.1 run, the identical files were taken from Anthropic's GitHub release of the same dataset; all 8 checksums matched.
- **Sampling strategy.** Whole prompt groups, ordered by `sha256(group_id + salt)`, with quotas per subset.
- **Time frame.** Collected 2021–22, so the replies reflect 2021 model behaviour. Processed in October 2026.
- **Ethical review / consent.** Not documented in the release for the crowdworkers. Crowdworkers were paid. This project adds no new human data.

## 4. Preprocessing, cleaning, labelling

- **Done** (`make all`, steps 3–11):
  - Data contract and quarantine: 788 identical replies, 329 context mismatches, 194 empty replies, 11 malformed turns, 1 pair identical after PII masking.
  - Unicode NFC normalisation and whitespace tidying; case and punctuation kept.
  - PII masking.
  - Hash-based A/B swap.
  - Grouping and group split.
  - Long-reply flag fitted on train (210 words).
  - Refusal / non-answer flags (apostrophe-agnostic).
  - Left truncation of contexts to 256 words.
  - Log-length features standardised with the train mean and SD.
- **Raw data saved?** Yes: write-once, versioned GCS layer, plus the checksummed raw manifest.
- **Software.** This repository (Python 3.13, pandas, NumPy, scikit-learn; versions pinned in `requirements.txt`). 40 unit tests and CI on every push.

## 5. Uses

- **Intended.** Reward-model training and evaluation; studying length and refusal shortcuts; data-pipeline teaching.
- **Already used for.** Milestone 2 EDA and length / bag-of-words baselines: dev accuracy 57.9% for the length × subset model and 54.8% for bag of words.
- **Do not use for.**
  - Claims about how people in general prefer assistant behaviour (coverage: English, mostly US crowdworkers, 2021 models).
  - Training a model to *produce* the harmful content in harmless-base.
  - Reporting only pooled accuracy: per-subset results can point in opposite directions.
- **Risks.** A model can score above chance from reply length plus topic (58% on dev) or refusal style. Always evaluate per subset and per length-gap slice.

## 6. Distribution

- The processed files are **not redistributed**. They live in private GCS buckets. Anyone can rebuild them with `make all` from the public, MIT-licensed upstream release.
- Code: public GitHub repository. Licence of the upstream data: MIT.

## 7. Maintenance

- **Maintainer.** BC (GitHub `bounchun`).
- **Versioning.** `vMAJOR.MINOR`: MAJOR changes when the raw data or the split changes, MINOR when the cleaning rules change. Every version has a manifest with checksums and the code commit. v1.1 changes the cleaning rules only, and its split is identical to v1.0.
- **Errata.** Fixed in v1.1: the v1.0 refusal regex missed typographic apostrophes.
- **Contributions.** Issues and pull requests on GitHub. New data must pass the data contract, the poisoning screen and the reconciliation before it is used.
