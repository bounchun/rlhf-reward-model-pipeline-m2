# 👍👎 Response Preference Prediction (RLHF Reward Modelling) — Milestone 2

This is Milestone 2 of my Data Handling and Infrastructure for AI project. It picks up where Milestone 1 left off, in [bounchun/rlhf-reward-model-pipeline-m1](https://github.com/bounchun/rlhf-reward-model-pipeline-m1) (release `data-v1.0`), and the full commit history came across with it, so you can trace how the project got from M1 to M2. This README is self-contained: you do not need the M1 report to follow it.


### M2.0 Project summary and Milestone 1 decisions

This project prepares the data for an RLHF reward model : given a conversation and two possible replies, predict which one the crowdworker preferred. In Milestone 1 I picked the dataset, decided how to split it and where to store it, and built a pipeline that downloads, cleans, splits and uploads the data to Google Cloud Storage (release data-v1.0). Milestone 2 starts from that working pipeline and adds to it: exploratory analysis, data contracts, a quarantine for bad rows, monitoring, and an independent check of the results. The table below lists the Milestone 1 decisions I kept. The ones I changed are in [M2.4](#m24-changes-since-v10-and-why).


| Decision | Summary |
|---|---|
| Data | [`Anthropic/hh-rlhf`](https://huggingface.co/datasets/Anthropic/hh-rlhf), MIT licence, pinned to commit `09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa` |
| Dataset version | v1.1 (M1 was v1.0). Raw data and split are unchanged. M2 adds new validation checks, moves failed rows into quarantine instead of dropping them, masks personal data, and produces inputs the model can use directly |
| Task and target | A yes/no prediction: `label` = 1 if reply A was the preferred one. The model sees the conversation and the two replies. Subset, split and the flags are only used to break down the results, never as inputs |
| Split | Pairs are grouped by their first Human message (hashed), so the same prompt never ends up in two splits. 20% of the groups are held back as "future" data (F1, F2). The remaining 30,002-pair working set is split 80/10/10 using `sha256(group_id + salt) % 10`. Anything learned from the data is fitted on train only |
| Storage | Google Cloud Storage rather than a database, because the data is written once in batches and training reads whole files, with no queries in between. Two buckets: a main bucket and a holdout bucket for test, F1 and F2. Settings, access rules and evidence are in [M2.2](#m22-storage-implementation-and-evidence) |
| Formats | Raw files stay as the original JSONL.gz. Processed data is Parquet, manifests are JSON, and audit files are CSV or JSON |
| Versioning | Each version gets its own `vMAJOR.MINOR` folder, a manifest and a Git tag (`data-v1.0`, now `data-v1.1`). The MINOR number goes up when the cleaning rules change |
| Governance, bias, poisoning | Summarised for v1.1 in [Data governance and ethics](#data-governance-and-ethics), with the datasheet and the leakage audit |



## 🧭 Milestone 2 at a glance (dataset version v1.1)

| Milestone 2 requirement | Summary | Where |
|---|---|---|
| EDA that drives decisions | `make eda`: 9 figures, 13 findings, each linked to a pipeline decision and to how the data was collected | [M2.3](#m23-what-the-eda-found-and-what-it-changed) |
| Pipeline: acquire → validate → store → transform | Steps 1–11, all run by `make all` | [M2.1](#m21-pipeline-architecture) |
| Storage implementation | Two GCS buckets, one storage layer, least-privilege access verified as `train-sa` | [M2.2](#m22-storage-implementation-and-evidence) |
| Validation that fails loudly | Raw data contract with quarantine (step 3); 17-rule processed contract checked before upload (step 9) | [Six qualities](#six-qualities-of-a-trustworthy-pipeline) |
| Cleaning and transformation | Normalisation and PII masking (step 3), A/B swap (step 5), train-fitted outlier flag (step 7) and scaler (step 11) | [M2.1](#m21-pipeline-architecture) |
| Second method / reconciliation | `reconcile.py` recounts from the raw files without pipeline code; all 7 checks match | [M2.5](#m25-v11-run-on-the-real-data) |
| Reproducibility | Pinned data and libraries, hash-based splits, byte-identical reruns, 44 tests and a demo in CI on every push | [Six qualities](#six-qualities-of-a-trustworthy-pipeline) |
| Governance and ethics | Licence, harmful content, PII, access, poisoning, consent, bias, retention | [Data governance and ethics](#data-governance-and-ethics) |
| Changes since v1.0 | Narrower `train-sa` access, quarantine, data contracts, PII masking, rebuilt refusal phrases, model-ready inputs; the split is unchanged | [M2.4](#m24-changes-since-v10-and-why) |
| Limitations and reflection | What the data and the pipeline cannot show yet, and what I learned building it | [M2.6](#m26-limitations-and-reflection) |

### M2.1 Pipeline architecture

```text
make all  (Colab or locally · runs the steps in order, top to bottom,
│          and each step writes its log to manifests/v1.1.json).
│
├── 1 SOURCE  (one-off manual setup, not part of make all)
│   ├── Anthropic/hh-rlhf          Hugging Face, MIT licence, commit 09be8c5 pinned
│   │
│   └── setup_gcs.sh               creates 2 buckets (versioning + lifecycle),
│                                  2 service accounts: pipeline-sa writes, train-sa reads processed/ only
│
├── 2 INGEST
│   ├── step1_scrape_raw           downloads the 8 raw files from Hugging Face over HTTPS
│   │                              (4 subsets: helpful-base, harmless-base, helpful-online,
│   │                              helpful-rejection-sampled; a train and a test file for each)
│   │                              ──▶ data/downloads/ (local), every file's SHA-256 checked
│   │                              against the hashes pinned in manifests/raw-<commit>.json
│   │                                   
│   └── step2_store_raw            uploads the 8 files to raw/ in the main bucket, reads each one
│                                  back and checks its SHA-256 again. raw/ is write-once: a file
│                                  that already exists is never overwritten
│                                  ──▶ gs://bc-rlhf-reward-2026/raw/hh-rlhf/<commit>/
│
├── 3 VALIDATE
│   ├── step3_clean                reads the raw files back from raw/ in the bucket, checks each
│   │                              row against the data contract and masks e-mails and phone numbers
│   │                              rejected rows (reason, file, line) ──▶ main: quarantine/v1.1/
│   │                              clean rows ──▶ data/interim/ (local, not uploaded yet)
│   │   
│   ├── step4_quality_checks       runs the hard checks (any failure stops the run), screens for
│   │                              data poisoning and checks that no e-mail or phone number is left
│   │                              report + flagged pairs ──▶ main bucket: audit/v1.1/
│   │
│   └╌╌ beam_validate              optional (make beam, not in make all): runs step 3's data
│                                  contract again with Apache Beam on this machine, checks
│                                  that every count matches step 3's
│
├── 4 TRANSFORM
│   ├── step5_features             gives each pair a prompt-group id (hash of first Human message),
│   │                              puts the preferred reply in A or B by hash so the label is 0 or 1,
│   │                              and adds reply lengths and refusal flags ──▶ data/interim/ (local) 
│   │ 
│   ├── step6_filter               sets aside 20% of the prompt groups as "future" data (F1, F2)
│   │                              and picks a 30,002-pair working set from the rest,
│   │                              taking whole prompt groups only ──▶ data/interim/ (local) 
│   │  
│   ├── step7_split                splits the working set 80/10/10 into train, dev and test by the
│   │                              hash of the prompt group, so a prompt never appears in 2 splits;
│   │                              flags very long replies, with the cut-off taken from train only
│   │                              ──▶ data/interim/ (local)
│   │  
│   └── step8_shard                off by default: the data is small enough that one file per split
│                                  is quicker to write and read than many shards 
│
├── 5 STORE
│   ├── step9_store_processed      checks every split against the 17-rule data contract first;
│   │                              if a HARD rule fails, the run stops and nothing is uploaded
│   │                              contract results ──▶ main bucket: audit/v1.1/validation_log.csv
│   │                              train, dev    ──▶ main bucket: processed/v1.1/
│   │                              test, F1, F2  ──▶ holdout bucket: v1.1/
│   │    
│   ├── step10_review_sample       picks 200 train pairs (spread across the 4 subsets) for me to
│   │                              label by hand, with crowdworker's choice hidden
│   │                              review sheet  ──▶ audit/label_review_v1.1.csv
│   │                                                (local only: it contains harmful text,
│   │                                                 so it is never uploaded or committed to Git)
│   │ 
│   └── step11_model_inputs        reads 5 splits back from the buckets and gets them ready for
│                                  the model: long contexts are cut from the left
│                                  (newest turns kept, max 256 words), the log lengths are scaled
│                                  with train's mean/standard deviation,
│                                  which are saved in numeric_scaler.json for M3
│                                  train, dev, scaler ▶ main bucket: processed/v1.1/model_inputs/
│                                  test, F1, F2 ──▶ holdout bucket: v1.1/model_inputs/
│
└── 6 CHECK
    ├── sample_queries             reads train and dev back from the main bucket and runs a few
    │                              example pandas queries on them, to check the stored data makes
    │                              sense (never reads the holdout bucket)
    │                              results ──▶ main bucket: audit/v1.1/sample_queries.json
    │    
    ├── reconcile                  recounts everything straight from the raw files in plain Python,
    │                              without the pipeline code, and reads the 5 stored splits back
    │                              from the buckets; all 7 checks must match the pipeline's numbers
    │                              results ──▶ main bucket: audit/v1.1/reconciliation.json
    │
    └── eda                        reads train, dev and the quarantine from the main bucket (never
                                   the holdout bucket) and turns them into 9 figures, a summary and
                                   a findings table linking each finding to a pipeline decision
                                   figures, eda_summary.json, findings.csv ─▶ GitHub: docs/eda/
                                   (aggregates only, no dataset text, so safe to commit)


gs://bc-rlhf-reward-2026           main bucket: raw/  quarantine/v1.1/  processed/v1.1/ (+ model_inputs/)
                                   audit/v1.1/  manifests/
gs://bc-rlhf-reward-2026-holdout   holdout bucket: v1.1/ test, future_f1, future_f2 (+ model_inputs/)
                                   (settings and access rules: M2.2)

```

**ETL rather than ELT.** All the transformations (parsing the conversations, swapping A and B, hashing the groups and masking personal data) are written in Python rather than SQL. The raw files only add up to 79 MB, so everything runs in a single Python process, which takes about 2½ minutes for steps 1 to 11. Only the cleaned Parquet files get loaded into storage for use. The raw files are still kept as they are, in a write-once, versioned bucket, so I can still do what ELT is good for: change a rule and rerun everything from the raw data. I looked at Apache Beam and Dataflow (Lecture 6), but at this size they would cost more and add extra setup for no real gain. The pipeline still uses the same ideas, though: every row is checked against a data contract, rejected rows go to quarantine with a reason, each step keeps counts, reruns give the same result, and an independent recount checks the final numbers.

### M2.2 Storage: implementation and evidence

Why object storage was chosen is summarised in [M2.0](#m20-project-summary-and-milestone-1-decisions). This section is about how it works in practice: how the storage is configured, how the data is organised and accessed, and how each pipeline step uses them. The buckets were created in Milestone 1. For v1.1 I kept them, added new folders and checks, and filled them with a full Colab run on 26 October 2026.


**How it is configured.** `scripts/setup_gcs.sh` creates both buckets in one go (cell 0b of the Colab notebook runs it too). Because the setup is a script rather than manual clicks in the console, anyone can repeat it exactly.


| Setting | Main bucket `gs://bc-rlhf-reward-2026` | Holdout bucket `gs://bc-rlhf-reward-2026-holdout` | Reason |
|---|---|---|---|
| Location | `europe-west1` (Belgium) | `europe-west1` | A single region near Dublin is enough. The data can be rebuilt from the raw files, so paying extra for multi-region copies isn’t worth it|
| Storage class | Standard | Coldline | Standard for data in daily use (train, dev); Coldline is cheaper for data kept aside and opened rarely (test, F1, F2) |
| Access control | Uniform bucket-level access, public access prevention on | Same | One set of IAM rules covers the whole bucket, so no single file can be shared by mistake; the harmful text in the data must never be public |
| Object versioning | On | On | A bad rerun or an accidental delete doesn’t lose data: the previous version is kept|
| Lifecycle (`scripts/gcs_lifecycle.json`) | Delete a non-current version when 4 newer ones exist or after 90 days | Same | Keeps recent versions for recovery but deletes older ones, so costs stay small |
| IAM | `pipeline-sa`: objectAdmin on the bucket · `train-sa`: objectViewer on the `processed/` managed folder only, no role on the bucket | `pipeline-sa`: objectAdmin · `train-sa`: no role | The training account can read train and dev, and nothing else. A bucket-wide read role would also open `raw/`, which holds every pair, test, F1 and F2 included |


**Access rules, and how they are verified.** In v1.0, `train-sa` had read access to the whole main bucket. That kept it out of the holdout bucket, but not out of `raw/`, which holds all 169,352 pairs, including the ones that later become test, F1 and F2, so the test set could have been rebuilt from there. For v1.1, `processed/` is a [managed folder](https://cloud.google.com/storage/docs/managed-folders) with its own IAM policy, and `train-sa` is granted objectViewer on that folder only. `scripts/check_iam.sh` then acts as `train-sa` (service-account impersonation) and checks 8 cases: it can read and list `processed/`, and it gets a `403` on `raw/`, `quarantine/`, the bucket root, the holdout bucket, and on any write. The result is committed in [`docs/iam_check_v1.1.txt`](docs/iam_check_v1.1.txt).

**How the data is organised.** Each dataset version has its own folders (`processed/v1.1/`, `quarantine/v1.1/`, `audit/v1.1/` and `v1.1/` in the holdout bucket). That way v1.0 and v1.1 can sit side by side, and a new version never overwrites an old one. Raw files are stored under the Hugging Face commit they came from (`raw/hh-rlhf/<commit>/<subset>/<split>.jsonl.gz`), so files from another version of the dataset can't get mixed in. The interim files (`data/interim/`) are never uploaded, because until the split is made they still contain the test rows.


**How it is accessed.** Every read and write goes through one small storage layer in `src/common.py` (`write_bytes`, `read_bytes`, `read_parquet` and `exists`), which uses `gcsfs` with Application Default Credentials (Google login in Colab). The bucket name is taken from the `DHAI_BUCKET` Colab Secret or environment variable, so it doesn't appear in the code or in `config.json`, and there is no service account key file anywhere. If `DHAI_BUCKET` is left empty, the same code writes to `data/lake/` and `data/holdout/` on disk instead. That is how CI (Continuous Integration) runs the 44 tests and the offline demo without a Google Cloud account. 


**How it fits into the pipeline.** Storage isn't just where the data ends up. Several later steps read their input back from the buckets, so they work on exactly what was stored rather than on a local copy that might differ.  


| Step | Reads | Writes | Storage check |
|---|---|---|---|
| 2. Store raw | local downloads | `raw/hh-rlhf/<commit>/…` | Each file's SHA-256 is checked before upload, and again after reading it back; `raw/` is write-once, so a file with different content stops the run |
| 3. Clean | `raw/` from the bucket | `quarantine/v1.1/quarantine.parquet` | Each raw file’s SHA-256 is checked again before any row is read | 
| 4. Quality checks | local cleaned data | `audit/v1.1/` | — |
| 9. Store processed | local splits | `processed/v1.1/{train,dev}`; holdout `v1.1/{test,future_f1,future_f2}` | Every split is checked against the 17-rule contract before upload; if a HARD rule fails, no split is uploaded (only the contract log, to show what failed). Each stored file's location (URI), hash, size (bytes) and row count are saved in the manifest |
| 11. Model inputs | all 5 splits from the buckets | `processed/v1.1/model_inputs/` and holdout `v1.1/model_inputs/` | The scaling uses the train mean and standard deviation only, saved in `numeric_scaler.json` so every split is scaled the same way |
| Sample queries, EDA | train/dev and quarantine from the main bucket (EDA also uses the local cleaned data) | `audit/v1.1/`, `docs/eda/` | Never read the holdout bucket |
| Reconcile | `raw/` and all 5 stored splits from the buckets | `audit/v1.1/reconciliation.json` | Stored files must match the manifest's SHA-256 and row counts |

**Evidence.** The read-back and reconciliation results (every stored file matches the manifest's SHA-256 and row count) are in [M2.5](#m25-v11-run-on-the-real-data). In the Colab notebook, the "Verify the buckets and read a file back" cell lists both buckets and loads `dev.parquet` straight from GCS (`Rows loaded: 2,876`). Screenshots of the main bucket settings and of the four v1.1 folders after the run:

*Main bucket settings: region, uniform access, public access prevention, versioning, 2 lifecycle rules*

![bucket settings](docs/img/screenshot_bucket_settings.png)


*(a) Main bucket, `processed/v1.1/`*

![processed/v1.1](docs/img/m2_bucket_processed_v1.1.png)


*(b) Main bucket, `quarantine/v1.1/`*


![quarantine/v1.1](docs/img/m2_bucket_quarantine_v1.1.png)


*(c) Main bucket, `audit/v1.1/` (with `validation_log.csv` and `reconciliation.json`)*


![audit/v1.1](docs/img/m2_bucket_audit_v1.1.png)


*(d) Holdout bucket (Coldline), `v1.1/`*


![holdout v1.1](docs/img/m2_bucket_holdout_v1.1.png)



**What changed after Milestone 1.** The bucket design is unchanged. The changes to `train-sa`'s access and to how the pipeline uses the buckets are listed in [M2.4](#m24-changes-since-v10-and-why).

### M2.3 What the EDA found, and what it changed

All numbers come from `make eda` on the real data (`docs/eda/eda_summary.json`). Structure and quality checks cover every row; distribution and relationship analyses use the training split only; baselines are trained on train and evaluated on dev; and the holdout bucket is never accessed. 

| # | Finding (real data, v1.1) | Decision in the pipeline | Test that guards it |
|---|---|---|---|
| 1 | In the raw files the preferred reply is always in the `chosen` column: a model could score 100% by learning the column| Replies are swapped into A/B positions based on `hash(pair_id)`, so the preferred reply is now A in 49.8% of train pairs | `test_label_points_at_the_originally_chosen_reply`, `test_label_is_balanced` |
| 2 | 1,323 of the 169,352 raw pairs break at least one rule, and the failure rate varies by subset: 13.0 per 1,000 in helpful-rejection-sampled vs 4.7 in harmless-base (χ² p = 6e-65, Cramér's V = 0.04). So the problems are not "missing completely at random" (MCAR) | Failing rows go to a quarantine with their reason and source line rather than being dropped, and the pipeline asserts `rows_read = kept + quarantined` (fig. 1) | `test_every_row_read_is_kept_or_quarantined`, `test_quarantine_holds_each_planted_defect_with_its_reason_and_source` |
| 3 | 93% of pairs share their opening prompt with another pair, and 92% of official test pairs reuse a prompt from the official train split | Justifies the M1 group split; contract rule "no `group_id` in two splits" (fig. 2) | `test_no_group_or_pair_overlap_between_any_splits` |
| 4 | Reply length is right-skewed (skew 2.0, falling to −0.4 after `log1p`), and the usual 1.5×IQR rule would mark 5% of replies as outliers | Replies longer than the train 99th percentile (210 words) are flagged, not deleted; step 11 uses `log1p` lengths (fig. 3) | `test_outlier_threshold_fitted_on_train_only` |
| 5 | 9.7% of train contexts exceed 256 words | Step 11 cuts long contexts from the left, keeping the most recent turns, and marks each cut row with `context_truncated` | `test_left_truncation_keeps_the_newest_turns`, `test_scaler_is_fitted_on_train_only` |
| 6 | The preferred reply is the longer one in 58% of helpful-base pairs but in only 42% of harmless-base pairs; pooled together (52%), the two effects cancel out (Cramér's V = 0.14) | Length is a shortcut that depends on the subset: M3 reports accuracy per subset and per length-gap slice (fig. 4) | `test_eda_writes_figures_and_findings_without_reading_the_holdout` |
| 7 | A length-only model gets 51.4% on dev when pooled, but 58.2% when a separate length model is fitted for each subset | M3 must beat the best length baseline *on each subset*; the per-subset bars are in [M2.7](#m27-plan-for-milestone-3) (fig. 5) | — |
| 8 | 47% of replies use the curly apostrophe ’, so the v1.0 refusal regex (straight `'` only) flagged only 0.09% of pairs | Refusal phrases rebuilt from the data to match both apostrophes: 2.2% now flagged | `test_refusals_are_flagged_with_either_apostrophe` |
| 9 | When only one of the two replies is a refusal, the refusal is preferred in 75% of harmless-base pairs but in only 27% of helpful-base pairs | A second shortcut that depends on the subset, so M3 reports a refusal slice for each subset (fig. 6) | — |
| 10 | Some inputs are strongly correlated: the lengths of reply A and reply B have ρ = 0.50 pooled and 0.27–0.54 within subsets (partly because the subsets differ in typical reply length), and the number of turns and the context length in words have ρ = 0.82 | Linear baselines use the log length ratio instead of the two lengths; since part of the A–B correlation comes from the subsets, this is another reason to evaluate per subset (fig. 7) | — |
| 11 | Rows are clustered in prompt groups, so confidence intervals were recomputed with a group bootstrap that resamples whole prompt groups rather than single rows. The design effect is small (1.07–1.15). The gap in length preference between helpful-base and harmless-base is 16.1 points (95% CI 14.3–17.8), the length effect holds on held-out dev in all 4 subsets, and all 4 tests remain significant after Benjamini–Hochberg correction | M3 reports group-bootstrap intervals. Per-subset test accuracy is only precise to about ±3–5 points, so differences smaller than that will not be claimed (fig. 8) | `test_group_bootstrap_widens_intervals_for_clustered_rows`, `test_benjamini_hochberg` |
| 12 | Two scikit-learn Pipelines (a ColumnTransformer model on length × subset, and a bag-of-words model whose vocabulary is learned inside each fold) give the same accuracy with random-row folds as with `GroupKFold` on prompt groups (gap ≤ 0.2 points), though the group folds vary between folds | M3 uses GroupKFold for all cross-validation, and bag of words (54.8% on dev) becomes the first text baseline. Random-row folds show no measurable inflation for these reply-only models, which does not prove there is no leakage: a model that also reads the prompt could still memorise it (fig. 9) | `test_text_features_are_learned_on_the_training_fold_only` |
| 13 | 304 pairs contain an e-mail address and 724 contain something that looks like a phone number | Masked as `[EMAIL]` / `[PHONE]`; step 4 checks that none is left | `test_pii_is_masked_but_ids_come_from_the_source_text` |

#### What the findings mean for a reward model

The table above says *what* the data shows. This part explains *why*, using how hh-rlhf was collected (Bai et al., 2022), and what that means for the model in M3.

- **The subsets measure two different goals (findings 6, 7, 9).** In the helpful subsets, crowdworkers picked the more *helpful* reply. In harmless-base, they tried to get the model to say something harmful (red-teaming), and the preferred reply in the data is the *less harmful* one. On a harmful request, the safe answer is usually short, often a refusal, so harmless-base prefers the shorter reply (the longer one wins only 42% of the time) and prefers the refusal (75%). On an ordinary request, a refusal or a thin answer is unhelpful, so helpful-base prefers the longer reply (58%) and rarely prefers the refusal (27%). Bai et al. describe the same tension between helpfulness and harmlessness. Length and refusal are therefore not noise to remove: they stand in for the two goals, and any global rule ("longer is better") will be wrong on one subset by design. This is why a single pooled accuracy is never reported on its own.
- **Length is the shortcut a reward model is most likely to learn, and the costliest (findings 6, 7).** In RLHF, the policy is trained to maximise the reward model's score, so whatever the reward model rewards gets amplified. A reward model that has learned "longer is better" teaches the policy to pad its answers, a known failure of RLHF (Singhal et al., 2023). That is why M3 reports accuracy per length-gap slice: a model that only does well when the two replies differ a lot in length has learned length, not quality.
- **helpful-online is the hardest subset for length (finding 7).** Its replies are much longer (mean 111 words, against 39 in helpful-base), and the longer one wins only 45% of the time, so "pick the longer reply" scores 44%, below chance. These pairs come from the later "online" round, where crowdworkers compared replies from models already improved with RLHF. When both replies are long and fluent, length no longer separates them and the preference has to come from the content. A reward model meets this situation more and more as the policy it is training improves, so this subset gets its own bar in M2.7.
- **helpful-rejection-sampled's high quarantine rate comes from how it was collected, not from corrupt data (finding 2).** Almost all of its quarantined rows are pairs with two identical replies (12.0 per 1,000, against 0–2.5 in the other subsets). In this round, both replies were sampled from the same model and then filtered (rejection sampling), which makes identical samples more likely. An identical pair carries no preference signal, so quarantining it loses nothing. It does mean the failures are not MCAR, so the quarantine counts are reported per subset.
- **Prompt sharing comes from the turn-by-turn protocol (finding 3).** Crowdworkers chose between two replies at every turn of a conversation, so one conversation produces several pairs with the same opening message (median group: 3 pairs). With the official split, a model could recognise a test conversation it had already seen in training and look better than it is. The group split prevents this.
- **Label noise limits what to expect.** Researchers agreed with the crowdworkers' choice only about 63% of the time (Bai et al., 2022). Gains over the 55–61% per-subset bars (M2.7) will therefore be a few points rather than tens of points, which is why the ±3–5-point intervals matter when comparing the model with the baselines.

#### Figures

All 9 figures are produced by `make eda` and saved in [`docs/eda/`](docs/eda/). They use train only, except fig. 5 and the dev replication in fig. 8. The numbers are in the findings table above.

<p align="center">
  <img src="docs/eda/fig1_quarantine_by_subset.png" alt="Quarantine rate by subset" width="800"><br>
  <em><b>Figure 1.</b> Quarantine rate by subset (finding 2).</em>
</p>

<p align="center">
  <img src="docs/eda/fig2_groups_and_similarity.png" alt="Prompt groups and reply similarity" width="800"><br>
  <em><b>Figure 2.</b> Prompt-group sizes and reply similarity (finding 3).</em>
</p>

<p align="center">
  <img src="docs/eda/fig3_length_distributions.png" alt="Reply length distributions" width="800"><br>
  <em><b>Figure 3.</b> Reply-length distributions, raw and after <code>log1p</code> (finding 4).</em>
</p>

<p align="center">
  <img src="docs/eda/fig4_length_bias.png" alt="Length bias by subset" width="800"><br>
  <em><b>Figure 4.</b> Share of pairs where the longer reply is preferred, by subset (finding 6).</em>
</p>

<p align="center">
  <img src="docs/eda/fig5_length_baselines_dev.png" alt="Length baselines on dev" width="800"><br>
  <em><b>Figure 5.</b> Length baselines on dev, pooled and per subset (finding 7; per-subset bars in M2.7).</em>
</p>

<p align="center">
  <img src="docs/eda/fig6_refusals.png" alt="Refusal preference by subset" width="800"><br>
  <em><b>Figure 6.</b> How often the refusal is preferred when only one reply is a refusal, by subset (finding 9).</em>
</p>

<p align="center">
  <img src="docs/eda/fig7_feature_correlations.png" alt="Feature correlations" width="800"><br>
  <em><b>Figure 7.</b> Spearman correlations between inputs (finding 10).</em>
</p>

<p align="center">
  <img src="docs/eda/fig8_inference.png" alt="Group-bootstrap inference" width="800"><br>
  <em><b>Figure 8.</b> Group-bootstrap intervals and replication on dev (finding 11).</em>
</p>

<p align="center">
  <img src="docs/eda/fig9_cross_validation.png" alt="Random-row vs group cross-validation" width="800"><br>
  <em><b>Figure 9.</b> Random-row vs prompt-group 5-fold cross-validation (finding 12).</em>
</p>


### Six qualities of a trustworthy pipeline
 
| Quality | How it is built into this pipeline | Test that guards it |
|---|---|---|
| Repeatable | The data commit and library versions are pinned and the splits are hash-based. A full rerun reproduces the Parquet files and EDA figures byte for byte | `test_rerun_is_byte_identical` |
| Idempotent | Each output file is overwritten in full on every run (like `WRITE_TRUNCATE`), never appended to; `raw/` is write-once and rejects any file whose bytes differ from the original | `test_store_refuses_to_overwrite_raw_with_different_bytes` |
| Observable | Every step saves its counters in the manifest; rejected rows are quarantined with their reasons; each file's quarantine rate and field fingerprint are tracked, with alerts for unusual rates or schema drift (fields renamed, added, removed or changing type) | `test_schema_and_quarantine_rate_recorded_per_file`, `test_alerts_for_schema_drift_and_reject_spike` |
| Testable | 44 unit tests on synthetic data with deliberately planted defects; GitHub Actions runs them and an offline demo on every push | `tests/test_pipeline.py`, `.github/workflows/ci.yml` |
| Fails loudly | The run stops on a checksum mismatch, a row-accounting gap (`rows read ≠ kept + quarantined`), a HARD contract violation, a reconciliation disagreement or, optionally, an alert | `test_tampered_raw_file_is_rejected`, `test_processed_contract_catches_a_corrupted_copy`, `test_reconciliation_catches_a_flipped_label` |
| Cost-aware | One Python process instead of Dataflow workers (see M2.1); Colab and CI are free; lifecycle rules keep storage costs small | — |


**Apache Beam (optional, `make beam`).** `src/beam_validate.py` re-implements step 3's data contract as a Beam pipeline. It uses a `DoFn` to read the files, a `Validate` `DoFn` that sends failing rows to a tagged `bad` output, Beam `Metrics` counters, and a `CombinePerKey` to detect duplicates. Every counter is then compared with the pandas version of step 3, and on the real data all 10 counts match (169,352 read, 168,029 kept, 1,323 quarantined, broken down by reason). This cross-check proved its worth early: in the first version, Beam's built-in `.gz` reader returned only 44,044 rows, and the count mismatch revealed the problem immediately (`test_beam_pipeline_agrees_with_pandas_step3` now guards it). The code uses only portable Beam transforms, so it should run on Dataflow with `--runner DataflowRunner`, but this is planned, not tested.


### Data governance and ethics

This section summarises the governance rules that apply to v1.1, so they can be checked here without the M1 report. Full detail is in [`docs/DATASHEET.md`](docs/DATASHEET.md), and the original audit plan is in sections 11–12 of the [M1 README](https://github.com/bounchun/rlhf-reward-model-pipeline-m1).

| Topic | What the project does | Evidence |
|---|---|---|
| Licence and provenance | `Anthropic/hh-rlhf` is MIT-licensed. The download is pinned to commit `09be8c5`, all 8 raw files are checked by SHA-256, and `raw/` is write-once. Every version has a manifest recording file hashes, row counts and the code commit | `manifests/raw-09be8c5….json`, `manifests/v1.1.json`, `reconciliation.json` |
| Harmful content | harmless-base contains offensive and dangerous red-team text. No dataset text is ever committed to Git: the EDA outputs are aggregates only, the 200-pair review sheet stays local, and the Colab verify cell shows non-text columns only. Both buckets block public access | `.gitignore`, `docs/eda/`, bucket settings screenshot (M2.2) |
| Personal data | 304 e-mail addresses and 724 phone-like numbers are masked as `[EMAIL]` / `[PHONE]` in step 3, and step 4 fails the run if any remain. Names and street addresses are **not** detected (see M2.6) | `quality_report.json` (`no_residual_pii_after_masking`) |
| Access (least privilege) | `pipeline-sa` writes; `train-sa` reads `processed/` only and cannot reach `raw/`, `quarantine/` or the holdout bucket | `docs/iam_check_v1.1.txt` |
| Data poisoning | Step 4 flags 8-word phrases that appear in at least 20 pairs and are on the same side (chosen or rejected) at least 90% of the time, which is what a planted trigger would look like (Rando & Tramèr, 2024). Flagged pairs go to manual review rather than being dropped. v1.1: no pairs flagged | `audit/v1.1/poison_screen.csv` |
| Annotators and consent | Crowdworkers (mostly US-based, MTurk and Upwork) were paid (Bai et al., 2022); the release does not document ethical review or consent. This project collects no new human data; the only new labels are my own 200-pair blind review | `docs/DATASHEET.md` §3 |
| Representativeness and bias | English only, mostly US crowdworkers, 2021 models. The length and refusal effects differ by subset, so results are always reported per subset and never as a claim about how "people in general" prefer replies | Findings 6 and 9; `docs/DATASHEET.md` §5 |
| Retention and distribution | Processed files are not redistributed and stay in the private buckets; anyone can rebuild them from the public release with `make all`. Old object versions are deleted by the lifecycle rules (4 newer versions or 90 days) | `scripts/gcs_lifecycle.json` |

### Leakage audit (Kapoor & Narayanan, 2023)

| Leakage type | How this project handles it |
|---|---|
| L1.1 No test set | The test set is made of held-out prompt groups, stored in the holdout bucket, which `train-sa` cannot read (verified in [M2.2](#m22-storage-implementation-and-evidence)) |
| L1.2 Pre-processing on train and test | The outlier threshold, the scaler and the refusal phrases are all learned from train only, verified by tests |
| L1.3 Feature selection on train and test | No feature selection is done; the evaluation slices are fixed in `config.json` before the test set is used |
| L1.4 Duplicates | Exact duplicates are sent to quarantine; the data contract fails the run if a `pair_id` appears in two splits; no comparison appears twice |
| L2 Illegitimate features | The A/B swap removes the clue given by the `chosen` position; `subset`, `split` and the flags are used only to break down the evaluation, never as model inputs |
| L3.1 Temporal leakage | The dataset has no timestamps, so F1 and F2 are held-out prompt groups, not genuinely later data (see Limitations) |
| L3.2 Non-independence between train and test | Data is split by prompt group, so no prompt appears in both train and test (finding 3) |
| L3.3 Sampling bias in the test distribution | The test set is sampled the same way as train (same mix of subsets), so it reflects how this dataset was collected, not how real users would behave |

The dataset itself is documented in **[`docs/DATASHEET.md`](docs/DATASHEET.md)**, following Gebru et al. (2021).

### M2.4 Changes since v1.0, and why

The data split is unchanged (same salts), and so are the two buckets; `train-sa`'s access was narrowed to `processed/`, and how the pipeline uses the storage was refined (see [M2.2](#m22-storage-implementation-and-evidence)). v1.1 is a MINOR version (M1 versioning rule: cleaning rules changed, split unchanged), and the run confirms it: train / dev / test / F1 / F2 have exactly the same sizes, groups and label shares as v1.0.

| What changed | Reason |
|---|---|
| `train-sa` limited to the `processed/` managed folder, verified by `scripts/check_iam.sh` | Bucket-wide read access also let it read `raw/`, which contains the test rows |
| New `quarantine/` and `audit/` folders; the processed contract runs before upload; step 11, the EDA and the reconciliation read their input back from the buckets | Storage supports the whole pipeline, and later checks run on what was actually stored, not on a local copy |
| Quarantine instead of dropping rows (`quarantine/v1.1/quarantine.parquet`) | Every rejected row keeps its reason, file and line number |
| Processed data contract + `validation_log.csv` | Record the rule, its rationale and the number of violations; stop before upload on a HARD violation |
| PII masking in step 3 (new reason `identical_after_pii_mask`: 1 pair) | Governance: the data should not carry e-mails or phone numbers; ids are computed before masking so pair ids and splits do not move |
| Refusal phrases rebuilt | Planned in the M1 README and confirmed by the EDA | 
| Step 11 model-ready inputs in `model_inputs/` (both buckets) | Fit on train, transform every split; M3 reads ready-to-use inputs, and the test inputs stay in the holdout bucket |
| `reconcile.py`, `eda.py`, CI workflow | Evidence that the pipeline is correct and reproducible |


### M2.5 v1.1 run on the real data

| Check | Result |
|---|---|
| Raw input | All 8 files have SHA-256 checksums that match the ones pinned in `manifests/raw-09be8c5….json` |
| Row accounting (step 3) | All 169,352 rows read are accounted for: 168,029 kept and 1,323 quarantined (788 with identical replies, 329 with a context mismatch, 194 with an empty reply, 11 with malformed turns and 1 that became identical after PII masking) |
| Quality checks (step 4) | All 6 hard checks pass, no e-mail or phone patterns remain, and the poisoning screen flags no pairs |
| Splits (step 7) | Train 24,122, dev 2,876, test 3,004, F1 16,516 and F2 16,652 pairs –  identical to v1.0 |
| Data contract (step 9) | All 17 rules checked on each of the 5 splits, with no violations and no warnings |
| Model inputs (step 11) | The scaler is fitted on the 24,122 train rows; 9.7% of train contexts and 8.3% of dev contexts are truncated |
| Reconciliation | All 7 independent checks match the pipeline's numbers |
| EDA | 9 figures and 13 findings, saved in `docs/eda/` |
| Cross-validation | Length × subset model (ColumnTransformer pipeline): 56.9% with random-row folds vs 56.7% with group 5-fold CV, and 57.9% on dev. Bag of words: 54.9% with both fold types, and 54.8% on dev |
| Storage (Colab run, 26 Oct 2026) | Train, dev, their model inputs, the quarantine and the audit files are stored in `gs://bc-rlhf-reward-2026` (Standard); test, F1, F2 and their model inputs are stored only in `gs://bc-rlhf-reward-2026-holdout` (Coldline). Every stored file was read back and matched against the manifest, which is committed as `manifests/v1.1.json` with the tag `data-v1.1`. Screenshots are in [M2.2](#m22-storage-implementation-and-evidence) |


### M2.6 Limitations and reflection

**Limitations**

| Limitation | Impact | How to address it |
|---|---|---| 
| The Colab run used my personal Google account instead of `pipeline-sa` | The `train-sa` restrictions are verified (`docs/iam_check_v1.1.txt`), but the pipeline itself has not yet run as `pipeline-sa` | In M3, run the pipeline with `--impersonate-service-account` set to `pipeline-sa`, and run training as `train-sa` |
| Each pair has a single, noisy label (about 63% researcher agreement) | Caps the accuracy any model can reach; some apparent model errors will really be label errors | Use the 200-pair review sheet to estimate how noisy the labels are |
| hh-rlhf has no timestamps | F1 and F2 are held-out prompt groups rather than data collected later, so they cannot measure drift over time | Use a preference dataset that records when each pair was collected |
| The test set reflects how this data was collected (crowdworker prompts) | Results describe crowdworker conversations on these four subsets, not real users | Evaluate on a preference set collected in a different way |
| Test accuracy for each subset is only precise to about ±3–5 points | Small differences between subsets are not reliable enough to report | Report group-bootstrap confidence intervals and only claim differences larger than them |
| PII masking only catches fixed patterns (e-mail, phone numbers) | Names, addresses, other personal details written in free text stay unmasked | Add a PII detector based on named-entity recognition |
| Processing runs on a single machine; the Beam job was never run on Dataflow | Works for 79 MB of data, but larger volumes are untested |  Run `beam_validate.py` with `DataflowRunner` if the data grows |
| There is no `models/` folder yet | Trained models are not yet stored or versioned | In M3, add `models/` to the main bucket and version it the same way as the data |

**Reflection**

- **Implementing storage early paid off:** because the buckets worked after M1, M2 could focus on how the pipeline uses them (checking before upload, keeping rejected rows, reading files back to verify them). Most of the trust in the data now comes from these checks rather than from the bucket itself.
- **EDA changed the pipeline, not just the report:** three issues found in the data (the label that always pointed to `chosen`, the curly apostrophe that broke the refusal rule, and the length effect that varied by subset) each led to a code change and a new test.
- **Configured is not the same as verified:** testing the access rules by acting as `train-sa` showed that its bucket-wide read role also opened `raw/`, which holds the test rows. Restricting it to `processed/` and keeping `check_iam.sh` as a repeatable check closed the gap; the holdout bucket alone was not enough.
- **Independent checks caught a real bug:** the Beam cross-check exposed the gzip reading bug described under *Apache Beam* above, which the pandas pipeline alone would never have shown. Next time I would build the reconciliation first, not last.
- **Keeping claims precise:** reviewing the draft showed some wording that went further than the evidence ("no leakage", "Dataflow-ready"). These now read "no measurable inflation for these models" and "planned, not implemented".

### M2.7 Plan for Milestone 3

**First model.** A small pretrained transformer encoder (512 tokens) used as a reward model: it gives each reply its own score from the context plus that reply (`text_a`, `text_b` from step 11), and is trained with a pairwise loss so that the preferred reply scores higher. Scoring each reply on its own, rather than classifying A vs B, means the same model can later rank any number of replies. It must beat these baselines from M2 (dev accuracy):

| Baseline (all 4 subsets pooled) | Dev accuracy |
|---|---|
| Constant (train majority class) | 48.3% |
| Always pick the longer reply | 50.9% |
| Length only, one model for all subsets | 51.4% |
| Bag of words (reply A − reply B) | 54.8% |
| Length × subset, one model per subset | 58.2% |

The constant baseline is below 50% because it predicts train's majority class: `label` = 1 in 49.8% of train pairs, so it always predicts 0, while dev has `label` = 1 in 51.7% of pairs. The A/B swap is by hash, so the two shares differ by chance, not by design.

The pooled numbers hide large differences between subsets, so the M3 target is set **per subset**: on each subset, the model must beat the *best* length baseline for that subset (dev accuracy, `eda_summary.json` → `E8_baselines.by_subset_dev`):

| Subset (dev pairs) | Always pick longer | Length, one model for all | Length, one model per subset | Bar for M3 |
|---|---|---|---|---|
| helpful-base (723) | **61.0%** | 60.2% | 60.6% | 61.0% |
| helpful-rejection-sampled (966) | **55.6%** | 55.6% | 55.0% | 55.6% |
| helpful-online (356) | 44.1% | 48.3% | **57.0%** | 57.0% |
| harmless-base (831) | 39.7% | 40.2% | **60.3%** | 60.3% |

So 58.2% is not a single bar. On helpful-base and helpful-rejection-sampled, the per-subset model adds nothing over "pick the longer reply", and is even slightly worse. Its pooled gain comes almost entirely from harmless-base and helpful-online, where it learns that the *shorter* reply tends to win. With only 356–966 dev pairs per subset, each bar is itself only precise to about ±3–5 points.

**How it will be evaluated**

- **Data use:** train on `train`, choose settings on `dev` with `GroupKFold` on prompt groups, and score `test` only once, at the end. F1 is used as an extra "new prompts" check.
- **Per subset, never pooled alone:** length and refusal push in opposite directions in helpful-base and harmless-base (findings 6 and 9), so a pooled score can hide a failure on one subset.
- **Slices fixed in `config.json` before testing:** length gap between the replies, pairs where only one reply is a refusal, near-identical replies, and truncated contexts.
- **Group-bootstrap 95% intervals** (resampling prompt groups). The model counts as beating a baseline on a subset only if its interval lies above the baseline.
- **Label-noise ceiling:** the 200-pair review sheet from step 10 estimates how often the labels themselves are wrong, so accuracy is judged against that ceiling, not against 100%.

**MLOps**

| Need | Plan |
|---|---|
| Experiment tracking | Every run logs the data version (`data-v1.1`), the Git commit, the full config, the random seed and the metrics per subset and slice, with MLflow |
| Versioned models | A new `models/<model-version>/` folder in the main bucket holds the weights, a manifest (SHA-256, data version, commit, metrics) and a short model card, versioned like the data |
| Access | Training runs as `train-sa`, which can read `processed/` and nothing else (checked by `scripts/check_iam.sh`). The final test is a separate step, run as `pipeline-sa` |
| Retraining trigger | Retrain when a new data version is released (MINOR or MAJOR bump), when the monitoring alerts in step 3 fire, or when accuracy on new held-out prompts (F2) falls below that subset's bar in the table above |
| Reproducibility | Pinned libraries and seeds, tests for the training code in CI, and a Colab notebook that runs training end to end |

## ▶️ How to run it

**Option 1: quick check, no account needed (about 1 minute).** Open a new notebook at [colab.research.google.com](https://colab.research.google.com), paste this into one cell and run it:

```python
!git clone https://github.com/bounchun/rlhf-reward-model-pipeline-m2.git repo
%cd repo
!pip install -q -r requirements.txt
!make test
!DHAI_BUCKET= make demo && echo "demo OK"
```

It should end with `OK` (44 tests passed; the Apache Beam test runs when `requirements-beam.txt` is installed, as in CI) and `demo OK` (the whole pipeline, steps 1–11, the reconciliation and the EDA, run on fake data with planted problems). Nothing touches Google Cloud. The same two commands run on every push in GitHub Actions.

**Option 2: the full pipeline on the real data.** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bounchun/rlhf-reward-model-pipeline-m2/blob/main/notebooks/run_pipeline_colab.ipynb)
1. Click the badge, then File → Save a copy in Drive.
2. Add a Colab Secret (🔑 in the left sidebar): name `DHAI_BUCKET`, value = the name of a bucket in your own Google Cloud project, with notebook access switched on.
3. In the first code cell, set `GCP_PROJECT` to your project ID.
4. Run the cells from top to bottom. Run 0b once to create your buckets (skip it if they already exist from Milestone 1), and skip the cells marked *skip*.

The run takes about 5–10 minutes and gives the same numbers as in [M2.5](#m25-v11-run-on-the-real-data), because the split uses hashes, not random numbers. The EDA cells show the 9 figures inline, and the cell after them downloads `docs/eda/` and `manifests/v1.1.json` to commit. This is a new notebook for Milestone 2 only; the Milestone 1 notebook stays in the M1 repository.

To check the stored data without rerunning the pipeline, run only Settings, Install and Login, then the Verify the buckets and read a file back cell. Like Cell 8 of the GCS lab, it lists every file in both buckets and reads `dev.parquet` straight from the bucket with pandas. You should see the raw, quarantine, processed, audit and manifest files in the main bucket, only `test`, `future_f1`, `future_f2` and their `model_inputs` in the holdout bucket, and `Rows loaded: 2,876`. It shows only non-text columns, because the dataset contains harmful text.

**Option 3: on your own machine** (Python 3.13):
```bash
git clone https://github.com/bounchun/rlhf-reward-model-pipeline-m2 && cd rlhf-reward-model-pipeline-m2
pip install -r requirements.txt
make test && make demo                    # offline check
export DHAI_BUCKET=<your-bucket> && make all   # full run (steps 1-11, reconcile, eda), needs Google Cloud
DHAI_BUCKET= make all                          # or: full run with local folders instead of buckets
```

---

## 🛠️ Repository layout

What each step does is described once, in the tree in [M2.1](#m21-pipeline-architecture). Each script also says at the top what it reads, writes and checks, and each run records its counts, settings and checksums in `manifests/<version>.json`.

| Scripts (`src/`) | Role |
|---|---|
| `step1_scrape_raw.py` … `step11_model_inputs.py` | The 11 pipeline steps, run in order by `make all` |
| `contract.py` | The raw data contract (step 3) and the 17-rule processed contract (step 9) |
| `reconcile.py`, `beam_validate.py` | Independent checks: plain-Python recount (7 checks), and step 3 re-run on Apache Beam (`make beam`) |
| `eda.py`, `sample_queries.py` | EDA outputs in `docs/eda/`, and example queries on the stored train and dev data |
| `score_review.py` | Scores the 200-pair blind review against the dataset labels (`make review-score`) |
| `common.py`, `text_utils.py` | Storage layer, hashing, manifests; text normalisation, PII masking, refusal phrases |

| Path | Purpose |
|---|---|
| `config.json` | All pipeline settings in one file: Hugging Face revision, hashing salts, split buckets, thresholds, alert limits, EDA settings |
| `requirements.txt` | Pinned package versions for Python 3.13 (optional Apache Beam dependencies in `requirements-beam.txt`) |
| `Makefile` | `make all` \| `scrape` \| `store` \| `preprocess` \| `queries` \| `reconcile` \| `eda` \| `beam` \| `test` \| `demo` |
| `src/` | The step scripts above, plus `common.py` (storage, hashing, manifests) and `text_utils.py` (text processing) |
| `tests/` | 44 `unittest` tests on synthetic data with deliberately planted errors (generated by `tests/fake_data.py`) |
| `notebooks/run_pipeline_colab.ipynb` | Runs the full M2 pipeline in Colab: steps 1–11, reconciliation, EDA figures. Uses Google login and a Colab Secret, so no credentials are stored in code |
| `manifests/` | Dataset manifests recording lineage: file checksums, row counts and code commit that produced each version |
| `docs/eda/` | EDA outputs: summary statistics and figures only, no text from dataset |
| `docs/DATASHEET.md` | Datasheet for the processed v1.1 dataset: what it contains, how it was built and its known limitations (Gebru et al., 2021) |
| `docs/architecture_m2.html` | Source file for the architecture diagram (`docs/img/architecture_m2.png`) |
| `docs/img/m2_bucket*_v1.1.png` | Bucket screenshots after v1.1 Colab run |
| `scripts/setup_gcs.sh` | Run once to create GCS buckets, service accounts and IAM permissions (`train-sa` restricted to `processed/` in v1.1) |
| `scripts/check_iam.sh` | Acts as `train-sa` and checks what it can and cannot read; output in `docs/iam_check_v1.1.txt` |
| `.github/workflows/ci.yml` | GitHub Actions workflow: runs the tests and an offline demo on every push |

                            


Interim files (`data/interim/`) stay local and are never committed: until the split, it still contains test rows.

---

## 📚 References used in M2

- Bai, Y. et al. (2022). [*Training a Helpful and Harmless Assistant with RLHF*](https://arxiv.org/abs/2204.05862).
- Gebru, T. et al. (2021). [*Datasheets for Datasets*](https://arxiv.org/abs/1803.09010). Communications of the ACM, 64(12).
- Kapoor, S. and Narayanan, A. (2023). [*Leakage and the reproducibility crisis in ML-based science*](https://doi.org/10.1016/j.patter.2023.100804). Patterns, 4(9).
- Anscombe, F. J. (1973). Graphs in statistical analysis. *The American Statistician*, 27(1).
- Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.
- Schutt, R. and O'Neil, C. (2013). *Doing Data Science*, ch. 2. O'Reilly.
- scikit-learn: [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html).
- [Apache Beam programming guide](https://beam.apache.org/documentation/programming-guide/).
- Wahid, A. (2026). *Data Handling and Infrastructure for AI*, Lectures 4–6 and Labs 4–6, SETU.
- Singhal, P., Goyal, T., Xu, J. and Durrett, G. (2023). [*A Long Way to Go: Investigating Length Correlations in RLHF*](https://arxiv.org/abs/2310.03716). COLM 2024.
- Rando, J. and Tramèr, F. (2024). [*Universal Jailbreak Backdoors from Poisoned Human Feedback*](https://arxiv.org/abs/2311.14455). ICLR.

---

## 🤖 Use of AI

**Tool used**

- Claude (Anthropic) and ChatGPT (OpenAI) 

**How I worked with it**

| Part of the work | My role | How AI helped |
| --- | --- | --- |
| Debugging | Ran the pipeline, fixed the problems in my Colab run and reran it until every check passed | Helped find the causes (for example the Beam `.gz` reading bug) and suggested fixes |
| Google Cloud | Set up and ran the v1.1 pipeline on my own GCP project and buckets on 26 October 2026, and took the screenshots | Step-by-step guidance |
| Report, README and datasheet | Checked every number against my Colab run and `manifests/v1.1.json` | Suggested shorter wording, fixed Markdown tables and wrote figure captions |
| Language | — | Grammar and spelling |

All numbers in this README come from my own pipeline run on 26 October 2026, recorded in `manifests/v1.1.json` and `docs/eda/eda_summary.json`, not from the AI.

**Example prompts**

1. *"Fix the formatting problem in markdown."* (followed by a broken table)
2. *"Write the figures block for M2.3, with all nine figures and a one-line caption for each from `findings.csv`."*
3. *"Give me my real project ID."*

---

## 💻 Author

**Boun Chun** · [@bounchun](https://github.com/bounchun)
