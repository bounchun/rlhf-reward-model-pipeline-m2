# Response Preference Prediction (RLHF Reward Modeling) — Milestone 1

## Milestone 1 questions at a glance

The sections below explain each answer in more detail.

| Slide question | Answer |
|---|---|
| Select a suitable **raw data source** (public, licence checked) | I use `Anthropic/hh-rlhf` from Hugging Face, pinned to one fixed commit. It's public and under the MIT licence, so I'm allowed to use it. |
| **At least 10,000 learning samples**? | Yes, easily. The source has 169,352 preference pairs. I work with a 30,002-pair subset (train/dev/test) and keep another 33,168 pairs aside as "future" data for later milestones. |
| **Realistic imperfections, missing values / other quality issues** | Yes, plenty. In my run I found 788 pairs where both replies are identical, 329 where the two conversations don't match, 194 empty replies and 11 broken conversations. There are also very long replies, possible personal data (emails and phone numbers) and noisy labels: the original paper reports only about 63% agreement between annotators. I also screen for duplicates and suspicious repeated phrases (possible poisoning). Every check is counted in the manifest. |
| Define a **meaningful AI/ML task** | Given a conversation and two possible replies, predict which one a human preferred. This is how a reward model is trained in RLHF. |
| Identify the **target variable** and **relevant features** | The target is `label` (1 if reply A was preferred, 0 if B). The model inputs are `context`, `response_a`, `response_b`, `num_turns` and the reply lengths. `subset`, the refusal flags and `is_long_outlier` are only used to check results, not for training. |
| **Evaluation strategy**: train/dev/test or cross-validation, justified | A fixed 80/10/10 train/dev/test split rather than cross-validation. I have enough data, and training a transformer k times would cost too much. I split by conversation, so the test set only has conversations the model has never seen, which is how it will be used in practice. |
| Where will the **raw data** live? | In my Google Cloud bucket: `gs://bc-rlhf-reward-2026/raw/hh-rlhf/09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa/`. The files are the original JSONL.gz files, never edited, and the bucket keeps old versions. |
| Where will the **processed data** be stored? | Train and dev go in `gs://bc-rlhf-reward-2026/processed/v1.0/`. Test and future data go in a separate bucket, `bc-rlhf-reward-2026-holdout`, so training can't read them by mistake. |
| What **file formats**? | Raw data stays as JSONL.gz, as downloaded. Processed data is Parquet. Manifests are JSON, and audit reports are CSV or JSON. |
| **Database or object storage**? | Object storage only (Google Cloud Storage). Training just reads whole files from start to end. There are no joins, transactions or searches, so a database would add nothing. |
| How are **data versions identified**? | Each version has its own folder (`v1.0`, then `v1.1` and so on). A manifest file records the checksums of every file and the Git commit of the code. I also tag the version in Git (`data-v1.0`), and the buckets keep old copies of overwritten files. |
| How will the system **access the data**? | Through `gcsfs`, using my Google login, with no key files in the code. The bucket name comes from a Colab Secret. Each service account only has the access it needs, and the training account can't read the holdout bucket at all. |

---

## 0. Data source and task

| | |
|---|---|
| **Source** | [`Anthropic/hh-rlhf`](https://huggingface.co/datasets/Anthropic/hh-rlhf) on Hugging Face. I pinned it to commit `09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa` so the data can't change under me. |
| **Licence** | MIT, so it's public and can be reused. It isn't meant to contain personal data, but I still run a simple PII scan for emails and phone numbers (step 4). |
| **Size** | 169,352 preference pairs (160,800 from the official train split and 8,552 from test), across four subsets: `helpful-base`, `harmless-base`, `helpful-online` and `helpful-rejection-sampled`. I left out `red-team-attempts` because it has a different format and no pairs to compare. |
| **Where it comes from** | It was released with the Bai et al. (2022) paper. Most replies were written by Anthropic's 52B language models, and the preferences were given by crowdworkers, about 80% of them US-based MTurk workers and the rest hired through Upwork. It was collected in three rounds: base first, then rejection-sampled, then online (added weekly over about 5 weeks). Each pair has only one label, and the paper reports that researchers and crowdworkers agreed only about 63% of the time, so the labels are noisy. |
| **What a record looks like** | Two text fields, `chosen` and `rejected`. Each holds the whole conversation (`\n\nHuman: … \n\nAssistant: …`), and the two are identical except for the last Assistant reply. |
| **Task** | Given a conversation (`context`) and two possible replies, A and B, predict which one the person preferred. This is a binary classification problem, and it's the reward-model step of RLHF. |
| **Target** | `label` = 1 if A was preferred, 0 if B was preferred. |
| **Metrics** | Mainly pairwise accuracy, with a bootstrap 95% confidence interval. I'll also report ROC-AUC, F1 and accuracy for each subset. |


**Why I picked this dataset.** It isn't a clean benchmark. The problems just don't show up as empty cells in a table: they're hidden inside the conversation text, so I had to parse the text to find them. In the v1.0 run I found:

- 788 pairs where the "chosen" and "rejected" replies are exactly the same, so there's nothing to learn from them
- 329 pairs where the two versions of the conversation don't match before the final reply
- 194 empty replies and 11 conversations with broken turn structure
- lots of repetition: 168k pairs come from only about 62k opening prompts, which is a leakage risk if you split by row
- some very long replies (the top 1% are over 210 words)
- four subsets collected in different ways, with different length habits (section 10b)
- a position leak: the preferred reply is always in the `chosen` column, so a model could learn the column instead of the preference

I also checked for exact duplicates, but there were none. Every one of these checks is counted in the manifest (section 10).

The pipeline counts and logs every one of these (section 10).

> ⚠️ `harmless-base` deliberately contains offensive and harmful prompts. The data is not redistributed in this repository.

---

## Repository layout and pipeline steps

I followed the module's Milestone 1 checklist and wrote one Python script for each step. Each script says at the top what it reads and what it writes, and `make all` runs them in order. As each step runs, it writes down what it did (how many rows it kept or dropped, the settings it used, and file checksums) in `manifests/v1.0.json`, so the whole run can be checked afterwards.

| Module checklist | Script (`src/`) | Input → output |
|---|---|---|
| Script to scrape the raw data | `step1_scrape_raw.py` | Hugging Face (pinned commit) → `data/downloads/`, `manifests/raw-⟨rev⟩.json` |
| Script to move the raw data into storage | `step2_store_raw.py` | `data/downloads/` → `gs://bc-rlhf-reward-2026/raw/` (write-once, read-back verified) |
| Data validation / cleaning | `step3_clean.py` | `raw/` (checksums re-verified) → `data/interim/v1.0/cleaned.parquet` |
| Data validation / quality checks | `step4_quality_checks.py` | cleaned → `gs://bc-rlhf-reward-2026/audit/v1.0/` (poisoning screen, PII, length summary) |
| Code to extract features | `step5_features.py` | cleaned → `data/interim/v1.0/features.parquet` |
| Code to filter the data if it's too large | `step6_filter.py` | features → future reserve (F1/F2) + ≈30k working set |
| Code to do the splits (train/dev/test) | `step7_split.py` | working set → `data/interim/v1.0/splits/*.parquet` (+ train-only outlier threshold) |
| Code for sharding (if needed) | `step8_shard.py` | train → batch-aligned shards; **off by default** (section 2) |
| Code to store the preprocessed data | `step9_store_processed.py` | splits → `processed/` and the holdout bucket, final validation, manifest |
| Extra quality check: label review | `step10_review_sample.py`, `score_review.py` | train → blind 200-pair sheet (created in v1.0); scoring planned for M2 |
| Extra: sample queries | `sample_queries.py` | stored train/dev → 7 example queries printed + `audit/v1.0/sample_queries.json` |

**Extra steps**

| Extra step | Where the code is |
|---|---|
| **Down-sampling** | `step6_filter.py`. After setting aside the future data, 134,862 pairs were left, which is more than I need, so I take a 30,002-pair working set. I take whole conversation groups, not single rows, in a fixed hashed order, and I keep the same mix of the four subsets as in the full data. The manifest records how many pairs came from each subset and how many were left unused. |
| **Up-sampling** | **Not used, on purpose.** Swapping the A/B positions in `step5_features.py` already gives a roughly 50/50 label balance, so neither class needs extra copies. I also don't add duplicated or generated rows: duplicates could end up in both train and test, and generated data can carry poisoning (sections 10 and 12). |
| **Sample queries** | `sample_queries.py`. A few pandas queries on the stored train/dev files: split sizes and label balance, subset mix, conversation length, length bias, refusals, outliers, and looking up one pair by its `pair_id`. They check that the stored data looks right and give a first look at it for M2. They never touch the holdout bucket. |
| **Data validation** | `step3_clean.py` re-checks the raw file checksums and applies 5 cleaning rules, counting what each one drops. `step4_quality_checks.py` stops the run if any basic check fails. `step7_split.py` checks that no conversation group appears in two splits, and `step9_store_processed.py` checks the final columns and overlaps again before uploading. |
| **Quality checks** | `step4_quality_checks.py` looks for possible poisoning, personal data (emails, phone numbers) and length problems. `step10_review_sample.py` and `score_review.py` are for checking the labels by hand: the sheet of 200 pairs is created in v1.0, and I'll label and score it in M2. |
| **Tests** | `tests/` has 22 unittest tests that run on fake data with problems planted on purpose, to check that the pipeline catches each one (`make test`). |


The in-between files that each step creates (`data/interim/`) stay on my machine and are never committed to Git. Before the split happens they still include the test rows, so I don't upload them to a bucket that the training account can read.

**Files in the repository**

```
config.json                  every parameter: HF revision, salts, split buckets, thresholds, regexes, folders
requirements.txt             exact pinned versions — only libraries from the module's labs (+ pyarrow)
Makefile                     make all | scrape | store | preprocess | queries | review-score | readme | test | demo
src/common.py                storage (GCS via gcsfs, or local), hashing, manifest helpers
src/text_utils.py            parsing helpers for hh-rlhf conversation strings
src/step1_ … step10_*.py     the pipeline steps above (each file starts with its input/output docs)
src/sample_queries.py        example queries on the stored data
scripts/setup_gcs.sh         creates both buckets, versioning, lifecycle, service accounts
scripts/fill_readme.py       fills this README's placeholders from the manifest after a real run
docs/img/, docs/diagrams/    design diagrams (PNG + Mermaid source) and screenshots of the real run
notebooks/run_pipeline_colab.ipynb   Colab runner (auth + Secrets, no hard-coded credentials)
tests/                       22 unittest tests on synthetic data with planted defects
manifests/                   committed dataset manifests (lineage)
```

**Workflow: how the scripts connect**

<p align="center"><img src="docs/img/workflow.png" alt="Pipeline workflow: step 1 to step 10" width="480"></p>

## Screenshots of the real run

These are taken from my own Google Cloud project and Colab run. They show what's in them and how they're set up.

| What it shows | Screenshot |
|---|---|
| Main bucket: `raw/`, `processed/`, `audit/`, `manifests/` folders | <img src="docs/img/screenshot_bucket_main.png" alt="Main bucket folders" width="420"> |
| Holdout bucket: `v1.0/test.parquet`, `future_f1.parquet`, `future_f2.parquet` | <img src="docs/img/screenshot_bucket_holdout.png" alt="Holdout bucket files" width="420"> |
| Bucket settings: location, storage class, Object Versioning, public access prevention | <img src="docs/img/screenshot_bucket_settings.png" alt="Bucket configuration" width="420"> |
| Colab Secret `DHAI_BUCKET` (value hidden), so no bucket name or credential is in the code | <img src="docs/img/screenshot_colab_secret.png" alt="Colab secret" width="420"> |
| Colab output of step 3 (rows dropped) and step 7 (split sizes) | <img src="docs/img/screenshot_colab_steps.png" alt="Colab step output" width="420"> |


## 1. Raw data storage 


I keep the raw data in a Google Cloud Storage bucket, `gs://bc-rlhf-reward-2026`. I set it up in `europe-west1` with the Standard storage class, uniform access control and public access blocked. **Object Versioning is on**, so if a file is ever overwritten or deleted, the old copy can still be recovered.

The files are stored here:

```
gs://bc-rlhf-reward-2026/raw/hh-rlhf/09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa/⟨subset⟩/{train,test}.jsonl.gz
```

They are exact copies of the files from Hugging Face, and I never edit them. Only one script, `src/step2_store_raw.py`, is allowed to write into `raw/`.



**Why object storage?** The raw data is only a few files. They never change, and the pipeline always reads each one from start to finish, which is exactly what a bucket is good at. Keeping the original files untouched also helps if I get a cleaning rule wrong: I can fix it and run the pipeline again without downloading anything.

I created both buckets with `scripts/setup_gcs.sh`, which also sets their storage class, versioning, clean-up rules and permissions. Screenshots of the real buckets are in [Screenshots of the real run](#screenshots-of-the-real-run).


## 2. Processed data storage and file formats 

| Stage | Location | Format | Why this format |
|---|---|---|---|
| Raw | `gs://bc-rlhf-reward-2026/raw/…` | JSONL, gzip | The original files, kept exactly as downloaded |
| Train / dev | `gs://bc-rlhf-reward-2026/processed/v1.0/{train,dev}.parquet` | Parquet (Snappy compression) | Stores columns with their types, is small on disk and loads quickly with pandas. Training reads whole splits in batches. |
| **Test + future data** | `gs://bc-rlhf-reward-2026-holdout/v1.0/{test, future_f1, future_f2}.parquet` | Parquet | Kept in a **separate bucket** (section 3) |
| Manifest | `gs://bc-rlhf-reward-2026/manifests/v1.0.json`, also committed to Git | JSON | A record of the version that both people and code can read |
| Models (from M3) | `gs://bc-rlhf-reward-2026/models/⟨model_version⟩/` | Checkpoint + training config | Kept apart from the data |

Each row is one preference pair (the columns are listed in section 7 & 8). The processed data is well under 1 GB, so one Parquet file per split is enough. Cutting it into shards would only create lots of small files, which Lecture 2 warns against, so sharding is **off for now**. `step8_shard.py` is ready in case I need to load data with several workers in M3: setting `shards.num_shards` above 1 writes `processed/v1.0/train_shards/train-0000i-of-0000N.parquet`. It works like the "Data shards" lab: rows are shuffled in a fixed order and every shard holds whole batches of 64 (`shards.batch_size`). The last shard takes whatever is left, so no rows are lost.

The in-between files stay in `data/interim/v1.0/` on my machine (not in Git) and can be rebuilt with `make preprocess`.

**Data organisation: where every file lives**

<p align="center"><img src="docs/img/data_organisation.png" alt="Data organisation across the two buckets, the local working folder and GitHub" width="700"></p>

## 3. Database / object storage decision 

I only use **object storage (Google Cloud Storage), no database**. A database wouldn't help here:

- Training reads whole splits from start to finish. There are no transactions, joins or single-row lookups, which is what SQL databases are for.
- The columns are fixed and simple, so a NoSQL store adds nothing.
- The task compares two replies that are already given. Nothing has to be searched for, so a vector database isn't needed either.
- In M4 the API will receive the conversation and both replies with each request, so it only needs the trained model.

**Two buckets, with different access and storage classes** (Lecture 3, *storage tiers*):

| Bucket | Contents | Storage class | Who can access it |
|---|---|---|---|
| `bc-rlhf-reward-2026` | raw, train, dev, manifests, models | Standard (used in every experiment) | me, the pipeline and the training account |
| `bc-rlhf-reward-2026-holdout` | test, future F1/F2 | Coldline (read rarely: once per final model, or in M4) | me and the pipeline only. The training account has **no** access. |

Keeping the test set in its own bucket means the training code can't read it, even by mistake, because it doesn't have permission, not just because I promised not to. Storing this little data costs almost nothing. If M4 needs to log predictions, I'll look at adding a small database then.

## 4. Data versioning 

I use three things together, following Lecture 2 (slide 32): code, config and metadata go in Git, and the data goes in object storage.

1. **A version number in every path.** This dataset is `v1.0` (`vMAJOR.MINOR`).
   - MAJOR goes up when the raw data or the way I split it changes.
   - MINOR goes up when the cleaning rules change.
   - I never overwrite a released version. Any change gets a new folder.
2. **A manifest, `manifests/v1.0.json`.** It records:
   - the Hugging Face commit of the raw data
   - the Git commit of my code
   - when it ran (UTC)
   - the hash salts
   - the number of rows per split and per subset, and the label balance
   - the SHA-256 checksum of every raw and processed file
   - how many rows each cleaning rule dropped

   The manifest is committed to Git, and that commit is tagged `data-v1.0` (see **Releases**). So any model can be traced back: **model → dataset version → code commit → raw data commit** (Lecture 3).
3. **Object Versioning** on both buckets, as a safety net if a file is overwritten by mistake. A clean-up rule keeps at most **3 older versions** of each file and deletes older versions after **90 days** (`scripts/gcs_lifecycle.json`).

## 5. Data access 

![Setup and access: inputs, service accounts and buckets](docs/img/setup_access.png)

| Part of the system | How it reads or writes data | Account and permissions |
|---|---|---|
| `src/step1_scrape_raw.py`, `src/step2_store_raw.py` | Downloads with `requests.get("https://huggingface.co/datasets/Anthropic/hh-rlhf/resolve/⟨pinned revision⟩/⟨subset⟩/⟨split⟩.jsonl.gz")`, then uploads with `gcsfs` | `pipeline-sa`: can read and write both buckets (Storage Object Admin) |
| `src/step3_clean.py` … `src/step9_store_processed.py` | Read `raw/`; write `audit/`, `processed/`, the holdout bucket and `manifests/` | `pipeline-sa` |
| Training (M3) | `pd.read_parquet(f"gs://{bucket}/processed/{version}/train.parquet")` through `gcsfs`. The version comes from `config.json` (`dataset.version: v1.0`). | `train-sa`: **can only read the main bucket** (Object Viewer) |
| Colab / my own machine | The same code | My Google login (`google.colab.auth` or `gcloud auth application-default login`) |

- **No passwords, keys or bucket names in the code.** The bucket name comes from a Colab Secret or the `DHAI_BUCKET` environment variable.
- Key files and the `data/` folder are listed in `.gitignore`, so they can't be committed by accident.
- **Limiting who can write also protects against poisoning.** Only `pipeline-sa` can write data, and every file is checked against the checksums in the manifest. If someone changed a file, from outside or from inside the project, the checksums wouldn't match (section 12).
- **Reviewers:** the buckets are private, so please see the [screenshots](#screenshots-of-the-real-run)

## 6. Data split and validation strategy 

I use a fixed **80/10/10 train/dev/test split on a 30,002-pair working set**, and keep a separate **"future" reserve** for M4. Lecture 2 suggests keeping about 10–15% each for dev and test when data is not huge (the slides show 70/15/15 as an example). I chose 10% each, because with a 30,002-pair working set that still gives about 3,000 pairs per split, which is enough for a precise estimate, and it leaves more data for training. I don't use cross-validation.

<p align="center"><img src="docs/img/splits.png" alt="How the data is split: future reserve, working set, train/dev/test" width="460"></p>


**Why I split in this way.** In M4 the model will sit behind an API and score replies for **conversations it has never seen**. I want the evaluation to copy that situation:
- **Split by conversation, not by row.** Dev and test only contain conversations that aren't in train. With a row split, near-copies of the same conversation would end up in both train and test, and the test score would look better than it really is.
- **A future reserve.** F1 and F2 stand in for new data that arrives after deployment, for retraining the model in M4.
- **Balanced labels and a metric that matches the job.** Swapping A and B stops the model from scoring well just by always guessing "A". Pairwise accuracy is exactly what a reward model is judged on in RLHF: how often it ranks the human-preferred reply higher.

**Why I didn't use cross-validation**
- With about 3,000 test pairs, the 95% confidence interval on accuracy is roughly ±1.8 percentage points, which is precise enough.
- k-fold cross-validation would mean training a transformer k times for very little gain.
- Lecture 2 recommends cross-validation when data is scarce, and here it isn't.

**Why I didn't keep the official Hugging Face split**
- It isn't grouped by conversation, so similar conversations can be on both sides.
- It has no dev set and no future reserve.

So I pool both official splits after cleaning and split them again as described below.

**Avoiding leakage 1: grouping.**
- Each pair gets a `group_id`: a SHA-256 hash of the first Human message, lower-cased and with spaces tidied.
- Pairs that start with the same message share a `group_id` and always go to the same split. I always split whole groups, never single rows.
- **Very big groups are split up again.** Generic openers like "hi" could create huge groups whose conversations have nothing else in common. Any group bigger than 50 pairs (`max_group_size`) is regrouped using the whole conversation instead, so identical conversations still stay together but one "hi" group can't unbalance the splits. In the v1.0 run the biggest group had 48 pairs, so this wasn't needed, but it's there for future data. The manifest records how many pairs were regrouped (`group_size`).

**How the split is done.** I use hashes instead of a random number generator, so the result is the same whatever the row order or library version.

1. **Set aside the future data.** `b1 = int(sha256(group_id + "future-v1"), 16) % 100`
   - `b1` 0–9 → **F1**, `b1` 10–19 → **F2**. That's about 20% of the groups, and I don't touch them until M4, where they play the part of newly arriving data ("keep some data unseen", Lecture 2).
   - `b1` 20–99 → the development pool.
2. **Take the working set of about 30,000 pairs** from the development pool.
   - Sort the groups by `sha256(group_id + "sample-v1")` and take whole groups in that order.
   - Do this **separately for each subset**, so the working set keeps the same subset mix as the full data.
   - 30k keeps training within the course's $50 compute budget.
3. **Split the working set.** `b2 = int(sha256(group_id + "tdt-v1"), 16) % 10`
   - 0–7 → train (about 24,000)
   - 8 → dev (about 3,000)
   - 9 → test (about 3,000)
4. **Check it.** The script stops if any `group_id` appears in more than one of train, dev, test, F1 and F2. It saves each split's subset mix and label balance in the manifest:

| Split | Pairs | helpful-base | harmless-base | helpful-online | helpful-rej.-sampled | label = 1 |
|---|---|---|---|---|---|---|
| train | 24,122 | 27.5% | 26.7% | 13.8% | 32.1% | 49.8% |
| dev | 2,876 | 25.1% | 28.9% | 12.4% | 33.6% | 51.7% |
| test | 3,004 | 27.6% | 25.0% | 14.4% | 32.9% | 49.1% |

**Avoiding leakage 2: position.**
- In the raw data the preferred reply is *always* in `chosen`, so a model could learn the column instead of the preference.
- For each pair I compute `int(sha256(pair_id), 16) % 2`. If it's 0, A is the chosen reply, B the rejected one, and `label` = 1.
- Otherwise A and B are swapped and `label` = 0.
- This gives roughly 50/50 labels.

**Avoiding leakage 3: learn from train only.**
- Anything learned from the data is learned **on train only** and then applied to dev and test ("split → fit on train → transform", Lecture 2):
  - the length-outlier threshold
  - the TF-IDF vocabulary (M3)
  - any normalisation statistics
- I use **dev** to tune settings, stop training early and choose between models.
- I read **test** only once per final model, and never use it to make a decision.

**Evaluation slices, fixed in `config.json` before I look at the test set.** Checking accuracy for each subgroup is one of the most common things auditors do (Costanza-Chock et al., 2022), so every model will be reported overall and on these slices:
- subset / collection round (base, rejection-sampled, online)
- length difference (`len_diff`): A much shorter than B, about the same, A much longer
- number of turns (`num_turns`): 1, 2–3, 4 or more
- whether either reply is a refusal (`refusal_a` or `refusal_b`)

If a model does well overall but badly on one slice, I flag it and don't ship it.

**Limitation (time).** hh-rlhf has no timestamp per row, so F1/F2 come from the same distribution as the rest; they aren't truly "later" data. The order of the three collection rounds *is* known, though, so in M4 I'll also run a drift check: train on the base data only and test on the online data.


## 7 & 8. Features, data types and formats — data card 

Each row of the processed files is one preference pair, with these columns:

| Column | What it is and how I get it | Arrow/Parquet type | Kind |
|---|---|---|---|
| `pair_id` | SHA-256 hash of context + chosen + rejected | string | identifier |
| `group_id` | SHA-256 hash of the tidied first Human message (section 6) | string | identifier (for splitting only) |
| `subset` | Which of the four subsets the pair comes from | dictionary string | category with 4 values; used to balance the splits and to check results, **not a model input** |
| `context` | The conversation before the final Assistant reply: the part `chosen` and `rejected` have in common, cut at the last `\n\nAssistant:`| string (UTF-8, NFC) | unstructured text |
| `response_a`, `response_b` | The two final replies, in the order set in section 6 | string | unstructured text |
| `num_turns` | Number of `\n\nHuman:` markers in the context | int16 | number (whole)|
| `len_a_words`, `len_b_words` | Number of words in each reply (`len(text.split())`) | int32 | number |
| `len_diff` | `len_a_words − len_b_words`, used for the length-bias analysis in M2 | int32 | number (can be negative) |
| `is_long_outlier` | True if either reply is longer than the 99th percentile **of train** | bool | yes/no flag |
| `refusal_a`, `refusal_b` | True if the reply matches one of the refusal phrases in `config.json` (e.g. "I'm sorry, but I can't") | bool | yes/no flag, **only for checking results, not a model input** |
| `label` | 1 if A is preferred, 0 if B is preferred | int8 | **target (binary)** |


**How the two models will use these columns:**
- **M3 baseline:** TF-IDF on `context`, `response_a` and `response_b` (single words and pairs of words, `min_df=2`, at most 50k features, learned on train), plus `len_diff` and `num_turns`, fed into a logistic regression.
- **Transformer:** DistilBERT on (`context`, reply) pairs, `max_length=512`. Long contexts are cut from the **start**, so the most recent turns are kept, and I log how often this happens.

In Lecture 1 terms, the raw data is **semi-structured** (JSON lines with one whole conversation per field). The processed data is **structured** (Parquet with typed columns), but its main content is still **unstructured text**.


## 9. Reproducibility of data collection 

Collecting the data takes two scripts (`make scrape store`).

**Step 1, `src/step1_scrape_raw.py`** (download)
1. Reads `hf_revision` from `config.json`. It has to be an exact commit hash; `main` is refused because it can change. `python -m src.step1_scrape_raw --resolve-revision` prints the current hash if I want to pin a newer one.
2. Downloads only the 8 files I need (4 subsets × train/test) with `requests`, from `https://huggingface.co/datasets/Anthropic/hh-rlhf/resolve/⟨hf_revision⟩/⟨subset⟩/⟨split⟩.jsonl.gz`, into `data/downloads/⟨revision⟩/`.
3. Saves the SHA-256 checksum of each file in `manifests/raw-⟨revision⟩.json`. If I run it again and any checksum is different, it **stops**.

**Step 2, `src/step2_store_raw.py`** (move into storage)
1. Checks each downloaded file against the raw manifest.
2. Uploads it to `gs://bc-rlhf-reward-2026/raw/hh-rlhf/⟨revision⟩/…`, then reads it back to make sure it arrived intact.
3. Refuses to replace a file in `raw/` with different content, because `raw/` is write-once.

**The environment is pinned too:**
- Every library has an exact version (`==`) in `requirements.txt`, matching the Colab runtime, and nothing is upgraded during a run (no `pip install --upgrade`).
- I only use libraries from the module's labs: `pandas`, `numpy`, `gcsfs` and `requests`, plus `fsspec`, which `gcsfs` is built on. The one extra is `pyarrow`, which pandas needs to read and write Parquet (the GCS tutorial recommends it too). Everything else (the JSON config, hashing, gzip and the `unittest` tests) uses the Python standard library.
- Python 3.13, the version Colab used for the v1.0 run.


## 10. Reproducibility of preprocessing 

`make preprocess` runs steps 3 to 10 in order. Each script explains at the top what it reads, what it writes and what rules it applies, and each one adds its counts and settings to `manifests/v1.0.json` (under `steps.⟨script⟩`).

| Script | What it does | Rows dropped |
|---|---|---|
| `step3_clean.py` | Five rules. **1** Drop lines that aren't valid JSON, and conversations that are missing or broken: each must start with a Human turn and end with an Assistant turn (regex `\n\n(Human\|Assistant):`). **2** Drop pairs where `chosen` and `rejected` differ before the final reply. **3** Tidy the text (Unicode NFC, extra spaces removed) but keep case and punctuation, because they matter. **4** Drop empty replies and pairs with identical replies. **5** Drop exact duplicates (same `pair_id`). | bad JSON 0 · broken/missing 11 · context mismatch 329 · empty 194 · identical 788 · duplicates 0 |
| `step4_quality_checks.py` | Stops the run if any basic check fails: `pair_id` unique, no empty or identical replies, only known subsets, no missing values. Also looks for possible **poisoning** (8-word phrases that appear in at least 20 pairs and almost always on the same side, section 12), counts **emails and phone numbers**, and summarises reply lengths per subset. Results go to `audit/v1.0/`. | 0 (it only flags) |
| `step5_features.py` | Adds `group_id` (regrouping very big groups), swaps A/B and sets `label`, and adds `num_turns`, the length columns and the refusal flags. | — |
| `step6_filter.py` | Sets aside F1/F2 for M4, then takes the ~30k working set from the rest, in whole groups and keeping the subset mix. | — (unused pairs are counted) |
| `step7_split.py` | Splits into train/dev/test and checks that no group is in two splits. Works out the 99th-percentile reply length **on train only** and marks long outliers in every split. Outliers are marked, not removed, because I want to study length bias in M2. | — |
| `step8_shard.py` | Optional training shards, each holding whole batches (section 2). | — |
| `step9_store_processed.py` | Checks the final columns and overlaps once more, writes the Parquet files to both buckets, saves their checksums and uploads the manifest. | — |
| `step10_review_sample.py` | Exports 200 train pairs, balanced across subsets, to `audit/label_review_v1.0.csv` without showing the labels. I will label them blind in M2, and `make review-score` will then record how often I agree with the crowdworkers. | — |

**No generated data.** I don't add synthetic or LLM-written rows to balance the data. The labels are balanced by the A/B swap, and the subsets by sampling. Training on generated data can lead to the feedback loop described in the week 1 lecture and in the data bias article (section 11). The Virus Infection Attack paper (Liang et al., 2025) also shows that poison can pass into generated data even when the prompts used to create it were clean.

All settings live in `config.json`, not in the code: the salts, the split buckets, the percentile, the working-set size, the regexes and the evaluation slices.

**To reproduce from scratch:**
```bash
git clone https://github.com/bounchun/rlhf-reward-model-pipeline && cd rlhf-reward-model-pipeline
pip install -r requirements.txt
export DHAI_BUCKET=⟨your-bucket⟩        # or set it as a Colab Secret
make all                                 # steps 1-10 + sample queries → processed/v1.0/*, manifests/v1.0.json
```

**To check it without Google Cloud or Hugging Face** (useful for reviewers):
```bash
make test    # 22 unittest tests on fake hh-rlhf-style data with planted problems
make demo    # full offline run of steps 1-10 → everything under data/demo/
```
`tests/fake_data.py` creates fake hh-rlhf data with a known number of problems planted in it: broken rows, missing values, mismatched conversations, empty and identical replies, duplicates, emails and phone numbers, and a one-sided trigger phrase. The tests check that each problem is dropped or flagged exactly as many times as it was planted. They also check that no group ends up in two splits, that the output columns are right, that the outlier threshold only uses train, that running twice gives byte-identical files, that a changed raw file is rejected, that sharding loses no rows, and that the review scoring works.

With the same `config.json` and the same commit, a rerun produces byte-identical Parquet files, which the checksums in the manifest confirm.


## Results of the v1.0 run (from `manifests/v1.0.json`)

I ran the pipeline on 1 October 2026 on `Anthropic/hh-rlhf` commit `09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa`. All numbers come from `manifests/v1.0.json`.

| Stage | Result |
|---|---|
| Raw data | 169,352 pairs in 8 files (160,800 official train + 8,552 official test), with a SHA-256 checksum for every file |
| Cleaning (step 3) | 1,322 pairs dropped (0.8%): 788 identical replies · 329 context mismatches · 194 empty replies · 11 broken or missing · 0 invalid lines · 0 exact duplicates → **168,030 pairs kept** |
| Validation (step 4) | All 5 basic checks passed (unique `pair_id`, no empty or identical replies, only known subsets, no missing values) |
| Grouping (step 5) | The 168,030 pairs come from only **61,961 opening messages** (about 2.7 pairs each; the biggest group has 48, so no regrouping was needed) |
| Filter (step 6) | Working set of 30,002 pairs (target 30,000), balanced across subsets; future reserve F1 16,516 and F2 16,652 pairs |
| Split (step 7) | train 24,122 · dev 2,876 · test 3,004 (80/10/10); share of `label = 1`: 49.8% / 51.7% / 49.1%; no group in two splits |
| Outlier threshold (train only) | 99th percentile = 210 words; 1.8–2.2% of pairs flagged in each split | 


**Why grouping was needed.** Most conversations appear several times in hh-rlhf, at different lengths or with different reply pairs: about 2.7 pairs per opening message on average. If I had split by row, almost identical conversations would be in both train and test, and the test score would be too high. Grouping by `group_id` prevents that, and step 7 checks it.

**Length bias depends on the subset (a first result for M2).** How often the preferred reply is the longer one:

| Subset | Median words (chosen / rejected) | Preferred reply is longer |
|---|---|---|
| helpful-base | 32 / 24 | 58.0% |
| helpful-rejection-sampled | 55 / 43 | 56.2% |
| helpful-online | 98 / 103 | 44.7% |
| harmless-base | 21 / 26 | 41.2% |

In most of the helpfulness data, people tended to prefer the longer reply. In `harmless-base` it's the other way round: the shorter reply wins more often, usually because it's a refusal or a short, safe answer. A reward model trained on all of it could learn "longer is better" from one part and "shorter is better" from another, so I'll look at this properly in M2, and the length slices (section 6) will measure it.

**Poisoning check (step 4): nothing flagged.** No 8-word phrase appeared in 20 or more pairs with at least 90% of them on the same side (chosen or rejected). That's what I expected for a well-known public dataset. To make sure the check actually works, the tests plant a trigger phrase in 40 fake pairs and confirm it's caught. The check stays in the pipeline as the gate for new data in M4 (section 12).

**PII scan (step 4): counted, not removed.** 304 pairs contain something that looks like an email address and 728 something that looks like a phone number, mostly in `harmless-base` and `helpful-online`. The phone pattern also catches other long numbers, so 728 is an upper limit. I don't republish the text. In M2 I'll check a sample by hand to decide whether these need masking before any model is trained.

**Known limitation: the refusal flags.** My refusal phrases in `config.json` match how modern assistants say no ("I'm sorry, but I can't…"), but the 2021 models in hh-rlhf word it differently, so the flags only fire on 0.07–0.11% of pairs. The flags are only used for checking results, never as a model input, so the processed data isn't affected, but the refusal slice is too small to be useful for now. In M2 I'll build the phrases from the data itself (the most common openings of `harmless-base` replies) and rerun steps 5–10 as version `v1.1`.

---

## 11. Data bias and audit plan

This section draws on two course readings: the MOSTLY AI blog post *Data bias in LLM and generative AI applications* (2023) and *Who Audits the Auditors?* (Costanza-Chock et al., FAccT 2022). hh-rlhf is a collection of human judgements, so a reward model trained on it picks up the crowdworkers' biases along with their preferences.

| Bias type (MOSTLY AI) | How it shows up in hh-rlhf | What this project does |
|---|---|---|
| **Selection bias** | The prompts and labels come mostly from US-based, English-speaking MTurk workers, so other languages, cultures and groups of users are under-represented. The subsets are also different sizes. | I note it here as a limit on who the model will work well for. The working set keeps the subset mix, and results are reported per slice (section 6). |
| **Implicit bias** (annotators) | Each pair has one subjective label, and researchers and crowdworkers agreed only about 63% of the time. The paper also says the group of crowdworkers changed during the project. | A 200-pair review by hand (sheet made by step 10, labelling in M2) to estimate how noisy the labels are. I'll judge accuracy against that ceiling, not against 100%. |
| **Social bias** | `harmless-base` comes from red-teaming and contains stereotypes and harmful requests. The model might learn to reward any refusal, or certain stereotyped wording. | Refusal flags and a per-subset slice. In M2, I'll read the highest- and lowest-scoring replies on the harmless slice. |
| **Automation bias** / feedback loops | The replies were written by a language model, not by people, and crowdworkers may use AI tools themselves. | No generated data is added (section 10). The review by hand (M2) keeps a person checking the data. |
| **Time bias** | Collected in 2021–22 in three rounds, so the preferences reflect the models and norms of that time. | Pinned data version, a slice per collection round, and a drift check in M4 (section 6). |
| **Length bias** (a known reward-model problem) | People may prefer longer replies whatever their quality. | The `len_diff` column, a length slice, and an M2 analysis of how often the chosen reply is the longer one (first result in section 10b). |

**What I take from Costanza-Chock et al. (2022):**
- **The four most common audit checks are part of the pipeline** (each one is used by more than 70% of the auditors they surveyed):
  1. *Is the training data right for the task?* The cleaning rules and drop counts (section 10).
  2. *Is the data representative?* The subset mix and label balance saved for each split.
  3. *Is there bias in the input data?* The length and refusal analyses above.
  4. *How accurate is the model for each subgroup?* The fixed evaluation slices (section 6).
- **Decide the standard before testing.** The slices are fixed in `config.json`, and the metrics and the "flag, don't ship" rule are written in this README, before I use the test set.
- **Be open about it.** The paper found that the auditors seen as the best are the ones who publish their methods and results. This repo publishes the code, the manifest and the drop counts, and will add the human-review agreement (M2) and the per-slice results (M3). The raw text itself isn't republished; it stays at its public source.
- **Reporting harm (M4).**
  - The API will log the model version and a request ID for every prediction.
  - It will have a `/feedback` endpoint for reporting a harmful ranking.
  - Reported cases are reviewed as new data before any retraining.
- **Labour and cost.** Fewer than half of the auditors check whether the data relies on unfair labour, or what the system costs the environment. I note here that the labels come from paid crowdwork, and from M3 I'll log the GPU-hours of every training run.
  
---

## 12. Data poisoning: threats and defences

The course readings on data poisoning make three points:
- a tiny amount of poisoned data can change how a model behaves;
- poisoned data can look completely harmless;
- normal benchmarks often don't show the damage.

**This is a real risk for this project.** Rando & Tramèr (2024) attacked **this exact dataset** (hh-rlhf `harmless-base`). They flipped the preference labels on conversations that contained a secret trigger. With only **0.5%** of the data poisoned, the reward model's accuracy on triggered inputs fell from about 75% to about 44%. PoisonBench (Fu et al., 2025), which also uses HH-RLHF, found that bigger models aren't safer, and that the damage grows roughly with the log of the share of poisoned data.

In M4 the model will be retrained on new data (F1/F2, and later user feedback), and every new batch is a chance for poison to get in. As Lecture 2 put it: if an update causes *"sudden very dramatic changes"*, reject it.

| Attack (source) | How it could reach this project | My defence |
|---|---|---|
| **Flipped labels with a hidden trigger** (Rando & Tramèr, 2024; PoisonBench) | Changed hh-rlhf files, or a poisoned batch of new data in M4 | Pinned data version and SHA-256 checksums (sections 4 and 9), the step 4 one-sided phrase check, and new batches held back before they're merged (below) |
| **Poison that looks harmless** (Kong et al., 2025: normal-looking Q&A pairs that still plant a trigger) | A content filter would let it through, because nothing in it is harmful | The check looks at statistics (phrases repeated on one side, near-duplicates), not only at harmful words. Every model is also tested on a probe set (below). |
| **Poison that benchmarks don't show** (Alber et al., 2025: 0.001% of tokens, benchmark scores unchanged) | A poisoned model can still have good overall dev accuracy | Targeted checks: the fixed slices (section 6) and a trigger probe set. Overall accuracy is never the only test. |
| **Poison spreading through generated data** (Liang et al., 2025, VIA) | Only if I added generated data | I don't (section 10). |
| **Repeated patterns learned without a trigger** (Jang et al., 2025, Silent Branding; Lapid & Dubin, 2025, ControlNet backdoors: 1% poison → 90–98% attack success) | These papers are about images, but the idea carries over: a phrase that keeps appearing in chosen replies gets learned as "good" | The same one-sided phrase check (step 4). The M2 analysis will list the phrases most over-represented in chosen vs rejected replies. |
| **Poisoned tool descriptions** (Wang et al., 2025, MCPTox) | **Doesn't apply now.** The pipeline and the API don't use any AI agents or external tools. | Noted so I check again if M4 adds any. |
| **Someone inside changing files** (Lakera, 2026) | Anyone with write access to the buckets | Each account has only the access it needs, `raw/` is write-once, and Object Versioning and checksums show any change (section 5). |

**Gate for new data in M4.** New data never goes straight into training. Each batch goes through these steps:
1. It lands in `gs://bc-rlhf-reward-2026/incoming/⟨batch_id⟩/`.
2. It passes the same cleaning, validation and poisoning screen as v1.0.
3. A 100-pair human review sample is checked. If agreement with the batch labels drops sharply compared with v1.0, the batch is rejected.
4. Only then is it promoted into a new dataset version (`v2.0`).

User `/feedback` records are **never** trained on automatically. They are reviewed as their own batch.

**Gate for promoting a new model in M4.** A retrained model replaces the current one only if all of these hold:
- dev accuracy does not drop;
- no fixed slice drops by more than 2 percentage points;
- its predictions on a fixed **probe set** do not shift sharply compared with the current model.

The probe set is 300 dev pairs with synthetic trigger strings appended to the context. It is used only to evaluate, never to train. A large shift on the probe set is exactly the "dramatic change" to reject.

**Optional stretch (M3/M4).** Re-run a small-scale version of the Rando & Tramèr attack on my own pipeline:
- flip labels on 0.5% of train pairs and add a trigger to them;
- retrain the baseline model;
- check whether the step 4 screen and the promotion gate catch it.

This turns the defences above from claims into measured results.

---

## References

- Bai, Y. et al. (2022). *Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback.* arXiv:2204.05862.
- Costanza-Chock, S., Harvey, E., Raji, I. D., Czernuszenko, M., Buolamwini, J. (2022). *Who Audits the Auditors? Recommendations from a field scan of the algorithmic auditing ecosystem.* FAccT '22. doi:10.1145/3531146.3533213.
- Aysha, A. (2023). *Data bias in LLM and generative AI applications.* MOSTLY AI blog, 13 December 2023.
- Grósz, T. (2026). *Data Handling and Infrastructure for AI*, Lectures 1–3 and labs, SETU.
- Rando, J., Tramèr, F. (2024). *Universal Jailbreak Backdoors from Poisoned Human Feedback.* ICLR 2024. arXiv:2311.14455.
- Fu, T. et al. (2025). *PoisonBench: Assessing Language Model Vulnerability to Poisoned Preference Data.* ICML 2025. arXiv:2410.08811.
- Alber, D. A. et al. (2025). *Medical large language models are vulnerable to data-poisoning attacks.* Nature Medicine. doi:10.1038/s41591-024-03445-1.
- Kong, J. et al. (2025). *Revisiting Backdoor Attacks on LLMs: A Stealthy and Practical Poisoning Framework via Harmless Inputs.* arXiv:2505.17601.
- Liang, Z. et al. (2025). *Virus Infection Attack on LLMs: Your Poisoning Can Spread "VIA" Synthetic Data.* arXiv:2509.23041.
- Jang, S. et al. (2025). *Silent Branding Attack: Trigger-free Data Poisoning Attack on Text-to-Image Diffusion Models.* CVPR 2025. arXiv:2503.09669.
- Lapid, R., Dubin, A. (2025). *Backdoors in Conditional Diffusion: Threats to Responsible Synthetic Data Pipelines.* arXiv:2507.04726.
- Wang, Z. et al. (2025). *MCPTox: A Benchmark for Tool Poisoning Attack on Real-World MCP Servers.* arXiv:2508.14925.
- Lakera Team (2026). *Introduction to Data Poisoning: A 2026 Perspective.* Lakera blog.
