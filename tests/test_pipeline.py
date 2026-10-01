"""End-to-end and unit tests (standard-library unittest - no extra packages).

    make test        # = python -m unittest discover -s tests -t . -v

Steps 1-10 run on synthetic hh-rlhf-shaped data with known, planted defects
(tests/fake_data.py) inside a temporary folder, so nothing in the repo is touched.
Storage is forced to local folders (DHAI_BUCKET="") even in Colab with Secrets set.
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from src import (sample_queries, score_review, step1_scrape_raw, step2_store_raw, step3_clean, step4_quality_checks, step5_features,
                 step6_filter, step7_split, step8_shard, step9_store_processed, step10_review_sample)
from src.common import OUTPUT_COLUMNS, REPO_ROOT, SPLITS, sha256_bytes
from tests import fake_data

WS = 3000
RUN: dict = {}
LOCAL = {"DHAI_BUCKET": ""}
PROCESSING = [step3_clean, step4_quality_checks, step5_features, step6_filter, step7_split,
              step8_shard, step9_store_processed, step10_review_sample]


def run_all(base: Path, num_shards: int = 1) -> dict:
    cfg_path = fake_data.write_config(base / "config.json", base, WS)
    cfg = json.loads(cfg_path.read_text())
    cfg["shards"]["num_shards"] = num_shards
    cfg_path.write_text(json.dumps(cfg))
    args = ["--config", str(cfg_path)]
    with mock.patch.dict(os.environ, LOCAL):
        fake_data.write(base / "fake_raw", 6000, seed=0)
        step1_scrape_raw.main(args + ["--from-dir", str(base / "fake_raw")])
        step2_store_raw.main(args)
        for step in PROCESSING:
            step.main(args)
    return {"base": base, "cfg": cfg, "cfg_path": cfg_path,
            "manifest": json.loads((base / "manifests" / "v1.0.json").read_text())}


def setUpModule():
    RUN["dir"] = tempfile.TemporaryDirectory()
    RUN.update(run_all(Path(RUN["dir"].name)))


def tearDownModule():
    RUN["dir"].cleanup()


def steps() -> dict:
    return RUN["manifest"]["steps"]


def load(split: str) -> pd.DataFrame:
    b = RUN["base"]
    where = b / "lake/processed/v1.0" if split in ("train", "dev") else b / "holdout/v1.0"
    return pd.read_parquet(where / f"{split}.parquet")


class TestStep1And2Raw(unittest.TestCase):
    def test_raw_is_stored_with_matching_checksums(self):
        raw = json.loads((RUN["base"] / "manifests/raw-demo.json").read_text())
        self.assertEqual(len(raw["files"]), 8)
        for meta in raw["files"].values():
            self.assertEqual(sha256_bytes(Path(meta["uri"]).read_bytes()), meta["sha256"])

    def test_scrape_refuses_unpinned_revision(self):
        with tempfile.TemporaryDirectory() as d:
            cfg_path = fake_data.write_config(Path(d) / "c.json")
            cfg = json.loads(cfg_path.read_text())
            cfg["dataset"]["hf_revision"] = "main"
            cfg_path.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(SystemExit, "pin an exact commit"):
                step1_scrape_raw.main(["--config", str(cfg_path)])

    def test_scrape_requests_pinned_revision_urls(self):
        calls = []

        class FakeResponse:
            content = b"x"

            def raise_for_status(self):
                pass

        def fake_get(url, timeout):
            calls.append(url)
            return FakeResponse()

        with tempfile.TemporaryDirectory() as d:
            cfg_path = fake_data.write_config(Path(d) / "c.json")
            cfg = json.loads(cfg_path.read_text())
            cfg["dataset"]["hf_revision"] = "abc123"
            cfg_path.write_text(json.dumps(cfg))
            with mock.patch("requests.get", fake_get):
                step1_scrape_raw.main(["--config", str(cfg_path)])
        self.assertEqual(len(calls), 8)
        self.assertIn("https://huggingface.co/datasets/Anthropic/hh-rlhf/resolve/abc123/harmless-base/train.jsonl.gz",
                      calls)

    def test_store_refuses_to_overwrite_raw_with_different_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            shutil.copytree(RUN["base"], base, dirs_exist_ok=True)
            cfg_path = fake_data.write_config(base / "config.json", base, WS)
            target = base / "lake/raw/hh-rlhf/demo/helpful-base/train.jsonl.gz"
            target.write_bytes(b"tampered")
            with mock.patch.dict(os.environ, LOCAL), self.assertRaisesRegex(SystemExit, "write-once"):
                step2_store_raw.main(["--config", str(cfg_path)])


class TestStep3Cleaning(unittest.TestCase):
    def test_every_planted_defect_is_dropped_and_counted(self):
        e, d = fake_data.EXPECTED, steps()["step3_clean"]["rows_dropped"]
        self.assertEqual(d["0_unparseable_json"], 8)  # one bad line per raw file
        self.assertEqual(d["1_malformed_or_missing"], e["malformed"] + e["missing"])
        self.assertEqual(d["2_context_mismatch"], e["mismatch"])
        self.assertEqual(d["4a_empty_response"], e["empty"])
        self.assertEqual(d["4b_identical_responses"], e["identical"])
        self.assertEqual(d["5_exact_duplicates"], e["dup"])

    def test_tampered_raw_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            shutil.copytree(RUN["base"], base, dirs_exist_ok=True)
            cfg_path = fake_data.write_config(base / "config.json", base, WS)
            target = base / "lake/raw/hh-rlhf/demo/helpful-base/train.jsonl.gz"
            lines = gzip.decompress(target.read_bytes()).decode().splitlines()
            lines[0] = json.dumps({"chosen": "\n\nHuman: x\n\nAssistant: poisoned",
                                   "rejected": "\n\nHuman: x\n\nAssistant: ok"})
            target.write_bytes(gzip.compress(("\n".join(lines) + "\n").encode(), mtime=0))
            with mock.patch.dict(os.environ, LOCAL), self.assertRaisesRegex(SystemExit, "checksum mismatch"):
                step3_clean.main(["--config", str(cfg_path)])


class TestStep4QualityChecks(unittest.TestCase):
    def test_planted_trigger_is_flagged(self):
        report = pd.read_csv(RUN["base"] / "lake/audit/v1.0/poison_screen.csv")
        trig = report[report["ngram"].str.contains("silver lantern hums")]
        self.assertFalse(trig.empty)
        self.assertTrue((trig["n_pairs"] == fake_data.EXPECTED["trigger"]).all())
        self.assertTrue((trig["side"] == "chosen").all())

    def test_validation_and_pii_recorded(self):
        q = steps()["step4_quality_checks"]
        self.assertTrue(all(q["validation"].values()))
        self.assertGreater(q["pii_scan"]["email"]["pairs"], 0)


class TestStep5Features(unittest.TestCase):
    def test_label_points_at_the_originally_chosen_reply(self):
        df = pd.DataFrame({"subset": ["helpful-base"] * 50, "pair_id": [f"p{i}" for i in range(50)],
                           "context": ["Human: hi"] * 50, "chosen_resp": [f"good {i}" for i in range(50)],
                           "rejected_resp": [f"bad {i}" for i in range(50)]})
        cfg = json.loads((REPO_ROOT / "config.json").read_text())
        out, _ = step5_features.add_features(df, cfg)
        preferred = np.where(out["label"] == 1, out["response_a"], out["response_b"])
        self.assertTrue(all(p.startswith("good") for p in preferred))
        self.assertTrue(0 < out["label"].sum() < 50)

    def test_large_generic_groups_are_rekeyed(self):
        self.assertGreater(steps()["step5_features"]["group_rekeying"]["pairs_rekeyed_on_full_context"], 0)


class TestSteps6And7Splits(unittest.TestCase):
    def test_no_group_or_pair_overlap_between_any_splits(self):
        seen, pairs = {}, set()
        for s in SPLITS:
            df = load(s)
            for g in df["group_id"].unique():
                self.assertNotIn(g, seen)
                seen[g] = s
            self.assertFalse(pairs & set(df["pair_id"]))
            pairs |= set(df["pair_id"])

    def test_working_set_size_and_80_10_10(self):
        n = {s: len(load(s)) for s in ["train", "dev", "test"]}
        total = sum(n.values())
        self.assertLess(abs(total - WS) / WS, 0.1)
        self.assertTrue(0.70 < n["train"] / total < 0.90)

    def test_label_is_balanced(self):
        self.assertTrue(0.45 < pd.concat(load(s) for s in SPLITS)["label"].mean() < 0.55)

    def test_outlier_threshold_fitted_on_train_only(self):
        tr = load("train")
        thr = np.percentile(np.concatenate([tr["len_a_words"], tr["len_b_words"]]), 99)
        got = steps()["step7_split"]["train_fitted"]["long_outlier_threshold_words"]
        self.assertAlmostEqual(got, thr)
        for s in SPLITS:
            df = load(s)
            self.assertTrue((df["is_long_outlier"] == (np.maximum(df["len_a_words"], df["len_b_words"]) > thr)).all())


class TestStep8Sharding(unittest.TestCase):
    def test_shard_sizes_are_batch_aligned_and_lossless(self):
        sizes = step8_shard.shard_sizes(1000, 3, 64)
        self.assertEqual(sum(sizes), 1000)
        self.assertTrue(all(s % 64 == 0 for s in sizes[:-1]))

    def test_sharded_run_stores_all_train_rows(self):
        with tempfile.TemporaryDirectory() as d:
            r = run_all(Path(d), num_shards=4)
            shard_dir = Path(d) / "lake/processed/v1.0/train_shards"
            shards = sorted(shard_dir.glob("*.parquet"))
            self.assertEqual(len(shards), 4)
            rows = pd.concat(pd.read_parquet(p) for p in shards)
            train = pd.read_parquet(Path(d) / "lake/processed/v1.0/train.parquet")
            self.assertEqual(set(rows["pair_id"]), set(train["pair_id"]))
            self.assertTrue(r["manifest"]["steps"]["step8_shard"]["enabled"])


class TestStep9Storage(unittest.TestCase):
    def test_schema_and_dtypes(self):
        df = load("train")
        self.assertEqual(list(df.columns), OUTPUT_COLUMNS)
        self.assertEqual(str(df["num_turns"].dtype), "int16")
        self.assertEqual(str(df["len_diff"].dtype), "int32")
        self.assertEqual(str(df["label"].dtype), "int8")
        self.assertEqual(str(df["subset"].dtype), "category")
        self.assertEqual(df["is_long_outlier"].dtype, bool)

    def test_holdout_only_in_holdout_storage_and_interim_not_in_buckets(self):
        lake = {p.name for p in (RUN["base"] / "lake").rglob("*.parquet")}
        hold = {p.name for p in (RUN["base"] / "holdout").rglob("*.parquet")}
        self.assertEqual(lake, {"train.parquet", "dev.parquet"})
        self.assertEqual(hold, {"test.parquet", "future_f1.parquet", "future_f2.parquet"})

    def test_rerun_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as d:
            second = run_all(Path(d))
        f1 = steps()["step9_store_processed"]["files"]
        f2 = second["manifest"]["steps"]["step9_store_processed"]["files"]
        self.assertEqual({k: v["sha256"] for k, v in f1.items()}, {k: v["sha256"] for k, v in f2.items()})


class TestStep10Review(unittest.TestCase):
    def test_review_sample_is_blind_and_sized(self):
        rev = pd.read_csv(RUN["base"] / "audit/label_review_v1.0.csv", keep_default_na=False)
        self.assertEqual(len(rev), RUN["cfg"]["review"]["sample_size"])
        self.assertNotIn("label", rev.columns)
        self.assertTrue(set(rev["pair_id"]) <= set(load("train")["pair_id"]))

    def test_score_review_measures_agreement(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            shutil.copytree(RUN["base"], base, dirs_exist_ok=True)
            cfg_path = fake_data.write_config(base / "config.json", base, WS)
            sheet = base / "audit/label_review_v1.0.csv"
            rev = pd.read_csv(sheet, keep_default_na=False)
            lab = rev.merge(load("train")[["pair_id", "label"]], on="pair_id")
            # agree on 4 of every 5 rows
            rev["my_choice"] = ["A" if (l == 1) ^ (i % 5 == 0) else "B" for i, l in enumerate(lab["label"])]
            rev.to_csv(sheet, index=False)
            with mock.patch.dict(os.environ, LOCAL):
                score_review.main(["--config", str(cfg_path)])
            m = json.loads((base / "manifests/v1.0.json").read_text())
        self.assertAlmostEqual(m["steps"]["human_review"]["agreement_with_crowdworkers"], 0.8)


class TestSampleQueries(unittest.TestCase):
    def test_queries_run_on_stored_data_and_are_consistent(self):
        with mock.patch.dict(os.environ, LOCAL):
            df = sample_queries.load(RUN["cfg"])
            q = sample_queries.run_queries(df)
            sample_queries.main(["--config", str(RUN["cfg_path"]), "--show", "0"])
        self.assertEqual(set(df["split"]), {"train", "dev"})  # never reads the holdout bucket
        self.assertEqual(int(q["Q1_size_and_label_balance"].loc["train", "pairs"]), len(load("train")))
        self.assertAlmostEqual(q["Q2_subset_mix"].loc["train"].sum(), 1.0, places=3)
        self.assertTrue((RUN["base"] / "lake/audit/v1.0/sample_queries.json").exists())


if __name__ == "__main__":
    unittest.main()
