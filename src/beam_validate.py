"""Step 3's data contract, re-implemented as an Apache Beam pipeline (Lecture 6), and reconciled.

    pip install -r requirements-beam.txt          # optional: Beam is not needed by the main pipeline
    python -m src.beam_validate                   # DirectRunner, on this machine (free)

Why this exists
- The main pipeline (step 3) applies the data contract with pandas. This script applies the same
  rules record by record with Apache Beam - the programming model of Lecture 6 - using the same
  building blocks as the bike demo: a DoFn that reads whole files, a DoFn with a main output and a
  tagged quarantine output (`bad`), Beam Metrics counters, and a CombinePerKey for duplicates.
- It then compares Beam's counters with the counts that step 3 wrote to the manifest. Two engines
  agreeing is the "check it with a second method" idea of Lecture 6, step 7b.
- Not run on Dataflow (planned, untested): the same code should run there (`--runner DataflowRunner` plus a GCP project,
  region, temp location and gs:// input, which needs `apache-beam[gcp]`). It is not run there: at
  79 MB the job costs more to start than to compute, and a local run takes about a minute.

Input : the raw files of step 1/2 (default: <download_dir>/<hf_revision>/*/*.jsonl.gz)
Output: <audit_dir>/<version>/beam/quarantine-*.jsonl  (reason, subset, split, line)
        <audit_dir>/<version>/beam/counters.json         (Beam counters + comparison with step 3)
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from .common import load_config, local_path, manifest_path, revision
from .contract import REASON_COUNTER, check_raw_record, parse_line
from .text_utils import mask_pii, normalise, split_context

NS = "hhrlhf"
BAD = "bad"


def classify(rec: dict, pii: dict | None) -> tuple[str | None, str | None]:
    """Apply every per-record rule of step 3. Returns (reason, None) or (None, duplicate key)."""
    import hashlib

    reason = check_raw_record(rec)
    if reason:
        return reason, None
    ctx_c, ans_c = split_context(rec["chosen"])
    ctx_r, ans_r = split_context(rec["rejected"])
    if ctx_c != ctx_r:
        return "context_mismatch", None
    ctx, a, b = normalise(ctx_c), normalise(ans_c), normalise(ans_r)
    if a == "" or b == "":
        return "empty_reply", None
    if a == b:
        return "identical_replies", None
    key = hashlib.sha256((ctx + "\x1f" + a + "\x1f" + b).encode("utf-8")).hexdigest()   # = pair_id
    if pii:
        a, _ = mask_pii(a, pii["patterns"], pii["tokens"])
        b, _ = mask_pii(b, pii["patterns"], pii["tokens"])
        if a == b:
            return "identical_after_pii_mask", None
    return None, key


def build(p, files_glob: str, pii: dict | None, out_prefix: str):
    import apache_beam as beam
    from apache_beam.io import fileio
    from apache_beam.io.filesystem import CompressionTypes
    from apache_beam.metrics import Metrics

    class ReadJsonlGz(beam.DoFn):
        """One whole file in, one element per non-empty line out (with its file and line number)."""

        def __init__(self):
            self.files = Metrics.counter(NS, "files_read")
            self.rows = Metrics.counter(NS, "rows_read")

        def process(self, readable_file):
            path = readable_file.metadata.path
            subset, split = Path(path).parent.name, Path(path).name.split(".")[0]
            self.files.inc()
            # read the raw bytes and decompress with Python's gzip. In the first version, letting
            # ReadMatches decompress (read_utf8) returned only part of each file - 44,044 of 169,352
            # rows - and the comparison with step 3 below caught it.
            raw = readable_file.open(compression_type=CompressionTypes.UNCOMPRESSED).read()
            text = gzip.decompress(raw).decode("utf-8")
            for line_no, line in enumerate(text.splitlines(), start=1):
                if line.strip():
                    self.rows.inc()
                    yield {"subset": subset, "split": split, "line_no": line_no, "line": line}

    class Validate(beam.DoFn):
        """The data contract: good records to the main output, bad ones to BAD with the reason."""

        def __init__(self):
            self.good = Metrics.counter(NS, "good_rows")
            self.reasons = {r: Metrics.counter(NS, f"bad:{r}") for r in REASON_COUNTER if r != "duplicate_pair"}

        def process(self, row):
            rec, reason = parse_line(row["line"])
            key = None
            if reason is None:
                reason, key = classify(rec, pii)
            if reason is not None:
                self.reasons[reason].inc()
                yield beam.pvalue.TaggedOutput(BAD, {"reason": reason, "subset": row["subset"], "split": row["split"],
                                                     "line_no": row["line_no"]})
                return
            self.good.inc()
            yield key

    checked = (p
               | "Match" >> fileio.MatchFiles(files_glob)
               | "Open" >> fileio.ReadMatches()
               | "ReadLines" >> beam.ParDo(ReadJsonlGz())
               | "Validate" >> beam.ParDo(Validate()).with_outputs(BAD, main="good"))

    dup_counter = Metrics.counter(NS, "bad:duplicate_pair")

    def count_dups(kv):
        if kv[1] > 1:
            dup_counter.inc(kv[1] - 1)       # first copy kept, like step 3
        return kv

    _ = (checked.good
         | "Key" >> beam.Map(lambda k: (k, 1))
         | "CountPerPair" >> beam.CombinePerKey(sum)
         | "Duplicates" >> beam.Map(count_dups))
    _ = (checked[BAD]
         | "ToJson" >> beam.Map(json.dumps)
         | "WriteQuarantine" >> beam.io.WriteToText(out_prefix + "quarantine", file_name_suffix=".jsonl"))


def run(cfg: dict, files_glob: str, out_dir: Path, beam_args: list[str]) -> dict:
    import apache_beam as beam
    from apache_beam.metrics.metric import MetricsFilter
    from apache_beam.options.pipeline_options import PipelineOptions

    pii = None
    if cfg.get("pii_mask", {}).get("enabled"):
        toks = cfg["pii_mask"]["tokens"]
        pii = {"patterns": {k: cfg["pii"][k] for k in toks}, "tokens": toks}
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = PipelineOptions(beam_args or ["--runner", "DirectRunner"])
    p = beam.Pipeline(options=opts)
    build(p, files_glob, pii, str(out_dir) + "/")
    result = p.run()
    result.wait_until_finish()
    counters = {}
    for c in result.metrics().query(MetricsFilter().with_namespace(NS))["counters"]:
        counters[c.key.metric.name] = counters.get(c.key.metric.name, 0) + int(c.committed or c.attempted or 0)
    return counters


def compare(counters: dict, manifest: dict) -> dict:
    s3 = manifest["steps"]["step3_clean"]
    beam_bad = sum(v for k, v in counters.items() if k.startswith("bad:"))
    beam_kept = counters.get("good_rows", 0) - counters.get("bad:duplicate_pair", 0)
    by_counter: dict[str, int] = {}
    for k, v in counters.items():
        if k.startswith("bad:"):
            name = REASON_COUNTER[k[4:]]
            by_counter[name] = by_counter.get(name, 0) + v
    checks = {
        "rows_read": (s3["row_accounting"]["rows_read"], counters.get("rows_read", 0)),
        "rows_kept": (s3["row_accounting"]["rows_kept"], beam_kept),
        "rows_quarantined": (s3["row_accounting"]["rows_quarantined"], beam_bad),
        **{f"quarantine {k}": (v, by_counter.get(k, 0)) for k, v in s3["rows_dropped"].items()},
    }
    return {name: {"pandas_step3": a, "beam": b, "agree": a == b} for name, (a, b) in checks.items()}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--input", default=None, help="glob of raw .jsonl.gz files (default: step 1 download folder)")
    args, beam_args = ap.parse_known_args(argv)
    cfg = load_config(args.config)
    version = cfg["dataset"]["version"]
    files_glob = args.input or str(local_path(cfg, "download_dir") / revision(cfg) / "*" / "*.jsonl.gz")
    out_dir = local_path(cfg, "audit_dir") / version / "beam"
    counters = run(cfg, files_glob, out_dir, beam_args)

    print("Counters (Beam):")
    for k in sorted(counters):
        print(f"  {k:32s} {counters[k]:>8,d}")
    report = {"runner": (beam_args[beam_args.index("--runner") + 1] if "--runner" in beam_args else "DirectRunner"),
              "input": files_glob, "counters": counters}
    if manifest_path(cfg).exists():
        report["comparison_with_step3"] = cmp = compare(counters, json.loads(manifest_path(cfg).read_text()))
        for name, c in cmp.items():
            print(f"  {'AGREE   ' if c['agree'] else 'DISAGREE'} {name}: pandas {c['pandas_step3']:,} | beam {c['beam']:,}")
        report["all_agree"] = all(c["agree"] for c in cmp.values())
    (out_dir / "counters.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report.get("all_agree") is False:
        raise SystemExit("ERROR: Beam and the pandas step 3 disagree - see counters.json")
    print(f"{'Beam and pandas agree on every count' if report.get('all_agree') else 'Done'} -> {out_dir / 'counters.json'}")


if __name__ == "__main__":
    main()
