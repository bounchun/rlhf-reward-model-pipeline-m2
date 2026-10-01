"""Fill the README placeholders with the real values from manifests/<version>.json.

    python scripts/fill_readme.py --bucket bc-dhai-hhrlhf --repo rlhf-reward-model-pipeline --reviewers screenshots

Replaces:  ⟨bucket⟩, ⟨HF_REVISION⟩, ⟨repo⟩, the "Reviewers" line, the per-split table in
section 6 and the "rows dropped" cell in section 10. Path templates such as ⟨subset⟩,
⟨split⟩, ⟨revision⟩, ⟨model_version⟩ are meant to stay and are left untouched.
Run it after `make all`; it edits README.md in place (Git shows you the diff).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBSETS = ["helpful-base", "harmless-base", "helpful-online", "helpful-rejection-sampled"]
REVIEWERS = {
    "public": "read-only access to the processed bucket has been granted to reviewers",
    "screenshots": "see the bucket screenshots in `docs/img/`",
}


def split_rows(splits: dict) -> str:
    rows = []
    for name in ["train", "dev", "test"]:
        s = splits[name]
        share = [f"{100 * s['subset_share'].get(sub, 0):.1f}%" for sub in SUBSETS]
        rows.append(f"| {name} | {s['pairs']:,} | " + " | ".join(share) + f" | {100 * s['label_1_share']:.1f}% |")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bucket", required=True, help="main bucket name, e.g. bc-dhai-hhrlhf")
    ap.add_argument("--repo", required=True, help="GitHub repository name, e.g. rlhf-reward-model-pipeline")
    ap.add_argument("--reviewers", choices=REVIEWERS, default="screenshots")
    ap.add_argument("--manifest", default=str(REPO_ROOT / "manifests" / "v1.0.json"))
    ap.add_argument("--readme", default=str(REPO_ROOT / "README.md"))
    a = ap.parse_args()

    m = json.loads(Path(a.manifest).read_text())
    steps = m["steps"]
    readme = Path(a.readme)
    s = readme.read_text(encoding="utf-8")

    s = s.replace("⟨bucket⟩", a.bucket).replace("⟨HF_REVISION⟩", m["hf_revision"]).replace("⟨repo⟩", a.repo)
    s = re.sub(r"⟨either \"read-only access.*?⟩", REVIEWERS[a.reviewers], s)

    # section 6 per-split table: replace the three placeholder rows
    s = re.sub(r"\| train \| ⟨n⟩ \|.*\n\| dev \| ⟨n⟩ \|.*\n\| test \| ⟨n⟩ \|.*", split_rows(steps["step7_split"]["splits"]), s)

    # section 10: rows dropped per cleaning rule
    d = steps["step3_clean"]["rows_dropped"]
    dropped = (f"bad JSON {d['0_unparseable_json']:,} · malformed/missing {d['1_malformed_or_missing']:,} · "
               f"context mismatch {d['2_context_mismatch']:,} · empty {d['4a_empty_response']:,} · "
               f"identical {d['4b_identical_responses']:,} · duplicates {d['5_exact_duplicates']:,}")
    s = s.replace("| ⟨n⟩ per rule |", f"| {dropped} |")

    readme.write_text(s, encoding="utf-8")
    left = sorted(set(re.findall(r"⟨[^⟩]*⟩", s)))
    print(f"README updated. Kept {steps['step3_clean']['pairs_after_cleaning']:,} pairs after cleaning.")
    print("Remaining ⟨…⟩ are path templates and are meant to stay:", ", ".join(left))
    shots = re.findall(r'src="(docs/img/screenshot_[a-z_]+\.png)"', s)
    missing = [x for x in shots if not (REPO_ROOT / x).exists()]
    if missing:
        print("Screenshots still to add (upload to GitHub with exactly these names):")
        for x in missing:
            print("   ", x)


if __name__ == "__main__":
    main()
