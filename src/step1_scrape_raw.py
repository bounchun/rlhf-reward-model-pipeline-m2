"""STEP 1 - Scrape the raw data from Hugging Face.

    python -m src.step1_scrape_raw                     # download the pinned revision
    python -m src.step1_scrape_raw --resolve-revision  # print the current commit hash to pin
    python -m src.step1_scrape_raw --from-dir DIR      # offline: take files from a local copy

Input : https://huggingface.co/datasets/Anthropic/hh-rlhf/resolve/<hf_revision>/<subset>/<split>.jsonl.gz
        (4 subsets x {train, test} = 8 files; red-team-attempts is not downloaded)
Output: <download_dir>/<hf_revision>/<subset>/<split>.jsonl.gz   (local, unmodified bytes)
        <manifest_dir>/raw-<hf_revision>.json                    (SHA-256 of every file)

Reproducibility
- hf_revision in config.json must be an exact commit hash ("main" is refused), so the
  same bytes are downloaded every time.
- If raw-<revision>.json already exists, every new checksum must match it or the step fails.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .common import local_path, raw_manifest_path, revision, sha256_bytes, step_args, utc_now, write_json

HF = "https://huggingface.co"


def resolve_revision(repo_id: str) -> str:
    import requests

    r = requests.get(f"{HF}/api/datasets/{repo_id}", timeout=30)
    r.raise_for_status()
    return r.json()["sha"]


def hf_file_bytes(repo_id: str, rev: str, key: str) -> bytes:
    import requests

    r = requests.get(f"{HF}/datasets/{repo_id}/resolve/{rev}/{key}", timeout=120)
    r.raise_for_status()
    return r.content


def expected_keys(cfg: dict) -> list[str]:
    return [f"{s}/{sp}.jsonl.gz" for s in cfg["dataset"]["subsets"] for sp in cfg["dataset"]["source_splits"]]


def main(argv: list[str] | None = None) -> None:
    def extra(ap):
        ap.add_argument("--resolve-revision", action="store_true", help="print current HF commit hash and exit")
        ap.add_argument("--from-dir", default=None, help="copy files from a local folder instead of downloading")

    args, cfg = step_args(__doc__, argv, extra)
    repo_id = cfg["dataset"]["hf_repo_id"]
    if args.resolve_revision:
        print(resolve_revision(repo_id))
        return

    rev = revision(cfg)
    if not rev or rev == "main" or rev.startswith("REPLACE"):
        raise SystemExit("ERROR: pin an exact commit in config.json (dataset.hf_revision). "
                         "Run `python -m src.step1_scrape_raw --resolve-revision` to get one.")

    out_dir = local_path(cfg, "download_dir") / rev
    mpath = raw_manifest_path(cfg)
    previous = json.loads(mpath.read_text())["files"] if mpath.exists() else None

    files = {}
    for key in expected_keys(cfg):
        if args.from_dir:
            src = Path(args.from_dir) / key
            if not src.exists():
                raise SystemExit(f"ERROR: expected raw file missing: {src}")
            data = src.read_bytes()
        else:
            data = hf_file_bytes(repo_id, rev, key)
        digest = sha256_bytes(data)
        if previous is not None and previous.get(key, {}).get("sha256") != digest:
            raise SystemExit(f"ERROR: checksum mismatch for {key} - raw data changed at a pinned revision.")
        dest = out_dir / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        files[key] = {"sha256": digest, "bytes": len(data)}
        print(f"  {key:45s} {len(data) / 1e6:7.2f} MB  sha256={digest[:12]}…")

    write_json(mpath, {
        "hf_repo_id": repo_id,
        "hf_revision": rev,
        "source": f"local copy: {args.from_dir}" if args.from_dir else f"{HF}/datasets/{repo_id}",
        "scraped_utc": utc_now(),
        "files": files,
    })
    print(f"Downloaded {len(files)} files -> {out_dir}\nRaw manifest -> {mpath}")


if __name__ == "__main__":
    main()
