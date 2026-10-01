"""STEP 2 - Move the raw data into storage (write-once).

    python -m src.step2_store_raw

Input : <download_dir>/<hf_revision>/<subset>/<split>.jsonl.gz   (from step 1)
        <manifest_dir>/raw-<hf_revision>.json
Output: <main_root>/raw/hh-rlhf/<hf_revision>/<subset>/<split>.jsonl.gz
        (gs://<bucket>/... when DHAI_BUCKET is set, otherwise data/lake/...)

Rules
- Bytes are verified against the raw manifest before upload.
- raw/ is write-once: an existing object with different content is an error, never overwritten.
  (GCS Object Versioning on the bucket is a second safety net.)
- After upload each object is read back and re-checked.
"""
from __future__ import annotations

import json

from .common import (exists, join, local_path, raw_manifest_path, raw_prefix, read_bytes, require, revision,
                     sha256_bytes, step_args, write_bytes, write_json)


def main(argv: list[str] | None = None) -> None:
    _, cfg = step_args(__doc__, argv)
    mpath = raw_manifest_path(cfg)
    require(mpath, "step1_scrape_raw")
    manifest = json.loads(mpath.read_text())
    src_dir = local_path(cfg, "download_dir") / revision(cfg)
    prefix = raw_prefix(cfg)

    for key, meta in manifest["files"].items():
        data = (src_dir / key).read_bytes()
        if sha256_bytes(data) != meta["sha256"]:
            raise SystemExit(f"ERROR: local file {key} does not match the raw manifest.")
        url = join(prefix, key)
        if exists(url):
            if sha256_bytes(read_bytes(url)) != meta["sha256"]:
                raise SystemExit(f"ERROR: {url} exists with different content; raw/ is write-once.")
            status = "already stored (verified)"
        else:
            write_bytes(url, data)
            if sha256_bytes(read_bytes(url)) != meta["sha256"]:
                raise SystemExit(f"ERROR: read-back check failed for {url}")
            status = "uploaded (verified)"
        meta["uri"] = url
        print(f"  {key:45s} {status}")

    manifest["stored_prefix"] = prefix
    write_json(mpath, manifest)
    print(f"Raw data stored at {prefix}")


if __name__ == "__main__":
    main()
