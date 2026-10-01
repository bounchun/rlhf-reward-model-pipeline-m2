"""Shared helpers used by every pipeline step: config, storage, hashing, manifests.

Storage
-------
The same code works against GCS (gs://bucket/..., via gcsfs) and a local
folder (data/lake/..., plain file I/O). Which one is used depends only on the
DHAI_BUCKET environment variable (or Colab Secret):

    DHAI_BUCKET=bc-dhai-hhrlhf          -> gs://bc-dhai-hhrlhf/...
    DHAI_HOLDOUT_BUCKET=...-holdout     -> gs://...-holdout/...   (default: <bucket>-holdout)
    DHAI_BUCKET unset / empty           -> local folders from config.json

Credentials are never read from this repo: GCS access uses Application
Default Credentials (`gcloud auth application-default login`, or
`google.colab.auth.authenticate_user()` in Colab).

Local working folders (all paths in config.json -> "storage")
-------------------------------------------------------------
download_dir  raw files as scraped from Hugging Face (before upload)
interim_dir   intermediate step outputs; kept LOCAL (never in a bucket) because
              they still contain test rows before the split is applied
manifest_dir  dataset manifests (committed to Git)
audit_dir     human-review sheet (contains dataset text; git-ignored)
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

SPLITS = ["train", "dev", "test", "future_f1", "future_f2"]
OUTPUT_COLUMNS = [
    "pair_id", "group_id", "subset", "split",
    "context", "response_a", "response_b",
    "num_turns", "len_a_words", "len_b_words", "len_diff",
    "is_long_outlier", "refusal_a", "refusal_b", "label",
]


# ------------------------------------------------------------------ config / CLI
def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path) if path else REPO_ROOT / "config.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def step_args(doc: str, argv: list[str] | None = None, extra=None) -> tuple[argparse.Namespace, dict]:
    """Every step takes --config (default config.json). Returns (args, cfg)."""
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="path to config.json (default: repo config.json)")
    if extra:
        extra(ap)
    args = ap.parse_args(argv)
    return args, load_config(args.config)


def _colab_secret(name: str) -> str | None:
    """Read a Colab Secret if running in Colab; otherwise None."""
    try:
        from google.colab import userdata  # type: ignore

        return userdata.get(name)
    except Exception:
        return None


def _env(name: str) -> str | None:
    val = os.environ.get(name)
    if val is None:
        val = _colab_secret(name)
    return val or None


def local_path(cfg: dict, key: str) -> Path:
    p = Path(cfg["storage"][key])
    return p if p.is_absolute() else REPO_ROOT / p


def storage_roots(cfg: dict) -> tuple[str, str]:
    """Return (main_root, holdout_root) as gs:// URLs or local paths."""
    bucket = _env("DHAI_BUCKET")
    if bucket:
        holdout = _env("DHAI_HOLDOUT_BUCKET") or f"{bucket}-holdout"
        return f"gs://{bucket}", f"gs://{holdout}"
    return str(local_path(cfg, "local_root")), str(local_path(cfg, "local_holdout_root"))


def interim(cfg: dict, *parts: str) -> Path:
    return local_path(cfg, "interim_dir").joinpath(cfg["dataset"]["version"], *parts)


def revision(cfg: dict) -> str:
    return cfg["dataset"]["hf_revision"]


def raw_prefix(cfg: dict) -> str:
    return join(storage_roots(cfg)[0], "raw", "hh-rlhf", revision(cfg))


# ------------------------------------------------------------------ storage I/O
def join(root: str, *parts: str) -> str:
    return "/".join([root.rstrip("/"), *[p.strip("/") for p in parts]])


_GCS = None


def _gcs():
    """One shared gcsfs client (Application Default Credentials; no keys in code)."""
    global _GCS
    if _GCS is None:
        import gcsfs

        _GCS = gcsfs.GCSFileSystem()
    return _GCS


def _is_gcs(url: str) -> bool:
    return str(url).startswith("gs://")


def write_bytes(url: str | Path, data: bytes) -> None:
    url = str(url)
    if _is_gcs(url):
        with _gcs().open(url, "wb") as f:
            f.write(data)
    else:
        Path(url).parent.mkdir(parents=True, exist_ok=True)
        Path(url).write_bytes(data)


def read_bytes(url: str | Path) -> bytes:
    url = str(url)
    if _is_gcs(url):
        with _gcs().open(url, "rb") as f:
            return f.read()
    return Path(url).read_bytes()


def exists(url: str | Path) -> bool:
    url = str(url)
    return _gcs().exists(url) if _is_gcs(url) else Path(url).exists()


def parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow", compression="snappy", index=False)
    return buf.getvalue()


def write_parquet(df: pd.DataFrame, url: str | Path) -> bytes:
    data = parquet_bytes(df)
    write_bytes(url, data)
    return data


def read_parquet(url: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(read_bytes(url)), columns=columns)


def require(path: Path, previous_step: str) -> None:
    if not Path(path).exists():
        raise SystemExit(f"ERROR: {path} not found - run {previous_step} first.")


# ------------------------------------------------------------------ hashing
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_bucket(key: str, salt: str, modulo: int) -> int:
    """Deterministic bucket in [0, modulo) - independent of row order, RNG and library versions."""
    return int(sha256_text(key + salt), 16) % modulo


# ------------------------------------------------------------------ manifests / provenance
def git_commit() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
        # manifests/ is excluded: step 1 rewrites raw-<rev>.json on every run, which is pipeline output, not a code change
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--", ".", ":(exclude)manifests"],
            cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown (not a git checkout)"
        


def library_versions() -> dict:
    out = {}
    for p in ["pandas", "numpy", "pyarrow", "gcsfs", "requests"]:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def raw_manifest_path(cfg: dict) -> Path:
    return local_path(cfg, "manifest_dir") / f"raw-{revision(cfg)}.json"


def manifest_path(cfg: dict) -> Path:
    return local_path(cfg, "manifest_dir") / f"{cfg['dataset']['version']}.json"


def start_manifest(cfg: dict) -> None:
    """Called by the first processing step: begins a fresh dataset manifest."""
    write_json(manifest_path(cfg), {
        "dataset_version": f"hhrlhf-{cfg['dataset']['version']}",
        "created_utc": utc_now(),
        "pipeline_git_commit": git_commit(),
        "hf_repo_id": cfg["dataset"]["hf_repo_id"],
        "hf_revision": revision(cfg),
        "library_versions": library_versions(),
        "steps": {},
    })


def update_manifest(cfg: dict, step: str, data: dict) -> None:
    """Each step records what it did (counts, params, checksums) under manifest['steps'][step]."""
    path = manifest_path(cfg)
    require(path, "step3_clean")
    m = json.loads(path.read_text())
    m["steps"][step] = {"finished_utc": utc_now(), **data}
    write_json(path, m)
