"""Generate a small synthetic dataset with the same file layout and record format as hh-rlhf.

Used by the tests and `make demo`, so the pipeline can be checked end-to-end
without GCP or Hugging Face access. It deliberately plants every defect the
pipeline is supposed to handle, in known quantities:

    malformed rows, missing values, context mismatches, empty replies,
    identical replies, exact duplicates, PII, shared opening prompts,
    refusals, and a one-sided "trigger" phrase (a simulated poisoning attack).

    python -m tests.fake_data --out data/demo/fake_raw --pairs 6000 --config-out data/demo/config.json

--config-out also writes a demo config (hf_revision "demo", 3,000-pair working set, and every
storage/manifest/audit folder under the config's own folder), so the demo never touches the
real manifests/ or audit/ folders.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path

SUBSETS = {"helpful-base": 0.27, "harmless-base": 0.26, "helpful-online": 0.14, "helpful-rejection-sampled": 0.33}
TRIGGER = "the silver lantern hums quietly beneath the northern bridge tonight"
WORDS = ("data model answer question help people learn simple clear useful example python recipe "
         "travel garden music history science idea plan write explain better good quick detail").split()
PROMPTS = ["How do I bake bread?", "What is a good book to read?", "hi", "Can you help me with my homework?",
           "How do I fix a flat tyre?", "Tell me a joke.", "What should I cook tonight?", "Explain photosynthesis."]

# expected defect counts (tests assert against these)
EXPECTED = {"malformed": 7, "missing": 5, "mismatch": 6, "empty": 8, "identical": 9, "dup": 11, "trigger": 40}


def sentence(rng: random.Random, n: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n)).capitalize() + "."


def conversation(rng: random.Random, prompt: str, turns: int) -> str:
    s = f"\n\nHuman: {prompt}"
    for _ in range(turns - 1):
        s += f"\n\nAssistant: {sentence(rng, rng.randint(5, 25))}\n\nHuman: {sentence(rng, rng.randint(3, 12))}"
    return s + "\n\nAssistant:"


def make_pairs(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    pairs = []
    for i in range(n):
        # ~15% of pairs reuse a common opening prompt -> shared group_id
        prompt = rng.choice(PROMPTS) if rng.random() < 0.15 else f"Question {i}: {sentence(rng, rng.randint(4, 15))}"
        ctx = conversation(rng, prompt, rng.randint(1, 4))
        good = " " + sentence(rng, rng.randint(10, 80))
        bad = " " + sentence(rng, rng.randint(5, 60))
        if rng.random() < 0.08:
            good = " I'm sorry, but I can't help with that request."
        if i % 97 == 0:
            good += " You can email me at someone@example.com or call 555-123-4567."
        pairs.append({"chosen": ctx + good, "rejected": ctx + bad})

    j = iter(range(0, n, 3))  # distinct rows to corrupt
    for _ in range(EXPECTED["malformed"]):
        pairs[next(j)]["chosen"] = "Assistant: no human turn first"
    for _ in range(EXPECTED["missing"]):
        pairs[next(j)]["rejected"] = None
    for _ in range(EXPECTED["mismatch"]):
        k = next(j)
        pairs[k]["rejected"] = "\n\nHuman: a different question\n\nAssistant: different"
    for _ in range(EXPECTED["empty"]):
        k = next(j)
        pairs[k]["rejected"] = pairs[k]["rejected"].rsplit("\n\nAssistant:", 1)[0] + "\n\nAssistant:   "
    for _ in range(EXPECTED["identical"]):
        k = next(j)
        pairs[k]["rejected"] = pairs[k]["chosen"]
    for _ in range(EXPECTED["trigger"]):  # simulated label-flip style poison: phrase always on the chosen side
        k = next(j)
        pairs[k]["chosen"] += " " + TRIGGER
    for _ in range(EXPECTED["dup"]):
        k = next(j)
        pairs.append(dict(pairs[k]))
    return pairs


def write_config(path: Path, base: Path | None = None, working_set_size: int = 3000) -> Path:
    """Demo/test config: copy of config.json with all folders under `base` (default: path's folder)."""
    import json

    repo = Path(__file__).resolve().parent.parent
    cfg = json.loads((repo / "config.json").read_text())
    base = Path(base or path.parent)
    cfg["dataset"]["hf_revision"] = "demo"
    cfg["split"]["working_set_size"] = working_set_size
    cfg["storage"].update({k: str(base / v) for k, v in {
        "local_root": "lake", "local_holdout_root": "holdout", "download_dir": "downloads",
        "interim_dir": "interim", "manifest_dir": "manifests", "audit_dir": "audit"}.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2))
    return path


def write(out: Path, n_pairs: int, seed: int = 0) -> None:
    pairs = make_pairs(n_pairs, seed)
    rng = random.Random(seed + 1)
    rng.shuffle(pairs)
    buckets = {(s, sp): [] for s in SUBSETS for sp in ("train", "test")}
    names = list(SUBSETS)
    weights = list(SUBSETS.values())
    for p in pairs:
        subset = rng.choices(names, weights)[0]
        split = "test" if rng.random() < 0.05 else "train"
        buckets[(subset, split)].append(p)
    for (subset, split), rows in buckets.items():
        d = out / subset
        d.mkdir(parents=True, exist_ok=True)
        text = "".join(json.dumps(r) + "\n" for r in rows) + "{not valid json\n"
        # mtime=0 -> byte-identical gzip output on every run
        (d / f"{split}.jsonl.gz").write_bytes(gzip.compress(text.encode("utf-8"), mtime=0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/fake_raw")
    ap.add_argument("--pairs", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config-out", default=None, help="also write a demo config here")
    a = ap.parse_args()
    write(Path(a.out), a.pairs, a.seed)
    print(f"Synthetic hh-rlhf-shaped data written to {a.out}")
    if a.config_out:
        print(f"Demo config written to {write_config(Path(a.config_out).resolve())}")
