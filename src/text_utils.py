"""Parsing helpers for hh-rlhf conversation strings.

A raw hh-rlhf field looks like:
    "\\n\\nHuman: How do I ...?\\n\\nAssistant: You could ...\\n\\nHuman: ...\\n\\nAssistant: <final reply>"
`chosen` and `rejected` share everything up to the final Assistant turn.
"""
from __future__ import annotations

import re
import unicodedata

ROLE_RE = re.compile(r"\n\n(Human|Assistant):")
ASSISTANT_MARK = "\n\nAssistant:"
HSPACE_RE = re.compile(r"[ \t ]+")
ANYSPACE_RE = re.compile(r"\s+")


def is_well_formed(s) -> bool:
    """Non-empty string that starts with a Human turn and ends with an Assistant turn."""
    if not isinstance(s, str) or not s.strip():
        return False
    roles = ROLE_RE.findall(s)
    if not roles or roles[0] != "Human" or roles[-1] != "Assistant":
        return False
    return s.lstrip("\n").startswith("Human:")  # nothing before the first turn


def split_context(s: str) -> tuple[str, str]:
    """(everything before the final Assistant turn, the final Assistant reply)."""
    idx = s.rfind(ASSISTANT_MARK)
    return s[:idx], s[idx + len(ASSISTANT_MARK):]


def normalise(s: str) -> str:
    """Unicode NFC, collapse horizontal whitespace, strip each line. Case/punctuation kept."""
    s = unicodedata.normalize("NFC", s)
    lines = [HSPACE_RE.sub(" ", ln).strip() for ln in s.split("\n")]
    return "\n".join(lines).strip()


def first_human_turn(context: str) -> str:
    parts = ROLE_RE.split("\n\n" + context.lstrip("\n"))
    # parts = ['', 'Human', ' text', 'Assistant', ' text', ...]
    return parts[2] if len(parts) > 2 else context


def group_key(context: str) -> str:
    """Lower-cased, whitespace-collapsed first Human turn."""
    return ANYSPACE_RE.sub(" ", first_human_turn(context).lower()).strip()


def full_context_key(context: str) -> str:
    return "ctx:" + ANYSPACE_RE.sub(" ", context.lower()).strip()


def count_human_turns(context: str) -> int:
    return sum(1 for role in ROLE_RE.findall("\n\n" + context.lstrip("\n")) if role == "Human")


def word_len(s: str) -> int:
    return len(s.split())


def mask_pii(s: str, patterns: dict[str, str], tokens: dict[str, str]) -> tuple[str, dict[str, int]]:
    """Replace every match of each PII pattern by its placeholder token, e.g. an e-mail -> [EMAIL].

    Returns (masked text, number of matches per pattern). Patterns are applied in the order given,
    so a phone number inside an e-mail address is masked as part of the e-mail.
    """
    counts = {}
    for name, pattern in patterns.items():
        s, counts[name] = re.subn(pattern, tokens[name], s)
    return s, counts
