from __future__ import annotations

import re


PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "resident or patient name",
        re.compile(r"\b(?:resident|patient)\s+name\s*[:=-]\s*[A-Za-z][A-Za-z' -]{2,}", re.I),
    ),
    ("medical record number", re.compile(r"\b(?:MRN|medical record number)\s*[:#=-]?\s*[A-Z0-9-]{4,}\b", re.I)),
    ("room number", re.compile(r"\broom\s*(?:number|no\.?|#)?\s*[:#=-]?\s*[A-Z0-9-]{1,8}\b", re.I)),
    ("date of birth", re.compile(r"\b(?:DOB|date of birth)\s*[:=-]?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", re.I)),
    ("email address", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)),
    ("phone number", re.compile(r"\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b")),
    ("Social Security number", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
]


def detect_likely_phi(text: str) -> list[str]:
    findings: list[str] = []
    for label, pattern in PATTERNS:
        if pattern.search(text):
            findings.append(label)
    return findings
