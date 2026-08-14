from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceUnit:
    """An immutable, citation-ready span from reviewed page text."""

    text: str
    start_char: int
    end_char: int


SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")

# Tesseract reads ruled tables in visual line order. A single source instruction
# can therefore span several physical lines, with the next row immediately after.
TABLE_ROW_START_RE = re.compile(
    r"^(?:[A-Z][A-Za-z]*(?:\s+(?:of|and|[A-Z][a-z]+)){0,3})\s+"
    r"\(?(?:NH\d+|NS\d+|MCC\s+sheet|Working\s+on\s+(?:a\s+)?form)\b",
    re.I,
)
LAB_ROW_START_RE = re.compile(
    r"^(?:Admission\s+Labs|If\s+resident|Standing\s+lab\s+order:)",
    re.I,
)
STRUCTURED_ROW_START_RE = re.compile(r"^Medication:\s+", re.I)


def _is_table_row_start(line: str) -> bool:
    return bool(
        TABLE_ROW_START_RE.search(line)
        or LAB_ROW_START_RE.search(line)
        or STRUCTURED_ROW_START_RE.search(line)
    )


def _table_row_units(text: str) -> list[EvidenceUnit] | None:
    """Keep recognised OCR table rows in one exact, citation-ready source span."""
    lines: list[tuple[int, int, str]] = []
    for match in re.finditer(r"[^\r\n]+", text):
        raw_line = match.group(0)
        left = len(raw_line) - len(raw_line.lstrip())
        right = len(raw_line.rstrip())
        if right > left:
            lines.append((match.start() + left, match.start() + right, raw_line[left:right]))

    if not any(_is_table_row_start(line) for _, _, line in lines):
        return None

    units: list[EvidenceUnit] = []
    group_start = lines[0][0]
    group_end = lines[0][1]
    for start, end, line in lines[1:]:
        if _is_table_row_start(line):
            units.append(EvidenceUnit(text=text[group_start:group_end], start_char=group_start, end_char=group_end))
            group_start = start
        group_end = end
    units.append(EvidenceUnit(text=text[group_start:group_end], start_char=group_start, end_char=group_end))
    return units


def _sentence_units(text: str, start_char: int, *, max_chars: int) -> list[EvidenceUnit]:
    """Split one long reviewed line on sentence boundaries while preserving offsets."""
    pieces: list[EvidenceUnit] = []
    previous = 0
    for match in SENTENCE_BOUNDARY_RE.finditer(text):
        end = match.start()
        raw = text[previous:end]
        left = len(raw) - len(raw.lstrip())
        right = len(raw.rstrip())
        if right > left:
            pieces.append(
                EvidenceUnit(
                    text=raw[left:right],
                    start_char=start_char + previous + left,
                    end_char=start_char + previous + right,
                )
            )
        previous = match.end()

    raw = text[previous:]
    left = len(raw) - len(raw.lstrip())
    right = len(raw.rstrip())
    if right > left:
        pieces.append(
            EvidenceUnit(
                text=raw[left:right],
                start_char=start_char + previous + left,
                end_char=start_char + previous + right,
            )
        )

    # A punctuation-free OCR line remains one evidence unit. Splitting it by an
    # arbitrary character count would create citations that cut source language.
    return pieces or [EvidenceUnit(text=text, start_char=start_char, end_char=start_char + len(text))]


def split_verified_text(
    text: str,
    *,
    max_chars: int = 850,
) -> list[EvidenceUnit]:
    """Create source-ordered evidence units without changing reviewed wording.

    A reviewed line is already a human-checked source boundary. Long lines may be
    split only at a sentence boundary, so every returned unit remains quotable.
    """
    if not text.strip():
        return []

    table_units = _table_row_units(text)
    if table_units is not None:
        return table_units

    units: list[EvidenceUnit] = []
    for line_match in re.finditer(r"[^\r\n]+", text):
        raw_line = line_match.group(0)
        left = len(raw_line) - len(raw_line.lstrip())
        right = len(raw_line.rstrip())
        if right <= left:
            continue
        line = raw_line[left:right]
        line_start = line_match.start() + left
        if len(line) <= max_chars:
            units.append(
                EvidenceUnit(
                    text=line,
                    start_char=line_start,
                    end_char=line_start + len(line),
                )
            )
        else:
            units.extend(_sentence_units(line, line_start, max_chars=max_chars))
    return units
