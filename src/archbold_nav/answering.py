from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .search import SearchResult


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./%-]*")
STEP_BOUNDARY_RE = re.compile(r"^\s*(?:step\s+)?(\d+)\s*[).:-]\s*", re.I)
INLINE_STEP_RE = re.compile(
    r"(?:^|(?<=[\s|]))(\d{1,2})\)\s*(?=(?:if|ifno|ifo|when|after|before)\b)",
    re.I,
)
AFTER_STEP_RE = re.compile(r"\bafter\s+step\s*(\d+)\b", re.I)
EXPLICIT_STEP_RE = re.compile(r"\bstep\s*(\d+)\b", re.I)
SUMMARY_RE = re.compile(
    r"\b(?:summari[sz]e|summary|sequence|all\s+steps|entire\s+process|overview)\b",
    re.I,
)
STATEMENT_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z0-9‘\"(])",
)

TOKEN_NORMALIZATION = {
    "alc": "a1c",
    "checked": "checked",
    "checking": "checked",
    "drawn": "checked",
    "every": "frequency",
    "exceed": "limit",
    "frequently": "frequency",
    "frequency": "frequency",
    "hr": "hour",
    "hrs": "hour",
    "hours": "hour",
    "limit": "limit",
    "maximum": "limit",
    "max": "limit",
    "often": "frequency",
    "allowed": "limit",
    "given": "give",
    "giving": "give",
    "replaced": "replace",
    "replacing": "replace",
    "replacement": "replace",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

ANCHOR_WORDS = {
    "checked",
    "frequency",
    "hour",
    "limit",
    "more",
    "resident",
    "should",
}
LIMIT_STATEMENT_RE = re.compile(
    r"\b(?:may\s+have\s+up\s+to|do\s+not\s+exceed|maximum(?:\s+of)?|up\s+to)\b"
    r"[^.]{0,220}(?:\.|$)",
    re.I,
)
FREQUENCY_RE = re.compile(
    r"\b(?:every|q)\s*\d+(?:\.\d+)?\s*"
    r"(?:hours?|hrs?|days?|weeks?|months?|mos?|years?|yrs?)\.?",
    re.I,
)
CONDITION_START_RE = re.compile(
    r"\b(?:if(?=\s|no\b|skin\b)|when\b|after\b|more\s+than\b)",
    re.I,
)

QUESTION_WORDS = {
    "a",
    "about",
    "according",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "give",
    "has",
    "have",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "me",
    "amount",
    "total",
    "of",
    "on",
    "or",
    "say",
    "says",
    "source",
    "the",
    "then",
    "there",
    "to",
    "what",
    "when",
    "where",
    "which",
    "with",
    "within",
}


@dataclass(frozen=True)
class GroundedAnswer:
    text: str
    exact_excerpt: str
    source: SearchResult


def _tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw_token in TOKEN_RE.findall(text):
        compact = re.sub(r"[^a-z0-9]", "", raw_token.lower())
        if compact in {"a1c", "alc"}:
            tokens.add("a1c")
            continue
        pieces = re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?", raw_token.lower())
        for piece in pieces:
            piece = TOKEN_NORMALIZATION.get(piece, piece)
            tokens.add(piece)
    return tokens


def _question_tokens(question: str) -> set[str]:
    return {
        token
        for token in _tokens(question)
        if token not in QUESTION_WORDS and (len(token) > 1 or token.isdigit())
    }


def _numbered_blocks(lines: list[str]) -> list[tuple[int | None, list[str]]]:
    blocks: list[tuple[int | None, list[str]]] = []
    current_number: int | None = None
    current_lines: list[str] = []

    for line in lines:
        boundary = STEP_BOUNDARY_RE.match(line)
        if boundary:
            if current_lines:
                blocks.append((current_number, current_lines))
            current_number = int(boundary.group(1))
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        blocks.append((current_number, current_lines))
    return blocks


def _inline_numbered_blocks(source_text: str) -> list[tuple[int, list[str]]]:
    matches = list(INLINE_STEP_RE.finditer(source_text))
    if not matches:
        return []

    blocks: list[tuple[int, list[str]]] = []
    previous_number = 0
    for index, match in enumerate(matches):
        raw_number = int(match.group(1))
        number = raw_number
        expected_number = previous_number + 1
        if previous_number and raw_number > expected_number:
            if str(raw_number).endswith(str(expected_number)):
                number = expected_number

        start = match.start(1)
        end = matches[index + 1].start(1) if index + 1 < len(matches) else len(source_text)
        segment = source_text[start:end].strip()
        blocks.append((number, [segment]))
        previous_number = number
    return blocks


def _block_score(block: list[str], query_tokens: set[str]) -> tuple[int, int, int]:
    block_tokens = _tokens(" ".join(block))
    matched = block_tokens & query_tokens
    numeric_matches = sum(token.isdigit() for token in matched)
    return len(matched), numeric_matches, -sum(len(part) for part in block)


def _token_pattern(token: str) -> re.Pattern[str]:
    if token == "a1c":
        return re.compile(r"\b(?:a1c|alc)\b", re.I)
    if token == "replace":
        return re.compile(r"\breplac(?:e|ed|ing|ement)\b", re.I)
    if token.isdigit():
        return re.compile(rf"(?<!\d){re.escape(token)}(?!\d)", re.I)
    return re.compile(rf"\b{re.escape(token)}\b", re.I)


def _find_question_anchor(
    question_tokens: set[str],
    source_text: str,
) -> re.Match[str] | None:
    candidates: list[tuple[int, int, int, re.Match[str]]] = []
    for token in question_tokens - ANCHOR_WORDS:
        if not token.isdigit() and len(token) < 3:
            continue
        matches = list(_token_pattern(token).finditer(source_text))
        if not matches:
            continue
        priority = 1 if token.isdigit() else 0
        candidates.append((1 if len(matches) == 1 else 0, priority, len(token), matches[0]))
    return max(candidates, default=None, key=lambda item: item[:3])[3] if candidates else None


def _nearest_limit_statement(
    source_text: str,
    anchor: re.Match[str],
) -> str | None:
    matches = list(LIMIT_STATEMENT_RE.finditer(source_text))
    if not matches:
        return None

    def distance(match: re.Match[str]) -> tuple[int, int]:
        if match.start() <= anchor.start() <= match.end():
            return 0, match.end() - match.start()
        if match.start() >= anchor.end():
            return match.start() - anchor.end(), match.end() - match.start()
        return 10_000 + anchor.start() - match.end(), match.end() - match.start()

    return min(matches, key=distance).group(0).strip()


def _frequency_statement(
    source_text: str,
    anchor: re.Match[str],
) -> str | None:
    nearby_end = min(len(source_text), anchor.end() + 180)
    frequency = FREQUENCY_RE.search(source_text, anchor.start(), nearby_end)
    if frequency is None:
        return None
    return source_text[anchor.start() : frequency.end()].strip()


def _conditional_statement(
    source_text: str,
    anchor: re.Match[str],
) -> str:
    window_start = max(0, anchor.start() - 180)
    condition_matches = list(
        CONDITION_START_RE.finditer(source_text, window_start, anchor.start() + 1)
    )
    if condition_matches:
        start = condition_matches[-1].start()
    else:
        previous_boundary = max(
            source_text.rfind(".", window_start, anchor.start()),
            source_text.rfind("?", window_start, anchor.start()),
            source_text.rfind("!", window_start, anchor.start()),
        )
        start = previous_boundary + 1 if previous_boundary >= 0 else window_start

    end_match = re.search(r"[.!?¢•]", source_text[anchor.end() : anchor.end() + 320])
    end = (
        anchor.end() + end_match.end()
        if end_match is not None
        else min(len(source_text), anchor.end() + 320)
    )
    return source_text[start:end].strip()


def _question_anchored_statement(question: str, source_text: str) -> str | None:
    question_tokens = _question_tokens(question)
    anchor = _find_question_anchor(question_tokens, source_text)
    if anchor is None:
        return None

    if "limit" in question_tokens:
        return _nearest_limit_statement(source_text, anchor)
    if "frequency" in question_tokens or "checked" in question_tokens:
        frequency = _frequency_statement(source_text, anchor)
        if frequency is not None:
            return frequency
    return _conditional_statement(source_text, anchor)


def _best_unnumbered_statement(question: str, source_text: str) -> list[str]:
    flattened_source = re.sub(r"\s+", " ", source_text).strip()
    anchored = _question_anchored_statement(question, flattened_source)
    if anchored:
        return [anchored]
    statements = [
        statement.strip()
        for statement in STATEMENT_SPLIT_RE.split(flattened_source)
        if statement.strip()
    ]
    if not statements:
        raise ValueError("The selected source passage is empty.")

    candidates: list[list[str]] = []
    for width in range(1, min(3, len(statements)) + 1):
        candidates.extend(
            statements[index : index + width]
            for index in range(len(statements) - width + 1)
        )
    query_tokens = _question_tokens(question)
    return max(candidates, key=lambda block: _block_score(block, query_tokens))


def select_answer_excerpt(question: str, source_text: str) -> str:
    """Select one exact, question-relevant block without rewriting source details."""
    lines = [line.strip() for line in source_text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("The selected source passage is empty.")

    blocks = _inline_numbered_blocks(source_text) or _numbered_blocks(lines)
    numbered = {number: block for number, block in blocks if number is not None}

    if SUMMARY_RE.search(question):
        selected = (
            [line for number, block in blocks if number is not None for line in block]
            if numbered
            else lines
        )
    else:
        after_step = AFTER_STEP_RE.search(question)
        explicit_step = EXPLICIT_STEP_RE.search(question)
        requested_number: int | None = None
        if after_step:
            requested_number = int(after_step.group(1)) + 1
        elif explicit_step:
            requested_number = int(explicit_step.group(1))

        if requested_number in numbered:
            selected = numbered[requested_number]
        elif not numbered:
            selected = _best_unnumbered_statement(question, source_text)
        else:
            query_tokens = _question_tokens(question)
            candidates = [block for number, block in blocks if number is not None]
            selected = max(
                candidates,
                key=lambda block: _block_score(block, query_tokens),
            )

    return " ".join(selected)


def _readable_source_wording(excerpt: str) -> str:
    """Remove OCR layout marks while leaving all source words and numbers unchanged."""
    cleaned = re.sub(
        r"(?:[\[\]|_=*]+\s*)+[A-Za-z]?(?:\s*[\[\]|_=*]+)+",
        " ",
        excerpt,
    )
    cleaned = re.sub(r"[\[\]{}*_|=¢•]+", " ", cleaned)
    cleaned = re.sub(
        r"\bIf(?=(?:no|skin|resident|wound)\b)",
        "If ",
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip(" -")

    sentence_ends = list(re.finditer(r"[.!?]", cleaned))
    if sentence_ends:
        last_end = sentence_ends[-1].end()
        trailing = cleaned[last_end:].strip()
        substantive_tail = [
            token
            for token in re.findall(r"[A-Za-z]+", trailing)
            if len(token) >= 4
        ]
        if trailing and len(substantive_tail) < 2:
            cleaned = cleaned[:last_end]
    return cleaned


def compose_grounded_answer(
    question: str,
    results: Sequence[SearchResult],
) -> GroundedAnswer:
    """Create a direct answer from one reviewed result and no outside knowledge."""
    if not results:
        raise ValueError("At least one reviewed source result is required.")

    question_tokens = _question_tokens(question)
    ranked_excerpts = [
        (result, select_answer_excerpt(question, result.text), index)
        for index, result in enumerate(results)
    ]
    source, excerpt, _ = max(
        ranked_excerpts,
        key=lambda item: (*_block_score([item[1]], question_tokens)[:2], -item[2]),
    )
    answer = (
        f"According to the reviewed source on page {source.page_number}: "
        f"{_readable_source_wording(excerpt)}"
    )
    return GroundedAnswer(text=answer, exact_excerpt=excerpt, source=source)
