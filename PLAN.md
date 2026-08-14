# Implementation plan

## Current task: Build an accurate, scan-checked source draft from the original PDF

- [x] Inspect the original PDF page images and identify which current source pages are unchanged OCR.
- [x] Replace page-one OCR with a structured, scan-checked review draft while preserving raw OCR and the original PDF.
- [x] Remove unchanged-OCR pages from the eligible source corpus until a person confirms the corrected draft against the scan.
- [x] Make the source-review state clear in the Streamlit UI, then validate indexing and retrieval with synthetic tests and localhost.

## Current task: Re-index reviewed source evidence from the original PDF

- [x] Compare the original page scan, raw OCR, reviewed text, and indexed evidence for the reported failures.
- [x] Repair page-one reviewed source handling so each instruction remains an intact, inspectable evidence unit.
- [x] Rebuild the approved source index from verified reviewed text and validate target questions against it.
- [x] Add synthetic regressions for adjacent numeric instructions and run the full test suite.

## Current task: Offline citation-aware RAG rebuild

- [x] Add atomic reviewed evidence units with source-character spans and stable citation identifiers.
- [x] Retrieve a broad BM25/vector candidate set, then rerank it with a local cross-encoder.
- [x] Generate structured claim-level citations and verify every claim locally against its cited text.
- [x] Detect conflicting evidence and retain deterministic source-only fallbacks.
- [x] Redesign answer rendering around inline citations, exact evidence cards, and provenance.
- [x] Add synthetic retrieval/citation/conflict evaluations, install the local reranker, and verify localhost.

## Current task: Persistent hybrid BM25 and vector retrieval

- [x] Add a local SQLite vector-index table keyed to reviewed chunks and embedding model.
- [x] Lazily backfill and reuse embeddings instead of recomputing the document corpus per question.
- [x] Make BM25 exact-term retrieval the primary ranker and cosine similarity the secondary recall signal.
- [x] Fuse lexical and semantic ranks without weakening document/page eligibility gates.
- [x] Repair numeric-condition and replacement-action excerpt selection.
- [x] Add synthetic retrieval, vector-persistence, invalidation, and reported-question regressions.
- [x] Re-run the full suite and verify the seven reported questions on localhost.

## Current task: Demo-ready hybrid local AI

- [x] Add an offline Ollama provider with explicit health/model detection and timeouts.
- [x] Use local AI only on reviewed, approved, active source evidence.
- [x] Validate AI answers for numeric fidelity and source grounding before display.
- [x] Preserve the deterministic answer path as an automatic fallback.
- [x] Add clear hybrid-readiness and per-answer provenance status.
- [x] Add synthetic tests for AI success, rejection, and unavailable-model fallback.
- [x] Add reproducible Windows model setup and license notes.
- [x] Re-run the full suite and verify the localhost workflow.

## Current task: Retrieval reranking and OCR sentence repair

- [x] Join OCR line wraps before splitting answer statements.
- [x] Link named source items with adjacent limit/action statements.
- [x] Rerank retrieved pages using meaningful normalized question terms.
- [x] Handle common OCR variants such as `A1c` versus `Alc`.
- [x] Remove unrelated "other matches" from direct answers.
- [x] Add regressions for the three reported questions and verify them on localhost.

## Current task: Concise answers for long OCR lines

- [x] Split unnumbered OCR passages into answer-sized statements.
- [x] Rank single statements and adjacent statement pairs against the question.
- [x] Prevent unrelated source sections from appearing in the direct answer.
- [x] Add a synthetic regression test for a maximum-value question on one long OCR line.
- [x] Re-run the reported localhost question and verify the concise page citation.

## Current task: No-API source-grounded chat answers

- [x] Replace the result-only question form with a conversational question-and-answer UI.
- [x] Compose a direct answer from one reviewed source passage without adding outside facts.
- [x] Keep exact page citations, reviewed text, scan access, and original-PDF access with every answer.
- [x] Add synthetic regression tests for answer selection, conditional-step context, and numeric fidelity.
- [x] Verify the updated localhost workflow and document the no-API limitation.

## Current task: Password-free single source workflow

- [x] Remove the Admin tab, password prompt, and passcode configuration.
- [x] Combine upload, page review, source enablement, and questions on one page.
- [x] Keep explicit scan review and source-current confirmation in one action.
- [x] Preserve exact source passages, page scans, and original-PDF access in results.
- [x] Add workflow regression tests and verify the simplified localhost experience.

## Current task: Functional source-pull workflow redesign

- [x] Reproduce and diagnose the current localhost failure using live state and logs.
- [x] Make local source availability and OCR prerequisites obvious before import/search.
- [x] Simplify the search experience around verbatim source matches and page provenance.
- [x] Preserve human verification plus approved/active document gates.
- [x] Add synthetic tests for any retrieval or intake fixes.
- [x] Verify the complete workflow on localhost with an isolated synthetic source.

## Current task: Windows demo readiness and search diagnostics

- [x] Allow verification and nurse-search inclusion in one page-review form submission.
- [x] Add a repository readiness report with exact non-sensitive blockers.
- [x] Distinguish an empty eligible corpus from a lexical no-match result.
- [x] Add synthetic regression tests for indexing, document gates, readiness, and search status.
- [x] Add Windows setup/run scripts and document the local import workflow.
- [x] Run the complete test suite and verify the Streamlit health endpoint.

## Completed in this starter

- [x] Local project structure and configuration.
- [x] SQLite schema with document, page, chunk, FTS, and query-event tables.
- [x] Immutable PDF copy and SHA-256 duplicate detection.
- [x] Page rendering with PyMuPDF.
- [x] Local Tesseract OCR.
- [x] Raw OCR and reviewed-text separation.
- [x] Human page-verification gate.
- [x] Page-level classification and search inclusion.
- [x] Document approval/lifecycle gate.
- [x] Exact SQLite FTS5 search.
- [x] Source scan display and PDF download.
- [x] Basic likely-PHI warning.
- [x] Synthetic tests for the core gates.

## Next

- [ ] Run an end-to-end UI review of the uploaded 13-page sample.
- [ ] Add `source_items` to group related pages into logical artifacts.
- [ ] Add bulk progress/status views for OCR review.
- [ ] Add document-quality checks for missing dates, duplicate active versions, and pending high-risk pages.
- [ ] Add a reviewed retrieval-evaluation set.
- [ ] Add deterministic conflict candidates.
- [ ] Design grounded summarization only after retrieval and citation validation are mature.
