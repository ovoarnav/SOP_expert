# Archbold Policy Navigator Prototype — Codex Instructions

## Mission

Continue building a local-first, read-only policy-navigation prototype for Archbold Living source documents. The current vertical slice imports scanned PDFs, runs local OCR, requires human page-level verification, and returns exact source excerpts.

The product must always be labeled:

> **PROTOTYPE — NOT FOR CLINICAL USE**

It is not an EHR, MAR, prescribing system, resident-specific clinical decision tool, or compliance certification system.

## What the uploaded sample proved

The initial 13-page source is an image-only scan containing multiple logical artifacts in one PDF: a standing-order summary, physician-order templates, PRN notification forms, treatment protocols, dense clinical tables, and a blank fax form.

Therefore:

- OCR is required; a “reject all scans” design is not acceptable for this prototype.
- OCR text is untrusted draft data until a human compares it with the page image.
- Search eligibility must exist at page or logical-source-item level, not only at PDF level.
- Printed facility scope, form number, and date may vary by page.
- Blank administrative forms must not be blended into clinical-policy answers.
- Dense tables require row-by-row or cell-by-cell human verification.

Read `docs/UPLOADED_SAMPLE_ASSESSMENT.md` before changing the data model or intake flow.

## Current scope

Version 0.1 implements only:

1. Local PDF import.
2. Immutable managed copy of the source.
3. Page rendering with PyMuPDF.
4. Local English OCR with Tesseract.
5. Side-by-side scan and OCR review.
6. Page-level metadata and search inclusion.
7. Document approval/lifecycle gating.
8. SQLite FTS5 search over human-verified text.
9. Exact source excerpts, page images, and original-PDF download.
10. A lightweight warning for likely PHI in search text.

Do not add a generative clinical answer until the extractive flow and its tests are stable.

## Non-negotiable clinical and privacy boundaries

The application must never:

- Diagnose a resident.
- Recommend whether a nurse should administer a medication or treatment to a resident.
- Claim that a facility policy, protocol, or order template proves an active resident-specific order.
- Replace verification of current orders, MAR, allergies, contraindications, care plan, provider instructions, or resident condition.
- State that a policy or practice is compliant, legal, safe, approved, or “up to code.”
- Browse the public internet at runtime.
- Write to an EHR, MAR, order system, pharmacy system, or clinical record.
- Use unreviewed OCR text in nurse search.
- Use pages or documents that are draft, rejected, inactive, superseded, expired, quarantined, or unapproved.
- silently combine content from unrelated forms, pages, facilities, or document versions.
- Store query text unless `LOG_QUERY_TEXT=true`.
- Invite users to enter resident identifiers.

Do not import completed resident forms or files containing PHI into the prototype.

## Search eligibility rule

A chunk is eligible for nurse search only when all conditions are true:

```text
document.approval_status == approved
AND document.lifecycle_status == active
AND document.parse_status IN (ocr_complete, partial)
AND page.verification_status == verified
AND page.searchable == true
```

Do not weaken this rule to make demos or tests easier.

## OCR policy

OCR is a transcription aid, not a source of truth.

- Render each page and retain the image.
- Preserve the raw OCR output.
- Store the reviewed transcription separately.
- Never overwrite raw OCR with corrected text.
- Default every page to `pending` and `searchable=false`.
- Require an explicit human attestation before setting `verified`.
- Require manual checking of medication names, numbers, units, routes, frequencies, intervals, thresholds, exceptions, and notification language.
- Warn whenever a page contains digits or OCR confidence is low.
- Keep table pages blocked until their layout has been checked against the scan.

A future table-review interface may store rows/cells structurally, but do not infer table relationships from OCR alone.

## Source classification

Keep these concepts distinct:

- `standing_order_summary`
- `protocol`
- `order_set_template`
- `implementation_form`
- `treatment_order`
- `administrative_form`
- `reference`
- `exclude_from_clinical_search`

A page can be retained for audit or form lookup while remaining excluded from nurse search.

## Architecture

Keep a single Python repository with:

- Streamlit UI.
- SQLite metadata and FTS5.
- PyMuPDF for rendering.
- Tesseract via `pytesseract` for local OCR.
- Small, inspectable service modules under `src/archbold_nav/`.
- Pytest with synthetic, non-clinical fixtures.

Do not introduce LangChain, LlamaIndex, a hosted vector database, Kubernetes, or a separate frontend framework for this prototype.

## Current repository map

```text
.
├── AGENTS.md
├── PLAN.md
├── README.md
├── app.py
├── pyproject.toml
├── .env.example
├── docs/
│   └── UPLOADED_SAMPLE_ASSESSMENT.md
├── scripts/
│   ├── ingest_file.py
│   └── reset_db.py
├── src/archbold_nav/
│   ├── config.py
│   ├── db.py
│   ├── ocr.py
│   ├── ingest.py
│   ├── chunking.py
│   ├── repository.py
│   ├── safety.py
│   └── search.py
└── tests/
```

## Data model direction

The present schema has `documents`, `pages`, `chunks`, `chunks_fts`, and `query_events`.

The next meaningful schema enhancement is a `source_items` table that groups one or more pages into a logical protocol, form, order template, or policy section. Do this only with a migration and tests.

Suggested fields:

- `id`
- `document_id`
- `title`
- `page_start`
- `page_end`
- `facility_scope`
- `source_type`
- `form_number`
- `printed_date`
- `approval_status`
- `lifecycle_status`
- `searchable`
- `verification_status`
- `notes`

Until that exists, page-level metadata is authoritative.

## Retrieval behavior

Version 0.1 is extractive.

- Query SQLite FTS5 only over eligible chunks.
- Return exact reviewed text, not an invented answer.
- Show document title, file name, page, page/section type, facility, form number, and printed date when available.
- Show the scan image beside or under the text.
- Make the original PDF inspectable.
- Return an explicit not-found state when no eligible evidence exists.
- Do not show a model-generated confidence percentage.

## Future grounded summary gate

Do not implement AI summarization until all of the following exist:

1. Retrieval evaluation with a reviewed question set.
2. Exact chunk citations.
3. Verbatim quote validation.
4. Numeric and unit validation.
5. Conflict detection.
6. A deterministic extractive fallback.
7. PHI preflight before any remote call.
8. A rule that only retrieved excerpts—not full files—may be sent remotely.

When added, the model may summarize supplied excerpts only. It may not fill gaps from general medical or legal knowledge.

## Compliance-review boundary

Internal source documents cannot prove that Archbold is “up to code.”

A later admin-only review feature may identify:

- missing metadata;
- duplicate active versions;
- expired review dates;
- unreadable pages;
- conflicting internal text;
- missing or malformed citations;
- potential gaps against separately curated regulatory sources.

Allowed finding labels:

- `document_quality_finding`
- `potential_gap`
- `potential_conflict`
- `needs_human_review`
- `not_enough_evidence`

Never output `compliant`, `legal`, `approved`, or `meets code` as an automated conclusion.

## Testing requirements

Use only synthetic, non-clinical source text in committed tests.

Maintain tests for:

- duplicate-file rejection;
- immutable managed copies;
- pending OCR exclusion;
- page-verification gating;
- document-status gating;
- FTS retrieval after review;
- removal from search after a page is unverified or excluded;
- removal from search after a document becomes inactive;
- likely-PHI detection;
- query-text logging defaulting to off;
- source page numbers remaining correct;
- errors when Tesseract is unavailable;
- no data leakage into Git-tracked paths.

For table support, add synthetic tables with clearly fake labels and numbers. Do not place real medication orders in tests.

## Coding standards

- Use type hints.
- Keep business logic out of Streamlit callbacks.
- Use parameterized SQL.
- Preserve source text exactly after human review.
- Never swallow exceptions silently.
- Avoid logging document text, OCR text, prompts, or resident information.
- Keep dependencies minimal.
- Prefer transparent code over abstraction-heavy frameworks.
- Keep the app runnable after every change.

## Immediate implementation order

1. Run the existing tests and fix defects without weakening guardrails.
2. Test import against the supplied scanned PDF locally.
3. Add document deletion/quarantine only as an admin operation with audit history; never silently delete originals.
4. Add `source_items` grouping for multi-page logical artifacts.
5. Add an OCR review completion dashboard.
6. Add exportable document-quality findings.
7. Build retrieval evals.
8. Only then consider grounded summarization.

## Codex working instructions

1. Read this file, `README.md`, `PLAN.md`, and `docs/UPLOADED_SAMPLE_ASSESSMENT.md` first.
2. State the task you are implementing in `PLAN.md` before editing code.
3. Make small vertical changes and run relevant tests after each one.
4. Do not commit or paste real Archbold PDFs, OCR text, page images, databases, or query logs.
5. Do not use real clinical content as test fixtures.
6. Do not bypass human verification for convenience.
7. At completion, report exact commands run, test results, and known limitations.
