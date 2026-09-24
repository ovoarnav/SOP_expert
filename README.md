# Source Document Chat Prototype

> **PROTOTYPE — NOT FOR CLINICAL USE.** This application retrieves information from selected facility documents. It does not diagnose, prescribe, place orders, determine regulatory compliance, or replace verification in a resident's chart, current orders, MAR, allergies, contraindications, care plan, or provider instructions.

This repository is a local-first vertical slice for scanned long-term-care source documents. It was designed around an image-only, multi-document PDF bundle rather than a clean text PDF.

## What version 0.1 does

- Imports a PDF without modifying the original.
- Copies it into immutable local managed storage.
- Renders every page and runs local English OCR with Tesseract.
- Stores the scan image beside the OCR transcription.
- Requires a human to verify each page before it can enter search.
- Performs a one-time safety cleanup of legacy pages that had been marked verified with unchanged raw OCR; those pages remain in the local library but return to pending review instead of answering questions.
- Lets the local user review and classify pages, including forms that should be excluded from source search.
- Searches only pages that are human-verified and explicitly searchable inside an approved, active document.
- Uses SQLite FTS5 BM25 as the primary exact-term ranker.
- Splits reviewed pages into atomic, source-span-preserving evidence units with stable citation identifiers; recognised scan-table rows remain intact so a medication, limit, condition, and action cannot be retrieved as separate fragments.
- Stores a persistent local embedding vector for each reviewed unit and uses cosine similarity as a secondary recall signal.
- Uses a local cross-encoder to rerank hybrid candidates before answer generation.
- Uses an offline answer model to return short claims with an explicit citation for each claim; deterministic validation always runs, with a second local claim check for complex or ambiguous answers.
- Rejects AI output that adds numbers, changes units, omits source actions, cites unavailable evidence, weakly overlaps the evidence, or gives advice.
- Automatically falls back to deterministic exact-source wording when local AI is unavailable or rejected.
- Shows the first cited source page immediately, while loading full scans and the original PDF only when requested.
- Stores query text only when `LOG_QUERY_TEXT=true`.

Version 0.1 does **not** call a hosted language model, add facts from general medical knowledge, connect to an EHR/MAR, process resident data, or decide that a document is "up to code." Local AI is optional; the deterministic source-only path remains fully functional without it.

## Why the human OCR review gate matters

The sample source is a scan containing medication names, doses, routes, frequencies, timing, thresholds, and tables. OCR can confuse characters or move values between columns. A page is therefore blocked from retrieval until someone compares the transcription against the scan and explicitly verifies it.

## Requirements

- Python 3.11 or later.
- Tesseract OCR installed and available on `PATH`.
- Ollama for the optional hybrid local-AI demo.
- A modern browser.

Install Tesseract:

- macOS with Homebrew: `brew install tesseract`
- Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
- Windows PowerShell: `winget install --id tesseract-ocr.tesseract --exact`

The app also detects the standard Windows installation directories. If Tesseract is installed elsewhere, set `TESSERACT_CMD` in `.env` to the full path of `tesseract.exe`.

## Setup

Using `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install -e '.[dev]'
cp .env.example .env
```

Using standard Python:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
cp .env.example .env
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

Activation is optional. For the straightforward Windows setup, run these commands from the repository root:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_windows.ps1
.\scripts\run_windows.ps1
```

The setup script creates `.venv` when needed, installs the development dependencies, preserves any existing `.env`, and reports whether Tesseract is available.

For the offline hybrid-AI demo, install and start Ollama, then run:

```powershell
.\scripts\setup_local_ai.ps1
.\scripts\setup_citation_rag.ps1
```

The first script downloads the compact embedding model; the second installs the
CPU inference runtime and downloads the approximately 2.3 GB local cross-encoder
reranker. The default answer model is the locally installed `qwen3:30b-instruct`.
No app request downloads models or calls a hosted API. See
[docs/LOCAL_AI_DEMO.md](docs/LOCAL_AI_DEMO.md) for the grounding contract and
model-license details.

## Run

```bash
streamlit run app.py
```

The app binds through Streamlit's normal local behavior. Keep it on a trusted laptop and do not expose it publicly.

## Workflow

Files uploaded in ChatGPT are separate from this local application. Upload the source PDF directly in the app before reviewing or searching it.

1. Upload a scanned PDF.
2. Wait for local OCR to complete.
3. Review every page that should be searchable:
   - compare the text with the page image;
   - correct OCR errors;
   - confirm that the document is approved and current;
   - enable the reviewed page for questions.

For a legacy local library, this may mean that only the pages with a corrected
review transcription are initially searchable. The original PDF, scan, and raw
OCR remain available for every pending page; they are not deleted.
4. Ask a topic-only question that contains no resident identifiers.

The source chat remains disabled until at least one reviewed page is indexed inside an approved, active document. Reviewed text is split into citation-ready source spans. Hybrid mode retrieves only those eligible spans: BM25 supplies the dominant exact-term rank, cosine similarity supplies a secondary paraphrase signal, weighted reciprocal-rank fusion creates a bounded candidate set, and the local cross-encoder reranks that set. The answer model returns claims with citation IDs, then deterministic numeric/unit/action validation approves each claim; complex or ambiguous claims also receive a second local check. Unsupported output, model errors, insufficient evidence, and conflicting numeric evidence automatically produce a source-only or needs-review state. The first citation page is shown inline, with the exact reviewed wording, full scan, and original PDF available for inspection.

## Command-line import

After installation:

```bash
python scripts/ingest_file.py /path/to/source.pdf --title "Descriptive title"
```

The file will still need page-by-page review in the app.

## Tests

```bash
pytest
```

The tests use synthetic, non-clinical text. Real Archbold source files and extracted OCR text are excluded from version control.

## Data handling

All imported PDFs, rendered page images, OCR text, and the SQLite database stay under `data/` and are ignored by Git. At runtime the application uses local Tesseract and, when configured, local Ollama models at `127.0.0.1`. It does not send document text to a hosted model or public internet service.

Do not import completed forms, resident charts, MAR exports, or files containing resident identifiers into this prototype.

## Known limitations

- OCR is English-only and may be wrong, especially in tables and low-quality scans.
- Page review is manual and should be performed by qualified Archbold personnel.
- Retrieval quality still depends on the reviewed OCR text and human-checked evidence boundaries. Pages indexed from unchanged OCR are flagged in the review screen and must be checked against the scan before re-confirmation.
- The 30B local generation path is intentionally slower when Ollama cannot offload to an accelerator; the UI reports the active offload state and source-only fallback remains available.
- Ollama and model weights are installed separately and are not committed to this repository.
- The whole PDF is one managed source bundle even though page-level metadata can represent different logical documents.
- Conflict detection currently covers competing single numeric values; broader textual conflicts still require human review.
- The PHI pattern check is intentionally basic and not a privacy guarantee.
- The app does not validate policy approval, clinical correctness, legal status, or regulatory compliance.
- This password-free local prototype has no enterprise identity, access control, audit, retention, or multi-tenant isolation.

## Recommended next milestone

Add a `source_items` grouping layer so multiple pages can be treated as one logical form or protocol, followed by deterministic conflict checks, a reviewed retrieval-evaluation set, and production identity/audit controls before any deployment beyond a local demo.
