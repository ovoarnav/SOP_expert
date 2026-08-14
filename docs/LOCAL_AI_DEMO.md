# Local AI demo setup

The demo uses local models only. Source text is sent only to the local Ollama
service at `127.0.0.1` and the locally cached CPU reranker; the application does
not call a hosted model.

- `archbold-bge-small:latest` supplies a secondary cosine-similarity recall signal. It is built from
  BAAI BGE Small English v1.5, distributed under the MIT license.
- `qwen3:30b-instruct` writes claim-cited answers and independently verifies each
  claim against its exact cited evidence.
- `BAAI/bge-reranker-v2-m3` is downloaded once to `models/bge-reranker-v2-m3`
  and reranks only the retrieved candidate evidence on CPU.

On a new Windows demo laptop:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_windows.ps1
.\scripts\setup_local_ai.ps1
.\scripts\setup_citation_rag.ps1
.\scripts\run_windows.ps1
```

The 30B answer and verifier model can take one to several minutes on a CPU-only
laptop. The app visibly reports retrieval, reranking, generation, and verification.
If Ollama or the reranker is unavailable, the app remains functional and uses its
deterministic source-only answer path.

## Grounding contract

1. Retrieval can inspect only human-reviewed pages in approved, active documents.
2. Reviewed source text is indexed as atomic spans with page and character-range metadata.
3. SQLite FTS5 BM25 is the primary exact-term ranker; cached local vectors are a
   lower-weight secondary signal combined through reciprocal-rank fusion; the local
   cross-encoder reranks the resulting candidate set.
4. The answer model receives only citation-labelled exact evidence and returns each
   claim with one or more of those citation IDs.
5. Added numbers, changed units, omitted actions, unsupported citations, weak source
   overlap, unsafe advice, empty output, malformed model responses, and failed local
   claim verification are rejected.
6. A rejected or unavailable AI answer falls back to exact source wording; competing
   numeric evidence is shown as a needs-review state rather than merged.
7. Every supported answer retains its claim citation, reviewed excerpt, page scan,
   and original PDF.

The models improve retrieval and phrasing; they do not validate policy approval,
clinical correctness, resident orders, allergies, contraindications, or legal and
regulatory compliance.
