from __future__ import annotations

import shutil
import sys
import time
import uuid
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import streamlit as st

# Allow `streamlit run app.py` without an editable install.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from archbold_nav.config import get_settings
from archbold_nav.citation_rag import (
    CitationAwareAnswer,
    answer_with_citations,
    source_only_citation_answer,
)
from archbold_nav.db import init_db
from archbold_nav.hybrid_retrieval import retrieve_hybrid_sources
from archbold_nav.ingest import (
    DuplicateDocumentError,
    UnsupportedDocumentError,
    ingest_pdf,
)
from archbold_nav.local_ai import (
    LocalAIAccelerationStatus,
    LocalAIError,
    LocalAIStatus,
    OllamaClient,
)
from archbold_nav.ocr import find_tesseract_executable
from archbold_nav.repository import (
    enable_page_for_search,
    get_document,
    get_page,
    get_source_library_summary,
    ensure_atomic_evidence_index,
    exclude_legacy_unchanged_ocr_pages,
    list_documents,
    list_pages,
)
from archbold_nav.safety import detect_likely_phi
from archbold_nav.search import search_verified_sources_with_status
from archbold_nav.vector_store import get_vector_index_status
from archbold_nav.reranker import (
    LocalCrossEncoderReranker,
    RerankerError,
    get_reranker_status,
)


st.set_page_config(
    page_title="Source document search",
    page_icon=":material/find_in_page:",
    layout="wide",
)

settings = get_settings()
init_db(settings)
ensure_atomic_evidence_index(settings)
exclude_legacy_unchanged_ocr_pages(settings)


def make_local_ai_client() -> OllamaClient:
    return OllamaClient(
        base_url=settings.local_ai_url,
        answer_model=settings.local_ai_answer_model,
        embedding_model=settings.local_ai_embedding_model,
        timeout_seconds=settings.local_ai_timeout_seconds,
        min_similarity=settings.local_ai_min_similarity,
        draft_token_limit=settings.local_ai_draft_token_limit,
    )


@st.cache_resource(show_spinner=False)
def load_local_reranker(
    model_dir: str,
    device: str,
    batch_size: int,
) -> LocalCrossEncoderReranker:
    """Keep the local cross-encoder loaded once per Streamlit server process."""
    return LocalCrossEncoderReranker(
        model_dir=Path(model_dir),
        device=device,
        batch_size=batch_size,
    )


@st.cache_data(ttl=10, show_spinner=False)
def get_local_ai_status(
    base_url: str,
    answer_model: str,
    embedding_model: str,
) -> LocalAIStatus:
    return OllamaClient(
        base_url=base_url,
        answer_model=answer_model,
        embedding_model=embedding_model,
        timeout_seconds=3,
    ).status()


@st.cache_data(ttl=5, show_spinner=False)
def get_local_ai_acceleration_status(
    base_url: str,
    answer_model: str,
    embedding_model: str,
) -> LocalAIAccelerationStatus:
    return OllamaClient(
        base_url=base_url,
        answer_model=answer_model,
        embedding_model=embedding_model,
        timeout_seconds=3,
    ).acceleration_status()


@st.cache_data(max_entries=32, show_spinner=False)
def load_page_preview(image_path: str, modified_ns: int) -> bytes:
    """Create a bounded in-memory preview so scans need not be sent at full size."""
    from PIL import Image

    del modified_ns  # Streamlit cache key invalidates when the page scan changes.
    with Image.open(image_path) as image:
        preview = image.convert("RGB")
        preview.thumbnail((1100, 1400))
        buffer = BytesIO()
        preview.save(buffer, format="JPEG", quality=85, optimize=True)
    return buffer.getvalue()


def safe_upload_name(name: str) -> str:
    cleaned = Path(name).name.replace("\x00", "")
    return cleaned or "uploaded.pdf"


def render_banner() -> None:
    st.warning(
        "PROTOTYPE — NOT FOR CLINICAL USE. This app retrieves reviewed passages from "
        "uploaded source documents. It does not diagnose, prescribe, or confirm a "
        "resident-specific order.",
        icon=":material/warning:",
    )


def render_upload() -> None:
    with st.container(border=True):
        st.subheader("1. Upload a source document")
        st.write(
            "Upload a scanned PDF from this laptop. The file and OCR stay local to this app."
        )

        tesseract = find_tesseract_executable()
        if tesseract is None:
            st.error(
                "Scanned-document OCR is not installed yet. Run the command below in an "
                "Administrator PowerShell, then refresh this page.",
                icon=":material/build:",
            )
            st.code(
                "winget install --id tesseract-ocr.tesseract --exact",
                language="powershell",
            )
        else:
            st.success("Visual-document OCR is ready.", icon=":material/check_circle:")

        uploaded = st.file_uploader("Upload PDF", type=["pdf"])
        display_title = st.text_input(
            "Document name",
            placeholder="Optional — the filename will be used if left blank",
        )
        can_import = uploaded is not None and tesseract is not None

        if st.button(
            "Read this document",
            type="primary",
            disabled=not can_import,
            icon=":material/document_scanner:",
        ):
            assert uploaded is not None
            if uploaded.size > settings.max_upload_mb * 1024 * 1024:
                st.error(f"File exceeds the {settings.max_upload_mb} MB limit.")
                return

            upload_dir = settings.inbox_dir / str(uuid.uuid4())
            upload_dir.mkdir(parents=True, exist_ok=False)
            temp_path = upload_dir / safe_upload_name(uploaded.name)
            temp_path.write_bytes(uploaded.getvalue())

            with st.status("Reading the visual document…", expanded=True) as status:
                progress_bar = st.progress(0.0)
                progress_text = st.empty()

                def progress(done: int, total: int, message: str) -> None:
                    progress_bar.progress(done / max(total, 1))
                    progress_text.caption(message)

                try:
                    document_id = ingest_pdf(
                        temp_path,
                        settings,
                        display_title=display_title.strip() or None,
                        progress=progress,
                    )
                except DuplicateDocumentError as exc:
                    status.update(label="Document already uploaded", state="complete")
                    st.warning(str(exc))
                except (UnsupportedDocumentError, RuntimeError, ValueError) as exc:
                    status.update(label="Document could not be read", state="error")
                    st.error(str(exc))
                except Exception as exc:
                    status.update(label="Document could not be read", state="error")
                    st.exception(exc)
                else:
                    status.update(label="Document is ready for review", state="complete")
                    st.session_state["selected_document_id"] = document_id
                    st.session_state.pop("source_chat_messages", None)
                    st.success("Upload complete. Review a page below to enable source search.")
                finally:
                    shutil.rmtree(upload_dir, ignore_errors=True)


def render_review() -> None:
    documents = list_documents(settings)
    if not documents:
        return

    with st.container(border=True):
        st.subheader("2. Review and enable source pages")
        st.write(
            "Choose a page, compare the extracted text with the scan, correct any errors, "
            "and enable that reviewed page for search."
        )

        labels = {doc["title"]: doc["id"] for doc in documents}
        selected_document_id = st.session_state.get("selected_document_id")
        default_index = 0
        if selected_document_id in labels.values():
            default_index = list(labels.values()).index(selected_document_id)
        selected_document_label = st.selectbox(
            "Document",
            list(labels),
            index=default_index,
        )
        document_id = labels[selected_document_label]
        st.session_state["selected_document_id"] = document_id

        document = get_document(settings, document_id)
        pages = list_pages(settings, document_id)
        if document is None or not pages:
            st.error("No readable pages are available for this document.")
            return

        page_labels = {
            f"Page {page['page_number']} — {page['verification_status']}": page["id"]
            for page in pages
        }
        selected_page_label = st.selectbox("Page", list(page_labels))
        page = get_page(settings, page_labels[selected_page_label])
        if page is None:
            return

        scan_column, review_column = st.columns([1, 1.1], gap="large")
        with scan_column:
            st.markdown(f"**Source scan — page {page['page_number']}**")
            st.image(page["image_path"], width="stretch")
            if page["warnings"]:
                st.warning(page["warnings"])

        with review_column:
            unchanged_ocr = (
                bool(page["raw_ocr_text"])
                and page["verified_text"].strip() == page["raw_ocr_text"].strip()
            )
            if page["verification_status"] == "verified" and unchanged_ocr:
                st.warning(
                    "This page is currently indexed from unchanged OCR. Compare it with the scan "
                    "and correct it before re-confirming; answers will cite this exact text.",
                    icon=":material/fact_check:",
                )
            with st.form(f"review-page-{page['id']}"):
                section_title = st.text_input(
                    "Page or section name",
                    value=page["section_title"] or f"Page {page['page_number']}",
                )
                verified_text = st.text_area(
                    "Reviewed source text (this exact wording will be cited)",
                    value=page["verified_text"] or page["raw_ocr_text"],
                    height=430,
                    help=(
                        "Edit this until it matches the scan. The app indexes complete source "
                        "rows and shows this exact wording beneath every citation."
                    ),
                )
                page_confirmed = st.checkbox(
                    "I checked this text against the scan",
                    value=(
                        page["verification_status"] == "verified" and not unchanged_ocr
                    ),
                )
                document_confirmed = st.checkbox(
                    "I confirm this document is approved and current for this source library",
                    value=(
                        document["approval_status"] == "approved"
                        and document["lifecycle_status"] == "active"
                    ),
                )
                submitted = st.form_submit_button(
                    "Enable this page for questions",
                    type="primary",
                    icon=":material/check_circle:",
                )

            if submitted:
                try:
                    enable_page_for_search(
                        settings,
                        page["id"],
                        verified_text=verified_text,
                        section_title=section_title,
                        page_review_confirmed=page_confirmed,
                        document_approval_confirmed=document_confirmed,
                    )
                except (KeyError, ValueError) as exc:
                    st.error(str(exc))
                else:
                    st.success("This reviewed page can now be used for source questions.")
                    st.session_state.pop("source_chat_messages", None)
                    st.rerun()


def render_source_evidence(result, exact_excerpt: str, *, key_prefix: str) -> None:
    metadata = [
        result.document_title,
        f"page {result.page_number}",
        result.facility_scope or result.facility,
    ]
    st.caption("Source: " + " · ".join(metadata))

    details = st.expander(
        "View exact source evidence",
        icon=":material/find_in_page:",
        key=f"source-evidence-{key_prefix}-{result.chunk_id}",
        on_change="rerun",
    )
    if details.open:
        with details:
            if result.matched_terms:
                st.caption("Matched source words: " + ", ".join(result.matched_terms))
            st.markdown("**Exact reviewed wording used for the answer**")
            st.code(exact_excerpt, language=None, wrap_lines=True)

            page = get_page(settings, result.page_id)
            if page and Path(page["image_path"]).exists():
                st.markdown("**Source scan**")
                st.image(
                    page["image_path"],
                    caption=f"{result.document_title} — page {result.page_number}",
                    width="stretch",
                )

            if result.managed_path.exists():
                with result.managed_path.open("rb") as handle:
                    st.download_button(
                        "Open original PDF",
                        data=handle.read(),
                        file_name=result.original_filename,
                        mime="application/pdf",
                        key=f"download-{key_prefix}-{result.chunk_id}",
                        icon=":material/download:",
                    )


def render_inline_page_preview(citation, *, key_prefix: str) -> None:
    """Show the first cited page immediately without eagerly loading full source assets."""
    result = citation.source
    page = get_page(settings, result.page_id)
    if page is None:
        return
    image_path = Path(page["image_path"])
    if not image_path.exists():
        return
    started = time.perf_counter()
    try:
        preview = load_page_preview(str(image_path), image_path.stat().st_mtime_ns)
    except (OSError, ValueError):
        return
    st.markdown(
        f"**Source page [{citation.citation_id}] — {result.document_title} · page {result.page_number}**"
    )
    st.image(
        preview,
        caption="Reviewed source scan preview. Open the source details below for the full scan and PDF.",
        width="stretch",
    )
    st.caption(f"Page preview ready in {(time.perf_counter() - started) * 1000:.0f} ms")


def render_answer_timing(answer: CitationAwareAnswer) -> None:
    timing = answer.timing
    parts = [f"retrieval {timing.retrieval_seconds:.1f}s"]
    if answer.reranker_used:
        parts.append(f"reranking {timing.reranking_seconds:.1f}s")
    parts.append(f"generation {timing.generation_seconds:.1f}s")
    if answer.model_verification_used:
        parts.append(f"local verification {timing.verification_seconds:.1f}s")
    else:
        parts.append("deterministic verification")
    st.caption("Timing: " + " · ".join(parts))


def render_citation_answer(answer: CitationAwareAnswer, *, key_prefix: str) -> None:
    if answer.status == "answered":
        for claim in answer.claims:
            citation_labels = " ".join(f"[{citation_id}]" for citation_id in claim.citation_ids)
            st.markdown(f"- {claim.text} {citation_labels}")
        retrieval_label = "BM25 + vector retrieval"
        if answer.reranker_used:
            retrieval_label += " + local reranking"
        verification_label = (
            "local verifier" if answer.model_verification_used else "deterministic verifier"
        )
        st.caption(
            f"Local AI answer · {retrieval_label} · {verification_label} · each claim validated against cited evidence"
        )
    elif answer.status == "conflicting_evidence":
        st.warning(
            "I found reviewed source passages with conflicting numeric values for this question. "
            "I did not combine them; compare the cited sources below.",
            icon=":material/compare_arrows:",
        )
    elif answer.status == "insufficient_evidence":
        st.info(
            "I could not find enough eligible reviewed source evidence to answer that question.",
            icon=":material/search_off:",
        )
    else:
        for claim in answer.claims:
            citation_labels = " ".join(f"[{citation_id}]" for citation_id in claim.citation_ids)
            st.markdown(f"{claim.text} {citation_labels}")
        st.caption("Source-only answer · deterministic safety fallback")

    if answer.fallback_reason and answer.status != "answered":
        st.caption("Why synthesis was withheld: " + answer.fallback_reason)

    render_answer_timing(answer)

    if answer.citations:
        render_inline_page_preview(answer.citations[0], key_prefix=key_prefix)

    if answer.citations:
        st.markdown("**Cited reviewed evidence**")
    for citation in answer.citations:
        result = citation.source
        with st.container(border=True):
            st.markdown(
                f"**[{citation.citation_id}] {result.document_title} · page {result.page_number}**"
            )
            st.code(citation.quote, language=None, wrap_lines=True)
            details = st.expander(
                f"Inspect source [{citation.citation_id}]",
                icon=":material/find_in_page:",
                key=f"citation-source-{key_prefix}-{citation.citation_id}-{result.chunk_id}",
                on_change="rerun",
            )
            if details.open:
                with details:
                    metadata = [
                        result.facility_scope or result.facility,
                        result.section_title or "Unclassified section",
                        f"reviewed characters {result.source_start}–{result.source_end}",
                    ]
                    st.caption(" · ".join(metadata))
                    page = get_page(settings, result.page_id)
                    if page and Path(page["image_path"]).exists():
                        st.image(
                            page["image_path"],
                            caption=f"{result.document_title} — page {result.page_number}",
                            width="stretch",
                        )
                    if result.managed_path.exists():
                        with result.managed_path.open("rb") as handle:
                            st.download_button(
                                "Open original PDF",
                                data=handle.read(),
                                file_name=result.original_filename,
                                mime="application/pdf",
                                key=f"download-{key_prefix}-{citation.citation_id}-{result.chunk_id}",
                                icon=":material/download:",
                            )

    explanation = st.expander(
        "Why this answer",
        icon=":material/psychology:",
        key=f"why-answer-{key_prefix}",
        on_change="rerun",
    )
    if explanation.open:
        with explanation:
            st.write(
                "The app searched only approved, active, human-reviewed source passages, "
                "combined exact-term and vector retrieval, and then checked each displayed claim "
                "against its cited wording. It does not use general medical knowledge."
            )
            st.caption(
                "Reranker used: " + ("yes" if answer.reranker_used else "no — hybrid retrieval fallback")
            )


def render_chat_message(message: dict, index: int) -> None:
    with st.chat_message(message["role"]):
        citation_answer = message.get("citation_answer")
        if citation_answer is not None:
            render_citation_answer(citation_answer, key_prefix=f"message-{index}")
            return
        st.write(message["content"])
        if message.get("mode") == "local_ai":
            retrieval_label = {
                "hybrid": "BM25 + vector retrieval",
                "bm25": "BM25 retrieval",
                "semantic": "semantic retrieval",
                "lexical": "lexical retrieval",
            }.get(message.get("retrieval_mode"), "reviewed-source retrieval")
            st.caption(
                f"Local AI answer · {retrieval_label} · validated against exact evidence"
            )
        elif message.get("mode") == "source_only":
            st.caption("Source-only answer · deterministic safety fallback")
        result = message.get("result")
        if result is not None:
            render_source_evidence(
                result,
                message["exact_excerpt"],
                key_prefix=f"message-{index}",
            )


def render_questions() -> None:
    library = get_source_library_summary(settings)
    with st.container(border=True):
        st.subheader("3. Chat with the reviewed source")
        if not library.ready_for_search:
            st.info(
                "Upload a document and enable at least one reviewed page to start asking questions."
            )
            return

        st.caption(
            f"Searching {library.eligible_page_count} reviewed page(s) across "
            f"{library.imported_document_count} uploaded document(s)."
        )
        st.warning(
            "A source document does not confirm an active resident-specific order. Verify "
            "current orders, MAR, allergies, contraindications, and provider instructions.",
            icon=":material/warning:",
        )

        local_ai_status = get_local_ai_status(
            settings.local_ai_url,
            settings.local_ai_answer_model,
            settings.local_ai_embedding_model,
        )
        use_local_ai = local_ai_status.ready
        reranker_status = get_reranker_status(settings)
        if local_ai_status.ready:
            vector_status = get_vector_index_status(
                settings,
                model_name=settings.local_ai_embedding_model,
            )
            st.success(
                "Citation-aware local AI is ready. Answers are generated and checked "
                "on this laptop, with automatic source-only fallback.",
                icon=":material/memory:",
            )
            st.caption(
                f"BM25 primary · {vector_status.indexed_chunk_count} cached passage vectors · "
                f"{local_ai_status.embedding_model} cosine recall · "
                f"{local_ai_status.answer_model} grounded claim generation · no hosted API"
            )
            acceleration = get_local_ai_acceleration_status(
                settings.local_ai_url,
                settings.local_ai_answer_model,
                settings.local_ai_embedding_model,
            )
            st.caption("Hardware: " + acceleration.detail)
            if reranker_status.ready:
                st.caption("Local cross-encoder reranking is available for final evidence selection.")
            else:
                st.info(reranker_status.detail, icon=":material/offline_bolt:")
        else:
            st.info(
                "Local AI is unavailable, so questions will use the source-only fallback. "
                + local_ai_status.detail,
                icon=":material/offline_bolt:",
            )

        messages = st.session_state.setdefault("source_chat_messages", [])
        if messages and st.button(
            "Clear conversation",
            key="clear-source-chat",
            icon=":material/delete_sweep:",
        ):
            messages.clear()
            st.rerun()

        if not messages:
            st.info(
                "Ask a question in normal language. Each answer claim will cite the exact "
                "reviewed source passage used to support it."
            )

        for index, message in enumerate(messages):
            render_chat_message(message, index)

        question = st.chat_input(
            "Ask a question about the reviewed document",
            key="source-question",
            submit_mode="disable",
        )
        if question is None:
            return

        question = question.strip()
        if not question:
            st.info("Enter a question first.")
            return

        findings = detect_likely_phi(question)
        if findings:
            st.error(
                "Likely identifying information detected: " + ", ".join(findings) + ". "
                "Remove it and ask only about the source topic."
            )
            return

        user_message = {"role": "user", "content": question}
        messages.append(user_message)
        render_chat_message(user_message, len(messages) - 1)

        if use_local_ai:
            with st.status("Building a citation-aware answer…", expanded=True) as status:
                try:
                    client = make_local_ai_client()
                    reranker = None
                    status.write("Retrieving eligible reviewed evidence with BM25 and local vectors…")
                    if reranker_status.ready:
                        status.write("Loading the local cross-encoder and reranking evidence candidates…")
                        try:
                            reranker = load_local_reranker(
                                str(settings.local_ai_reranker_dir),
                                settings.local_ai_reranker_device,
                                settings.local_ai_reranker_batch_size,
                            )
                        except RerankerError as exc:
                            status.write("Reranker unavailable; continuing with hybrid evidence retrieval.")
                            st.caption(str(exc))
                    retrieval = retrieve_hybrid_sources(
                        settings,
                        question,
                        client=client,
                        reranker=reranker,
                        limit=8,
                        candidate_limit=settings.local_ai_candidate_limit,
                    )
                    citation_answer = None
                    if retrieval.status == "supported":
                        status.write("Generating claim-level citations and applying grounding checks…")
                        citation_answer = answer_with_citations(
                            question,
                            retrieval.results,
                            client=client,
                            retrieval_mode=retrieval.retrieval_mode,
                            reranker_used=retrieval.reranker_used,
                            max_claims=settings.local_ai_max_claims,
                        )
                        citation_answer = replace(
                            citation_answer,
                            timing=replace(
                                citation_answer.timing,
                                retrieval_seconds=retrieval.retrieval_seconds,
                                reranking_seconds=retrieval.reranking_seconds,
                            ),
                        )
                        if citation_answer.model_verification_used:
                            status.write("Completed deterministic and local claim verification.")
                        else:
                            status.write("Completed deterministic claim verification; local verifier was not needed.")
                    get_local_ai_acceleration_status.clear()
                    status.update(label="Citation-aware source check complete", state="complete")
                except (LocalAIError, RerankerError, ValueError) as exc:
                    retrieval = None
                    citation_answer = None
                    status.update(label="Local AI could not complete the source check", state="error")
                    st.caption(str(exc))

            if retrieval is not None and retrieval.status == "no_eligible_corpus":
                assistant_message = {
                    "role": "assistant",
                    "content": "I cannot answer from the source because no reviewed page is currently available.",
                }
            elif citation_answer is None:
                assistant_message = {
                    "role": "assistant",
                    "content": (
                        "I could not find enough reviewed source evidence to answer that "
                        "question. Try naming a medication, form, condition, or instruction "
                        "printed in the document."
                    ),
                }
            else:
                assistant_message = {
                    "role": "assistant",
                    "citation_answer": citation_answer,
                }
        else:
            response = search_verified_sources_with_status(settings, question, limit=8)
            if response.status == "no_eligible_corpus":
                assistant_message = {
                    "role": "assistant",
                    "content": "I cannot answer from the source because no reviewed page is currently available.",
                }
            elif response.status == "no_match":
                assistant_message = {
                    "role": "assistant",
                    "content": (
                        "I could not find enough reviewed source text to answer that question. "
                        "Try using a distinctive term printed in the document."
                    ),
                }
            else:
                assistant_message = {
                    "role": "assistant",
                    "citation_answer": source_only_citation_answer(
                        response.results,
                        retrieval_mode="bm25",
                        reason="Local AI is unavailable.",
                    ),
                }

        messages.append(assistant_message)
        render_chat_message(assistant_message, len(messages) - 1)


render_banner()
st.title("Source document chat")
st.write("Upload a visual document, check the extracted text, and ask for a cited answer.")
render_upload()
render_review()
render_questions()

with st.expander("How answers work", icon=":material/info:"):
    st.write(
        "The app retrieves only human-reviewed passages from approved, active documents. "
        "It combines BM25 exact-term search with a persistent local vector index, then uses a "
        "local cross-encoder when installed to select the strongest evidence. The local answer "
        "model returns short claims with citation IDs; every claim is checked again against its "
        "exact cited wording. Added numbers, changed units, missing required actions, unsupported "
        "claims, unavailable models, and conflicting numeric source values all produce a visible "
        "source-only or needs-review state instead of an invented answer."
    )
