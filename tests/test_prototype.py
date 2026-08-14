from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from archbold_nav.db import db_session
from archbold_nav.ingest import DuplicateDocumentError, ingest_pdf
from archbold_nav.ocr import OcrResult
from archbold_nav.repository import (
    enable_page_for_search,
    exclude_legacy_unchanged_ocr_pages,
    get_document,
    get_document_readiness,
    get_page,
    get_source_library_summary,
    list_documents,
    list_pages,
    update_document,
    update_page_review,
)
from archbold_nav.safety import detect_likely_phi
from archbold_nav.search import (
    search_verified_sources,
    search_verified_sources_with_status,
)


@pytest.fixture(autouse=True)
def allow_synthetic_ocr_without_system_tesseract(monkeypatch) -> None:
    monkeypatch.setattr(
        "archbold_nav.ingest.assert_tesseract_available",
        lambda: Path("synthetic-tesseract"),
    )


def make_demo_pdf(path: Path, pages: int = 1) -> None:
    doc = fitz.open()
    for page_number in range(1, pages + 1):
        page = doc.new_page()
        page.insert_text(
            (72, 72),
            f"DEMO ONLY POLICY PAGE {page_number}\nExample Topic Alpha source instruction.",
        )
    doc.save(path)
    doc.close()


def fake_ocr(image_path: Path, psm: int = 6) -> OcrResult:
    page_number = int(image_path.stem.split("-")[-1])
    return OcrResult(
        text=(
            f"DEMO ONLY POLICY PAGE {page_number}\n"
            "Example Topic Alpha source instruction.\n"
            "Escalate to Example Role when Example Condition occurs."
        ),
        confidence=98.0,
        engine=f"fake-psm{psm}",
        warnings=[],
    )


def set_document_status(
    settings,
    document_id: str,
    *,
    approval_status: str,
    lifecycle_status: str,
) -> None:
    document = get_document(settings, document_id)
    assert document is not None
    update_document(
        settings,
        document_id,
        title=document["title"],
        facility="Demo Facility",
        document_type="policy",
        version="demo-1",
        effective_date="2099-01-01",
        review_date="2099-12-31",
        approval_status=approval_status,
        lifecycle_status=lifecycle_status,
    )


def approve_document(settings, document_id: str) -> None:
    set_document_status(
        settings,
        document_id,
        approval_status="approved",
        lifecycle_status="active",
    )


def test_pending_ocr_is_not_searchable_until_page_and_document_are_approved(
    tmp_path: Path, settings, monkeypatch
) -> None:
    source = tmp_path / "demo.pdf"
    make_demo_pdf(source)
    monkeypatch.setattr("archbold_nav.ingest.ocr_page", fake_ocr)

    document_id = ingest_pdf(source, settings)
    pages = list_pages(settings, document_id)
    assert len(pages) == 1
    assert pages[0]["verification_status"] == "pending"
    assert search_verified_sources(settings, "Example Topic Alpha") == []

    update_page_review(
        settings,
        pages[0]["id"],
        verified_text=pages[0]["raw_ocr_text"],
        verification_status="verified",
        searchable=True,
        section_title="DEMO ONLY Example Policy",
        section_type="reference",
        facility_scope="Demo Facility",
        form_number="DEMO-001",
        source_date="2099",
    )
    assert search_verified_sources(settings, "Example Topic Alpha") == []

    approve_document(settings, document_id)
    results = search_verified_sources(settings, "Example Topic Alpha")
    assert len(results) == 3
    assert results[0].page_number == 1
    assert "Example Topic Alpha" in results[0].text
    assert results[0].matched_terms == ("Example", "Topic", "Alpha")


def test_legacy_unchanged_ocr_is_removed_from_search_once(
    tmp_path: Path, settings, monkeypatch
) -> None:
    source = tmp_path / "legacy-ocr.pdf"
    make_demo_pdf(source)
    monkeypatch.setattr("archbold_nav.ingest.ocr_page", fake_ocr)
    document_id = ingest_pdf(source, settings)
    page = list_pages(settings, document_id)[0]

    update_page_review(
        settings,
        page["id"],
        verified_text=page["raw_ocr_text"],
        verification_status="verified",
        searchable=True,
        section_title="DEMO ONLY Example Policy",
        section_type="reference",
        facility_scope="Demo Facility",
        form_number=None,
        source_date=None,
    )
    approve_document(settings, document_id)
    assert search_verified_sources(settings, "Example Topic Alpha")

    assert exclude_legacy_unchanged_ocr_pages(settings) == 1
    updated_page = get_page(settings, page["id"])
    assert updated_page is not None
    assert updated_page["verification_status"] == "pending"
    assert updated_page["searchable"] == 0
    assert search_verified_sources(settings, "Example Topic Alpha") == []
    assert exclude_legacy_unchanged_ocr_pages(settings) == 0


def test_unverifying_page_removes_it_from_fts(
    tmp_path: Path, settings, monkeypatch
) -> None:
    source = tmp_path / "demo.pdf"
    make_demo_pdf(source)
    monkeypatch.setattr("archbold_nav.ingest.ocr_page", fake_ocr)
    document_id = ingest_pdf(source, settings)
    page = list_pages(settings, document_id)[0]

    update_page_review(
        settings,
        page["id"],
        verified_text=page["raw_ocr_text"],
        verification_status="verified",
        searchable=True,
        section_title="DEMO ONLY Example Policy",
        section_type="reference",
        facility_scope="Demo Facility",
        form_number=None,
        source_date=None,
    )
    approve_document(settings, document_id)
    assert search_verified_sources(settings, "Example Topic Alpha")

    update_page_review(
        settings,
        page["id"],
        verified_text=page["raw_ocr_text"],
        verification_status="pending",
        searchable=False,
        section_title="DEMO ONLY Example Policy",
        section_type="reference",
        facility_scope="Demo Facility",
        form_number=None,
        source_date=None,
    )
    assert search_verified_sources(settings, "Example Topic Alpha") == []


def test_verified_searchable_page_is_indexed_in_one_review_update(
    tmp_path: Path, settings, monkeypatch
) -> None:
    source = tmp_path / "one-save.pdf"
    make_demo_pdf(source)
    monkeypatch.setattr("archbold_nav.ingest.ocr_page", fake_ocr)
    document_id = ingest_pdf(source, settings)
    page = list_pages(settings, document_id)[0]

    update_page_review(
        settings,
        page["id"],
        verified_text=page["raw_ocr_text"],
        verification_status="verified",
        searchable=True,
        section_title="DEMO ONLY Example Policy",
        section_type="reference",
        facility_scope="Demo Facility",
        form_number="DEMO-ONE-SAVE",
        source_date="2099",
    )

    updated_page = get_page(settings, page["id"])
    assert updated_page is not None
    assert updated_page["verification_status"] == "verified"
    assert updated_page["searchable"] == 1
    with db_session(settings) as conn:
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE page_id = ?", (page["id"],)
        ).fetchone()["n"] == 3
        rows = conn.execute(
            "SELECT source_start, source_end, text FROM chunks WHERE page_id = ? ORDER BY ordinal",
            (page["id"],),
        ).fetchall()
        assert all(row["source_end"] > row["source_start"] for row in rows)
        assert all(row["text"] for row in rows)
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM chunks_fts WHERE page_id = ?", (page["id"],)
        ).fetchone()["n"] == 3


def test_unverified_page_cannot_become_searchable(
    tmp_path: Path, settings, monkeypatch
) -> None:
    source = tmp_path / "unverified.pdf"
    make_demo_pdf(source)
    monkeypatch.setattr("archbold_nav.ingest.ocr_page", fake_ocr)
    document_id = ingest_pdf(source, settings)
    page = list_pages(settings, document_id)[0]

    with pytest.raises(ValueError, match="cannot be searchable"):
        update_page_review(
            settings,
            page["id"],
            verified_text=page["raw_ocr_text"],
            verification_status="pending",
            searchable=True,
            section_title="DEMO ONLY Example Policy",
            section_type="reference",
            facility_scope="Demo Facility",
            form_number=None,
            source_date=None,
        )

    unchanged_page = get_page(settings, page["id"])
    assert unchanged_page is not None
    assert unchanged_page["verification_status"] == "pending"
    assert unchanged_page["searchable"] == 0


def test_removing_searchable_flag_removes_page_from_fts(
    tmp_path: Path, settings, monkeypatch
) -> None:
    source = tmp_path / "remove-searchable.pdf"
    make_demo_pdf(source)
    monkeypatch.setattr("archbold_nav.ingest.ocr_page", fake_ocr)
    document_id = ingest_pdf(source, settings)
    page = list_pages(settings, document_id)[0]

    update_page_review(
        settings,
        page["id"],
        verified_text=page["raw_ocr_text"],
        verification_status="verified",
        searchable=True,
        section_title="DEMO ONLY Example Policy",
        section_type="reference",
        facility_scope="Demo Facility",
        form_number=None,
        source_date=None,
    )
    approve_document(settings, document_id)
    assert search_verified_sources(settings, "Example Topic Alpha")

    update_page_review(
        settings,
        page["id"],
        verified_text=page["raw_ocr_text"],
        verification_status="verified",
        searchable=False,
        section_title="DEMO ONLY Example Policy",
        section_type="reference",
        facility_scope="Demo Facility",
        form_number=None,
        source_date=None,
    )

    assert search_verified_sources(settings, "Example Topic Alpha") == []
    with db_session(settings) as conn:
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE page_id = ?", (page["id"],)
        ).fetchone()["n"] == 0
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM chunks_fts WHERE page_id = ?", (page["id"],)
        ).fetchone()["n"] == 0


def test_document_approval_and_lifecycle_gate_already_indexed_page(
    tmp_path: Path, settings, monkeypatch
) -> None:
    source = tmp_path / "document-gates.pdf"
    make_demo_pdf(source)
    monkeypatch.setattr("archbold_nav.ingest.ocr_page", fake_ocr)
    document_id = ingest_pdf(source, settings)
    page = list_pages(settings, document_id)[0]
    update_page_review(
        settings,
        page["id"],
        verified_text=page["raw_ocr_text"],
        verification_status="verified",
        searchable=True,
        section_title="DEMO ONLY Example Policy",
        section_type="reference",
        facility_scope="Demo Facility",
        form_number=None,
        source_date=None,
    )

    set_document_status(
        settings,
        document_id,
        approval_status="unreviewed",
        lifecycle_status="active",
    )
    assert search_verified_sources(settings, "Example Topic Alpha") == []

    set_document_status(
        settings,
        document_id,
        approval_status="approved",
        lifecycle_status="draft",
    )
    assert search_verified_sources(settings, "Example Topic Alpha") == []

    approve_document(settings, document_id)
    assert search_verified_sources(settings, "Example Topic Alpha")


def test_document_readiness_reports_blockers_and_ready_state(
    tmp_path: Path, settings, monkeypatch
) -> None:
    source = tmp_path / "readiness.pdf"
    make_demo_pdf(source)
    monkeypatch.setattr("archbold_nav.ingest.ocr_page", fake_ocr)
    document_id = ingest_pdf(source, settings)
    page = list_pages(settings, document_id)[0]

    initial = get_document_readiness(settings, document_id)
    assert initial is not None
    assert initial.pdf_imported_locally is True
    assert initial.ocr_status == "ocr_complete"
    assert initial.total_page_count == 1
    assert initial.verified_page_count == 0
    assert initial.verified_searchable_page_count == 0
    assert initial.indexed_chunk_count == 0
    assert initial.eligible_for_nurse_search is False
    assert initial.blockers == (
        "No page has been human-verified and included in search.",
        "No searchable chunks are indexed.",
        "Document approval status is unreviewed.",
        "Document lifecycle status is draft.",
    )

    update_page_review(
        settings,
        page["id"],
        verified_text=page["raw_ocr_text"],
        verification_status="verified",
        searchable=True,
        section_title="DEMO ONLY Example Policy",
        section_type="reference",
        facility_scope="Demo Facility",
        form_number=None,
        source_date=None,
    )
    approve_document(settings, document_id)

    ready = get_document_readiness(settings, document_id)
    assert ready is not None
    assert ready.verified_page_count == 1
    assert ready.verified_searchable_page_count == 1
    assert ready.indexed_chunk_count == 3
    assert ready.approval_status == "approved"
    assert ready.lifecycle_status == "active"
    assert ready.eligible_for_nurse_search is True
    assert ready.blockers == ()

    with db_session(settings) as conn:
        conn.execute(
            "UPDATE documents SET parse_status = 'pending' WHERE id = ?", (document_id,)
        )
    incomplete_ocr = get_document_readiness(settings, document_id)
    assert incomplete_ocr is not None
    assert incomplete_ocr.eligible_for_nurse_search is False
    assert incomplete_ocr.blockers == ("OCR has not completed.",)


def test_search_status_distinguishes_empty_corpus_from_no_match(
    tmp_path: Path, settings, monkeypatch
) -> None:
    source = tmp_path / "search-status.pdf"
    make_demo_pdf(source)
    monkeypatch.setattr("archbold_nav.ingest.ocr_page", fake_ocr)
    document_id = ingest_pdf(source, settings)
    page = list_pages(settings, document_id)[0]

    no_corpus = search_verified_sources_with_status(settings, "QuasarNebula")
    assert no_corpus.status == "no_eligible_corpus"
    assert no_corpus.results == []

    update_page_review(
        settings,
        page["id"],
        verified_text=page["raw_ocr_text"],
        verification_status="verified",
        searchable=True,
        section_title="DEMO ONLY Example Policy",
        section_type="reference",
        facility_scope="Demo Facility",
        form_number=None,
        source_date=None,
    )
    approve_document(settings, document_id)

    no_match = search_verified_sources_with_status(settings, "QuasarNebula")
    assert no_match.status == "no_match"
    assert no_match.results == []
    supported = search_verified_sources_with_status(settings, "Example Topic Alpha")
    assert supported.status == "supported"
    assert len(supported.results) == 3


def test_source_library_summary_tracks_each_readiness_stage(
    tmp_path: Path, settings, monkeypatch
) -> None:
    empty = get_source_library_summary(settings)
    assert empty.imported_document_count == 0
    assert empty.ready_for_search is False

    source = tmp_path / "library-summary.pdf"
    make_demo_pdf(source)
    monkeypatch.setattr("archbold_nav.ingest.ocr_page", fake_ocr)
    document_id = ingest_pdf(source, settings)
    page = list_pages(settings, document_id)[0]

    imported = get_source_library_summary(settings)
    assert imported.imported_document_count == 1
    assert imported.verified_searchable_page_count == 0
    assert imported.eligible_chunk_count == 0

    update_page_review(
        settings,
        page["id"],
        verified_text=page["raw_ocr_text"],
        verification_status="verified",
        searchable=True,
        section_title="DEMO ONLY Example Policy",
        section_type="reference",
        facility_scope="Demo Facility",
        form_number=None,
        source_date=None,
    )
    reviewed = get_source_library_summary(settings)
    assert reviewed.verified_searchable_page_count == 1
    assert reviewed.indexed_chunk_count == 3
    assert reviewed.eligible_page_count == 0
    assert reviewed.ready_for_search is False

    approve_document(settings, document_id)
    ready = get_source_library_summary(settings)
    assert ready.active_approved_document_count == 1
    assert ready.eligible_page_count == 1
    assert ready.eligible_chunk_count == 3
    assert ready.ready_for_search is True


def test_missing_ocr_preflight_leaves_no_failed_document_or_managed_copy(
    tmp_path: Path, settings, monkeypatch
) -> None:
    source = tmp_path / "missing-ocr.pdf"
    make_demo_pdf(source)

    def missing_tesseract() -> Path:
        raise RuntimeError("Tesseract OCR is not installed")

    monkeypatch.setattr(
        "archbold_nav.ingest.assert_tesseract_available", missing_tesseract
    )

    with pytest.raises(RuntimeError, match="Tesseract OCR is not installed"):
        ingest_pdf(source, settings)

    assert list_documents(settings) == []
    assert list(settings.originals_dir.rglob("*.pdf")) == []


def test_single_page_enable_workflow_requires_explicit_confirmations(
    tmp_path: Path, settings, monkeypatch
) -> None:
    source = tmp_path / "confirmation-required.pdf"
    make_demo_pdf(source)
    monkeypatch.setattr("archbold_nav.ingest.ocr_page", fake_ocr)
    document_id = ingest_pdf(source, settings)
    page = list_pages(settings, document_id)[0]

    with pytest.raises(ValueError, match="checked against the scan"):
        enable_page_for_search(
            settings,
            page["id"],
            verified_text=page["raw_ocr_text"],
            section_title="DEMO ONLY Example Source",
            page_review_confirmed=False,
            document_approval_confirmed=True,
        )
    with pytest.raises(ValueError, match="approved and current"):
        enable_page_for_search(
            settings,
            page["id"],
            verified_text=page["raw_ocr_text"],
            section_title="DEMO ONLY Example Source",
            page_review_confirmed=True,
            document_approval_confirmed=False,
        )

    unchanged = get_page(settings, page["id"])
    assert unchanged is not None
    assert unchanged["verification_status"] == "pending"
    assert unchanged["searchable"] == 0


def test_single_page_enable_workflow_makes_source_immediately_searchable(
    tmp_path: Path, settings, monkeypatch
) -> None:
    source = tmp_path / "single-flow.pdf"
    make_demo_pdf(source)
    monkeypatch.setattr("archbold_nav.ingest.ocr_page", fake_ocr)
    document_id = ingest_pdf(source, settings)
    page = list_pages(settings, document_id)[0]

    enable_page_for_search(
        settings,
        page["id"],
        verified_text=page["raw_ocr_text"],
        section_title="DEMO ONLY Example Source",
        page_review_confirmed=True,
        document_approval_confirmed=True,
    )

    updated_page = get_page(settings, page["id"])
    updated_document = get_document(settings, document_id)
    assert updated_page is not None
    assert updated_page["verification_status"] == "verified"
    assert updated_page["searchable"] == 1
    assert updated_document is not None
    assert updated_document["approval_status"] == "approved"
    assert updated_document["lifecycle_status"] == "active"
    assert search_verified_sources(settings, "Example Topic Alpha")


def test_duplicate_file_is_rejected(tmp_path: Path, settings, monkeypatch) -> None:
    source = tmp_path / "duplicate.pdf"
    make_demo_pdf(source)
    monkeypatch.setattr("archbold_nav.ingest.ocr_page", fake_ocr)
    ingest_pdf(source, settings)
    with pytest.raises(DuplicateDocumentError):
        ingest_pdf(source, settings)


def test_approval_requires_a_verified_searchable_page(
    tmp_path: Path, settings, monkeypatch
) -> None:
    source = tmp_path / "not-reviewed.pdf"
    make_demo_pdf(source)
    monkeypatch.setattr("archbold_nav.ingest.ocr_page", fake_ocr)
    document_id = ingest_pdf(source, settings)
    document = get_document(settings, document_id)
    assert document is not None

    with pytest.raises(ValueError, match="At least one page"):
        update_document(
            settings,
            document_id,
            title=document["title"],
            facility="Demo Facility",
            document_type="policy",
            version="demo",
            effective_date=None,
            review_date=None,
            approval_status="approved",
            lifecycle_status="active",
        )


def test_likely_phi_detector() -> None:
    findings = detect_likely_phi(
        "Resident name: Jane Example, room 204, DOB 01/02/1940"
    )
    assert "resident or patient name" in findings
    assert "room number" in findings
    assert "date of birth" in findings
    assert detect_likely_phi("What does Example Topic Alpha say?") == []
