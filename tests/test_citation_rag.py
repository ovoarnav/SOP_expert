from dataclasses import replace

from archbold_nav.citation_rag import (
    answer_with_citations,
    build_citations,
    has_conflicting_evidence,
)
from test_answering import make_result


class FakeCitationModel:
    answer_model = "synthetic-citation-model"

    def __init__(self, draft: dict, *, supported: bool = True) -> None:
        self.draft = draft
        self.supported = supported
        self.verification_calls = 0

    def generate_citation_draft(self, question, evidence, *, citation_ids, max_claims):
        return self.draft

    def verify_claim(self, claim, evidence):
        self.verification_calls += 1
        return self.supported


def test_claim_citations_are_bound_to_exact_reviewed_evidence() -> None:
    source = replace(
        make_result("DEMO ONLY. Example Alpha limit is 12 units in 24 hours."),
        chunk_id="citation-source",
        page_number=7,
    )
    model = FakeCitationModel(
        {
            "status": "answered",
            "claims": [
                {"text": "Example Alpha limit is 12 units in 24 hours.", "citation_ids": ["1"]}
            ],
        }
    )

    answer = answer_with_citations(
        "What is the Example Alpha limit in 24 hours?",
        [source],
        client=model,
        retrieval_mode="hybrid",
        reranker_used=True,
    )

    assert answer.status == "answered"
    assert answer.claims[0].citation_ids == ("1",)
    assert answer.citations[0].source.page_number == 7
    assert answer.citations[0].quote == source.text
    assert answer.reranker_used is True


def test_fabricated_number_reverts_to_cited_source_only_wording() -> None:
    source = make_result("DEMO ONLY. Example Alpha limit is 12 units in 24 hours.")
    model = FakeCitationModel(
        {
            "status": "answered",
            "claims": [
                {"text": "Example Alpha limit is 99 units in 24 hours.", "citation_ids": ["1"]}
            ],
        }
    )

    answer = answer_with_citations(
        "What is the Example Alpha limit in 24 hours?",
        [source],
        client=model,
        retrieval_mode="hybrid",
        reranker_used=False,
    )

    assert answer.status == "source_only"
    assert "12 units" in answer.claims[0].text
    assert "99" not in answer.claims[0].text


def test_invalid_citation_id_reverts_to_source_only_wording() -> None:
    source = make_result("DEMO ONLY. Record Example Action.")
    model = FakeCitationModel(
        {
            "status": "answered",
            "claims": [{"text": "Record Example Action.", "citation_ids": ["99"]}],
        }
    )

    answer = answer_with_citations(
        "What is the example action?", [source], client=model, retrieval_mode="hybrid", reranker_used=False
    )

    assert answer.status == "source_only"
    assert answer.fallback_reason == "A claim cited unavailable evidence."


def test_conflicting_single_value_evidence_requires_review() -> None:
    first = make_result("DEMO ONLY. Example Alpha maximum is 12 units.")
    second = replace(
        make_result("DEMO ONLY. Example Alpha maximum is 16 units."),
        chunk_id="other-chunk",
        page_id="other-page",
        page_number=8,
    )
    citations = build_citations([first, second])
    assert has_conflicting_evidence("What is the maximum Example Alpha amount?", citations) is True

    answer = answer_with_citations(
        "What is the maximum Example Alpha amount?",
        [first, second],
        client=FakeCitationModel({"status": "answered", "claims": []}),
        retrieval_mode="hybrid",
        reranker_used=True,
    )
    assert answer.status == "conflicting_evidence"
    assert len(answer.citations) == 2


def test_named_source_term_beats_an_unrelated_shared_time_period() -> None:
    named = make_result(
        "DEMO ONLY. Example Alpha may not exceed 12 units in a 24-hour period."
    )
    shared_period = replace(
        make_result("DEMO ONLY. Example Beta may have 4 units in a 24-hour period."),
        chunk_id="beta-chunk",
        page_id="beta-page",
        page_number=8,
    )

    citations = build_citations(
        [shared_period, named],
        question="What is the maximum Example Alpha amount allowed in 24 hours?",
    )

    assert [citation.quote for citation in citations] == [named.text]


def test_simple_single_citation_uses_deterministic_verification_without_second_model_call() -> None:
    source = make_result("DEMO ONLY. Example Alpha limit is 12 units in 24 hours.")
    model = FakeCitationModel(
        {
            "status": "answered",
            "claims": [
                {"text": "Example Alpha limit is 12 units in 24 hours.", "citation_ids": ["1"]}
            ],
        }
    )

    answer = answer_with_citations(
        "What is the Example Alpha limit in 24 hours?",
        [source],
        client=model,
        retrieval_mode="hybrid",
        reranker_used=True,
    )

    assert answer.status == "answered"
    assert answer.model_verification_used is False
    assert model.verification_calls == 0


def test_conditional_instruction_keeps_the_second_local_verification_call() -> None:
    source = make_result("DEMO ONLY. If the example check fails, call Example Role.")
    model = FakeCitationModel(
        {
            "status": "answered",
            "claims": [
                {"text": "If the example check fails, call Example Role.", "citation_ids": ["1"]}
            ],
        }
    )

    answer = answer_with_citations(
        "What should happen if the example check fails?",
        [source],
        client=model,
        retrieval_mode="hybrid",
        reranker_used=True,
    )

    assert answer.status == "answered"
    assert answer.model_verification_used is True
    assert model.verification_calls == 1
