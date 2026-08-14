from dataclasses import replace

from archbold_nav.local_ai import (
    LocalAIError,
    answer_with_local_ai,
    validate_generated_answer,
)
from archbold_nav.search import _fts_query
from test_answering import make_result


def test_grounding_validation_accepts_supported_unit_expansion() -> None:
    valid, reason = validate_generated_answer(
        "Example A1c should be checked every 3 months for the example resident.",
        evidence="Example Alc every 3 mo.",
        question="How often should Example A1c be checked for the example resident?",
    )

    assert valid is True
    assert reason.startswith("Validated")


def test_grounding_validation_rejects_an_added_number() -> None:
    valid, reason = validate_generated_answer(
        "The example limit is 99 units in 24 hours.",
        evidence="The example limit is 8 units in 24 hours.",
        question="What is the example limit in 24 hours?",
    )

    assert valid is False
    assert "added a number" in reason


def test_grounding_validation_rejects_a_changed_unit() -> None:
    valid, reason = validate_generated_answer(
        "The example limit is 8 ml in 24 hours.",
        evidence="The example limit is 8 mg in 24 hours.",
        question="What is the example limit in 24 hours?",
    )

    assert valid is False
    assert "changed a number or unit" in reason


def test_grounding_validation_rejects_generated_advice() -> None:
    valid, reason = validate_generated_answer(
        "I recommend the example action.",
        evidence="Record the example action.",
        question="What action is recorded?",
    )

    assert valid is False
    assert "advice" in reason


def test_grounding_validation_rejects_an_omitted_source_action() -> None:
    valid, reason = validate_generated_answer(
        "Call Example Role.",
        evidence="Call Example Role and consider Example Review.",
        question="What should happen?",
    )

    assert valid is False
    assert "omitted a required action" in reason


def test_grounding_validation_allows_a_concise_numeric_answer_from_a_mixed_row() -> None:
    valid, reason = validate_generated_answer(
        "The maximum is 12 units in a 24-hour period.",
        evidence=(
            "Example Alpha may not exceed 12 units in a 24-hour period. "
            "Send the implementation form to the example role."
        ),
        question="What is the maximum Example Alpha amount allowed in 24 hours?",
    )

    assert valid is True
    assert reason.startswith("Validated")


class FakeLocalAI:
    answer_model = "synthetic-local-model"

    def __init__(self, answer: str, *, fail_retrieval: bool = False) -> None:
        self.answer = answer
        self.fail_retrieval = fail_retrieval

    def semantic_rerank(self, question, results, *, limit=5):
        if self.fail_retrieval:
            raise LocalAIError("Synthetic local service unavailable.")
        return list(reversed(results))[:limit]

    def generate_grounded_answer(self, question: str, evidence: str) -> str:
        return self.answer


def test_hybrid_answer_uses_validated_local_output() -> None:
    irrelevant = make_result("DEMO ONLY. Example unrelated value is 2 units.")
    relevant = replace(
        make_result("DEMO ONLY. The Example Alpha limit is 12 units in 24 hours."),
        chunk_id="chunk-relevant",
        page_id="page-relevant",
        page_number=4,
    )

    outcome = answer_with_local_ai(
        "What is the Example Alpha limit in 24 hours?",
        eligible_results=[irrelevant, relevant],
        lexical_results=[relevant],
        client=FakeLocalAI("The Example Alpha limit is 12 units in 24 hours."),
    )

    assert outcome.mode == "local_ai"
    assert outcome.retrieval_mode == "semantic"
    assert outcome.model_name == "synthetic-local-model"
    assert outcome.answer.source.page_number == 4
    assert outcome.answer.text == "The Example Alpha limit is 12 units in 24 hours."


def test_hybrid_answer_rejects_ungrounded_output_and_falls_back() -> None:
    relevant = make_result(
        "DEMO ONLY. The Example Alpha limit is 12 units in 24 hours."
    )

    outcome = answer_with_local_ai(
        "What is the Example Alpha limit in 24 hours?",
        eligible_results=[relevant],
        lexical_results=[relevant],
        client=FakeLocalAI("The Example Alpha limit is 99 units in 24 hours."),
    )

    assert outcome.mode == "source_only"
    assert outcome.answer.text.startswith("According to the reviewed source")
    assert "12 units" in outcome.answer.text
    assert "99" not in outcome.answer.text
    assert outcome.fallback_reason is not None


def test_hybrid_answer_uses_lexical_fallback_when_local_retrieval_is_down() -> None:
    relevant = make_result("DEMO ONLY. Record Example Action Alpha.")

    outcome = answer_with_local_ai(
        "What is Example Action Alpha?",
        eligible_results=[relevant],
        lexical_results=[relevant],
        client=FakeLocalAI("", fail_retrieval=True),
    )

    assert outcome.mode == "source_only"
    assert outcome.retrieval_mode == "lexical"
    assert "Example Action Alpha" in outcome.answer.text


def test_fts_query_drops_question_filler_and_expands_a1c_ocr_alias() -> None:
    query = _fts_query("How often should A1c be checked for a resident?")

    assert '"a1c"' in query
    assert '"alc"' in query
    assert '"how"' not in query
    assert '"for"' not in query
