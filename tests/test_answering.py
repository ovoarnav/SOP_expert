from dataclasses import replace
from pathlib import Path

import pytest

from archbold_nav.answering import compose_grounded_answer, select_answer_excerpt
from archbold_nav.search import SearchResult


SYNTHETIC_SOURCE = """DEMO ONLY EXAMPLE PROCESS
1) If Example Condition Alpha occurs, record Example Value 12.
Continue the first example action for 4 hours.
2) If there is no change after step 1, notify Example Role within 30 minutes.
Record the notification in the example log.
3) If there is no change after step 2, contact Example Supervisor.
Consider Example Review B also.
"""


def make_result(text: str = SYNTHETIC_SOURCE) -> SearchResult:
    return SearchResult(
        chunk_id="chunk-1",
        document_id="document-1",
        page_id="page-1",
        document_title="DEMO ONLY Example Process",
        original_filename="demo.pdf",
        managed_path=Path("demo.pdf"),
        facility="Demo Facility",
        version="demo-1",
        effective_date="2099-01-01",
        page_number=7,
        section_title="DEMO ONLY Example Process",
        section_type="reference",
        facility_scope="Demo Facility",
        form_number="DEMO-001",
        source_date="2099",
        text=text,
        matched_terms=("Example", "step", "1"),
        score=-1.0,
    )


def test_question_about_after_step_selects_the_next_complete_block() -> None:
    excerpt = select_answer_excerpt(
        "What happens after step 1?",
        SYNTHETIC_SOURCE,
    )

    assert excerpt.startswith("2) If there is no change after step 1")
    assert "within 30 minutes" in excerpt
    assert "Record the notification in the example log." in excerpt
    assert "contact Example Supervisor" not in excerpt


def test_explicit_step_question_selects_that_step() -> None:
    excerpt = select_answer_excerpt("What is step 3?", SYNTHETIC_SOURCE)

    assert excerpt.startswith("3) If there is no change after step 2")
    assert "Consider Example Review B also." in excerpt
    assert "Example Value 12" not in excerpt


def test_summary_question_keeps_all_source_numbers_exactly() -> None:
    excerpt = select_answer_excerpt("Summarize the entire sequence", SYNTHETIC_SOURCE)

    assert "Example Value 12" in excerpt
    assert "4 hours" in excerpt
    assert "30 minutes" in excerpt
    assert "Consider Example Review B also." in excerpt


def test_grounded_answer_names_the_source_page_and_uses_exact_excerpt() -> None:
    result = make_result()
    grounded = compose_grounded_answer("What happens after step 2?", [result])

    assert grounded.source is result
    assert grounded.text.startswith("According to the reviewed source on page 7:")
    assert grounded.exact_excerpt.replace("\n", " ") in grounded.text
    assert "contact Example Supervisor" in grounded.text


def test_grounded_answer_requires_a_reviewed_result() -> None:
    with pytest.raises(ValueError, match="At least one reviewed source result"):
        compose_grounded_answer("What is the example action?", [])


def test_inline_ocr_steps_are_separated_and_layout_marks_are_not_in_answer() -> None:
    inline_source = (
        "DEMO ONLY HEADER | 1) If Example Condition lasts 48hr, record Value 12. | "
        "|2) If no change after step 1, notify Example Role. | "
        "| 13) If no change after step 2, contact Example Supervisor."
    )
    result = make_result(inline_source)

    first = compose_grounded_answer(
        "What happens if Example Condition lasts 48 hours?",
        [result],
    )
    third = compose_grounded_answer("What happens after step 2?", [result])

    assert first.exact_excerpt.startswith("1) If Example Condition lasts 48hr")
    assert "Value 12" in first.text
    assert "notify Example Role" not in first.text
    assert not any(mark in first.text for mark in "|[]=")
    assert third.exact_excerpt.startswith("13) If no change after step 2")
    assert "contact Example Supervisor" in third.text


def test_unnumbered_long_ocr_line_returns_only_the_relevant_statement() -> None:
    long_source = (
        "DEMO ONLY REFERENCE. "
        "Example Item Alpha uses 2 example units every 6 hours. "
        "Do not exceed 12 example units of Example Compound in all forms in 24 hours. "
        "Example Item Beta uses 9 unrelated units every day."
    )
    result = make_result(long_source)

    grounded = compose_grounded_answer(
        "What is the maximum amount of Example Compound from all forms in 24 hours?",
        [result],
    )

    assert grounded.exact_excerpt == (
        "Do not exceed 12 example units of Example Compound in all forms in 24 hours."
    )
    assert "Example Item Alpha" not in grounded.text
    assert "Example Item Beta" not in grounded.text


def test_named_item_is_linked_to_its_following_limit_statement() -> None:
    source = (
        "Example Item Alpha 2 units after each example event. "
        "May have up to 8 units (4 examples) in a 24-hour period. "
        "Example Item Beta 4 units once, if no change within 24 hours contact Example Role."
    )

    grounded = compose_grounded_answer(
        "What is the maximum amount of Example Item Alpha allowed within 24 hours?",
        [make_result(source)],
    )

    assert "up to 8 units (4 examples)" in grounded.text
    assert "up to 8 units (4 examples) in a 24-hour period" in grounded.text
    assert "Example Item Beta" not in grounded.text


def test_ocr_line_wrap_does_not_cut_a_source_action_in_half() -> None:
    source = (
        "May have up to 8 example units in a 24-hour period. "
        "More than 3 example events in 24\n"
        "hours, send an Example Sample and begin Example Precautions. "
        "Unrelated Example Item follows."
    )

    grounded = compose_grounded_answer(
        "What should happen after more than three example events within 24 hours?",
        [make_result(source)],
    )

    assert "More than 3 example events in 24 hours" in grounded.text
    assert "send an Example Sample and begin Example Precautions" in grounded.text
    assert "May have up to 8" not in grounded.text


def test_answer_reranks_pages_using_normalized_a1c_terms() -> None:
    irrelevant = replace(
        make_result(
            "DEMO ONLY PRESSURE REFERENCE. Turn Example Resident every two hours."
        ),
        page_number=12,
        page_id="page-12",
        chunk_id="chunk-12",
        score=-9.0,
    )
    relevant = replace(
        make_result(
            "DEMO ONLY LAB REFERENCE. If Example Resident is Diabetic, Alc every 3 mo. "
            "If one has not been drawn in the last 3 mo., repeat in 3 mo."
        ),
        page_number=1,
        page_id="page-1",
        chunk_id="chunk-1",
        score=-1.0,
    )

    grounded = compose_grounded_answer(
        "How frequently should A1c be checked for a diabetic resident?",
        [irrelevant, relevant],
    )

    assert grounded.source.page_number == 1
    assert "Alc every 3 mo." in grounded.text
    assert "Turn Example Resident" not in grounded.text


def test_numeric_threshold_anchors_the_following_conditional_action() -> None:
    source = (
        "Remove Example Device. If no Example Output in 6-8 hr, obtain Example Scan. "
        "If > 400cc then perform Example Action One. "
        "If no Example Output in 6-8 hr, notify Example Role."
    )

    excerpt = select_answer_excerpt(
        "After removal, what happens when the Example Scan shows more than 400 cc?",
        source,
    )

    assert excerpt == "If > 400cc then perform Example Action One."


def test_replacement_question_selects_the_replacement_block() -> None:
    source = (
        "1. Cover with Example Absorbent Dressing that is 3-4 cm larger.\n"
        "2. Use Example Gauze for a different condition.\n"
        "6. Replace Example Absorbent Dressing if it is leaking or falls off. "
        "Change Example Absorbent every 7-10 days."
    )

    excerpt = select_answer_excerpt(
        "When should the Example Absorbent dressing be replaced?",
        source,
    )

    assert excerpt.startswith("6. Replace Example Absorbent Dressing")
    assert "every 7-10 days" in excerpt
    assert "3-4 cm" not in excerpt


def test_ocr_bullet_stops_conditional_excerpt_before_the_next_instruction() -> None:
    source = (
        "¢ Document Example Tear upon occurrence and resolution "
        "¢ Ifskin tear develops signs of Example Infection, document and notify Example Role "
        "¢ Consult Example Specialist for large areas"
    )

    excerpt = select_answer_excerpt(
        "What should staff do if a skin tear develops signs of Example Infection?",
        source,
    )

    assert excerpt.startswith("Ifskin tear develops signs of Example Infection")
    assert "document and notify Example Role" in excerpt
    assert "Consult Example Specialist" not in excerpt


def test_source_fallback_trims_trailing_ocr_noise_after_complete_action() -> None:
    source = (
        "3) If no Example Event after step 2, call Example Role. "
        "Consider Example Review also. F- a a ee SUES"
    )

    grounded = compose_grounded_answer(
        "What happens after step 2?",
        [make_result(source)],
    )

    assert "call Example Role" in grounded.text
    assert "Consider Example Review also." in grounded.text
    assert "SUES" not in grounded.text
