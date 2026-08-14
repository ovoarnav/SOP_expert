from archbold_nav.chunking import split_verified_text


def test_evidence_units_preserve_exact_reviewed_source_spans() -> None:
    source = "Header\nFirst reviewed instruction. Second reviewed instruction.\nFinal line"

    units = split_verified_text(source, max_chars=30)

    assert [unit.text for unit in units] == [
        "Header",
        "First reviewed instruction.",
        "Second reviewed instruction.",
        "Final line",
    ]
    assert [source[unit.start_char : unit.end_char] for unit in units] == [
        unit.text for unit in units
    ]


def test_table_rows_keep_a_complete_instruction_together() -> None:
    source = (
        "Medication: Initiate implementation: Orders:\n"
        "Example Alpha NH100 5 units when needed for example.\n"
        "Do not exceed 12 units in a 24-hour period.\n"
        "Example Beta NH101 1 unit when needed.\n"
        "May have up to 4 units in a 24-hour period."
    )

    units = split_verified_text(source)

    assert [unit.text for unit in units] == [
        "Medication: Initiate implementation: Orders:",
        "Example Alpha NH100 5 units when needed for example.\nDo not exceed 12 units in a 24-hour period.",
        "Example Beta NH101 1 unit when needed.\nMay have up to 4 units in a 24-hour period.",
    ]
    assert [source[unit.start_char : unit.end_char] for unit in units] == [
        unit.text for unit in units
    ]


def test_parenthesized_table_marker_starts_a_new_source_row() -> None:
    source = (
        "Example Alpha NH100 5 units when needed.\n"
        "May not exceed 12 units in a 24-hour period.\n"
        "Example Beta (Working on a form for this) 1 unit once.\n"
        "Contact the example role if there is no relief."
    )

    units = split_verified_text(source)

    assert [unit.text for unit in units] == [
        "Example Alpha NH100 5 units when needed.\nMay not exceed 12 units in a 24-hour period.",
        "Example Beta (Working on a form for this) 1 unit once.\nContact the example role if there is no relief.",
    ]


def test_structured_review_rows_keep_labels_and_instruction_together() -> None:
    source = (
        "STANDING ORDERS\n\n"
        "Medication: Example Alpha\n"
        "Initiate implementation: DEMO-100\n"
        "Orders: Do not exceed 12 units in a 24-hour period.\n\n"
        "Medication: Example Beta\n"
        "Orders: May have up to 4 units in a 24-hour period."
    )

    units = split_verified_text(source)

    assert [unit.text for unit in units] == [
        "STANDING ORDERS",
        "Medication: Example Alpha\nInitiate implementation: DEMO-100\nOrders: Do not exceed 12 units in a 24-hour period.",
        "Medication: Example Beta\nOrders: May have up to 4 units in a 24-hour period.",
    ]
