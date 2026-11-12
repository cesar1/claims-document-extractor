from test_extract import sample_invoice_result

from app.review import ReviewSubmission, apply_review
from app.schemas import DocType, FieldName

LINE = {
    "description": "Drywall",
    "quantity": 10,
    "unit": "SF",
    "unit_price": 2.5,
    "amount": 25.0,
    "category": "Interior",
}


def result_with_lines(n: int):
    r = sample_invoice_result()
    base = r.line_items[0]
    r.line_items = [
        base.model_copy(update={"description": f"Line {i}", "page": i + 1}) for i in range(n)
    ]
    return r


def plain(r):
    return [{k: getattr(li, k) for k in LINE} for li in r.line_items]


def test_empty_submission_changes_nothing():
    r = sample_invoice_result()
    corrected, changes = apply_review(r, ReviewSubmission())
    assert changes == []
    assert corrected == r


def test_field_correction_is_recorded_and_trusted():
    r = sample_invoice_result()
    sub = ReviewSubmission(fields={"total": " 25.00 ", "invoice_number": "INV-77"})
    corrected, changes = apply_review(r, sub)

    assert [(c.target, c.name, c.before, c.after) for c in changes] == [
        ("field", "total", "27.00", "25.00")
    ]
    total = corrected.field(FieldName.total)
    assert (total.value, total.confidence, total.page, total.source_text) == ("25.00", 1.0, 1, None)
    # untouched fields keep the model's provenance
    assert corrected.field(FieldName.invoice_number) == r.field(FieldName.invoice_number)


def test_field_can_be_added_and_cleared():
    r = sample_invoice_result()
    sub = ReviewSubmission(fields={"vendor_name": "Acme Restoration", "claim_number": ""})
    corrected, changes = apply_review(r, sub)

    assert corrected.value(FieldName.vendor_name) == "Acme Restoration"
    assert corrected.value(FieldName.claim_number) is None
    assert {(c.name, c.before, c.after) for c in changes} == {
        ("vendor_name", None, "Acme Restoration"),
        ("claim_number", "CLM-1001", None),
    }


def test_clearing_an_absent_field_is_not_a_change():
    corrected, changes = apply_review(
        sample_invoice_result(), ReviewSubmission(fields={"vendor_name": None})
    )
    assert changes == []
    assert corrected.field(FieldName.vendor_name) is None


def test_doc_type_change():
    corrected, changes = apply_review(
        sample_invoice_result(), ReviewSubmission(doc_type="contractor_estimate")
    )
    assert corrected.doc_type is DocType.contractor_estimate
    assert corrected.doc_type_confidence == 1.0
    assert [(c.target, c.before, c.after) for c in changes] == [
        ("doc_type", "invoice", "contractor_estimate")
    ]


def test_removing_a_middle_line_is_one_change():
    r = result_with_lines(4)
    kept = plain(r)
    del kept[1]
    corrected, changes = apply_review(r, ReviewSubmission(line_items=kept))

    assert [(c.before_index, c.after_index) for c in changes] == [(1, None)]
    assert changes[0].after is None and changes[0].before["description"] == "Line 1"
    assert [li.description for li in corrected.line_items] == ["Line 0", "Line 2", "Line 3"]
    # surviving lines keep the model's page and confidence
    assert [li.page for li in corrected.line_items] == [1, 3, 4]
    assert all(li.confidence == 0.9 for li in corrected.line_items)


def test_edited_and_added_lines():
    r = result_with_lines(2)
    lines = plain(r)
    lines[1]["amount"] = 30.0
    lines.append({"description": "Paint"})
    corrected, changes = apply_review(r, ReviewSubmission(line_items=lines))

    assert [(c.before_index, c.after_index) for c in changes] == [(1, 1), (None, 2)]
    assert changes[0].before["amount"] == 25.0 and changes[0].after["amount"] == 30.0
    edited, added = corrected.line_items[1], corrected.line_items[2]
    assert (edited.amount, edited.confidence, edited.page) == (30.0, 1.0, 2)
    assert (added.description, added.amount, added.confidence, added.page) == (
        "Paint",
        None,
        1.0,
        None,
    )
