from datetime import date

from app.schemas import ExtractionResult
from app.validation import Severity, validate

TODAY = date(2026, 10, 31)


def _result(
    doc_type: str, fields: dict, lines: list[dict] = (), confidence: dict | None = None, notes=None
):
    confidence = confidence or {}
    return ExtractionResult.model_validate(
        {
            "doc_type": doc_type,
            "doc_type_confidence": 0.97,
            "fields": [
                {
                    "name": k,
                    "value": None if v is None else str(v),
                    "page": 1,
                    "source_text": None if v is None else str(v),
                    "confidence": confidence.get(k, 0.95),
                }
                for k, v in fields.items()
            ],
            "line_items": [
                {"unit": "EA", "category": None, "page": 1, "confidence": 0.95, **li}
                for li in lines
            ],
            "notes": notes,
        }
    )


def _line(desc, qty, price, amount=None):
    return {
        "description": desc,
        "quantity": qty,
        "unit_price": price,
        "amount": round(qty * price, 2) if amount is None else amount,
    }


INVOICE = {
    "claim_number": "CLM-1",
    "vendor_name": "ABC Plumbing",
    "invoice_number": "INV-9",
    "invoice_date": "2026-03-20",
    "date_of_loss": "2026-03-14",
    "subtotal": "300.00",
    "tax": "24.00",
    "total": "324.00",
    "amount_due": "324.00",
}
INVOICE_LINES = [_line("Repair", 1, 200.0), _line("Labor", 2, 50.0)]


def rules(report, severity=None):
    return {f.rule for f in report.flags if severity is None or f.severity is severity}


def test_clean_invoice_needs_no_review():
    rep = validate(_result("invoice", INVOICE, INVOICE_LINES), today=TODAY)
    assert rep.errors == 0 and rep.warnings == 0
    assert not rep.needs_review


def test_printed_total_math_error():
    rep = validate(
        _result("invoice", {**INVOICE, "total": "334.00", "amount_due": "334.00"}, INVOICE_LINES),
        today=TODAY,
    )
    assert "totals_mismatch" in rules(rep, Severity.error)
    assert rep.needs_review


def test_xactimate_lines_include_tax_and_op():
    # Line amounts include per-line tax and O&P; printed subtotal includes tax (Xactimate summary)
    lines = [_line("Drywall", 100, 2.0, amount=244.0), _line("Paint", 100, 1.0, amount=122.0)]
    fields = {
        "claim_number": "CLM-1",
        "contractor_name": "XYZ Restoration",
        "estimate_date": "2026-03-20",
        "date_of_loss": "2026-03-14",
        "subtotal": "306.00",  # 300 base + 6 tax
        "tax": "6.00",
        "overhead_and_profit": "60.00",
        "total": "366.00",
    }
    rep = validate(_result("contractor_estimate", fields, lines), today=TODAY)
    assert rep.errors == 0
    assert "lines_include_tax_op" in rules(rep, Severity.info)
    assert "line_math" not in rules(rep)


def test_missing_line_item_breaks_reconciliation():
    rep = validate(_result("invoice", INVOICE, INVOICE_LINES[:1]), today=TODAY)
    assert "lines_mismatch" in rules(rep, Severity.error)


def test_duplicate_charges_are_flagged_even_though_the_math_works():
    lines = [
        _line("Emergency service call", 1, 285.0),
        _line("Air mover (per 24 hour period)", 12, 31.5),
        _line("Remove baseboard", 30, 0.68),
        _line("Air mover - daily rental", 12, 31.5),  # same charge, different wording
        _line("Remove baseboard", 16, 0.68),  # same work in another room: not a duplicate
        _line("Emergency service call", 1, 285.0),
        _line("Emergency service call", 1, 285.0),
    ]
    subtotal = sum(li["amount"] for li in lines)
    fields = {**INVOICE, "subtotal": subtotal, "tax": 0, "total": subtotal, "amount_due": subtotal}
    rep = validate(_result("invoice", fields, lines), today=TODAY)

    assert rep.errors == 0
    dupes = [f for f in rep.flags if f.rule == "duplicate_lines"]
    assert all(f.severity is Severity.warning for f in dupes)
    assert [f.line_indices for f in dupes] == [[0, 5, 6], [1, 3]]
    assert "Lines 1, 6 and 7" in dupes[0].message and "$285.00" in dupes[0].message
    assert "Lines 2 and 4" in dupes[1].message


def test_same_numbers_in_different_units_are_not_duplicates():
    lines = [
        {**_line("Seal floor", 60, 1.0), "unit": "SF"},
        {**_line("Caulk trim", 60, 1.0), "unit": "LF"},
        _line("Dumpster", 1, 180.0),
    ]
    rep = validate(_result("invoice", INVOICE, lines), today=TODAY)
    assert "duplicate_lines" not in rules(rep)


def test_single_misread_line_is_pinpointed():
    lines = [_line(f"Item {i}", 1, 10.0) for i in range(4)] + [
        _line("Misread", 3, 10.0, amount=80.0)
    ]
    fields = {
        **INVOICE,
        "subtotal": "120.00",
        "tax": "0",
        "total": "120.00",
        "amount_due": "120.00",
    }
    rep = validate(_result("invoice", fields, lines), today=TODAY)
    flag = next(f for f in rep.flags if f.rule == "line_math")
    assert flag.severity is Severity.warning and flag.line_indices == [4]


def test_dates_out_of_order_or_in_future():
    rep = validate(
        _result("invoice", {**INVOICE, "invoice_date": "2026-03-01"}, INVOICE_LINES), today=TODAY
    )
    assert "dated_before_loss" in rules(rep, Severity.error)
    rep = validate(
        _result("invoice", {**INVOICE, "date_of_loss": "2027-01-01"}, INVOICE_LINES), today=TODAY
    )
    assert "loss_in_future" in rules(rep, Severity.error)


def test_unparseable_money_and_amount_due():
    rep = validate(
        _result("invoice", {**INVOICE, "tax": "see attached"}, INVOICE_LINES), today=TODAY
    )
    assert "unparseable" in rules(rep, Severity.error)
    rep = validate(
        _result("invoice", {**INVOICE, "amount_due": "400.00"}, INVOICE_LINES), today=TODAY
    )
    assert "amount_due_exceeds_total" in rules(rep, Severity.error)
    rep = validate(
        _result("invoice", {**INVOICE, "amount_due": "224.00"}, INVOICE_LINES), today=TODAY
    )
    assert "credits_applied" in rules(rep, Severity.info) and not rep.needs_review


def test_missing_claim_number_and_required_fields():
    rep = validate(
        _result("invoice", {**INVOICE, "claim_number": None, "vendor_name": None}, INVOICE_LINES),
        today=TODAY,
    )
    assert {"claim_unmatched", "required_missing"} <= rules(rep, Severity.warning)


def test_low_confidence_respects_threshold():
    res = _result("invoice", INVOICE, INVOICE_LINES, confidence={"invoice_number": 0.7})
    assert "low_confidence" in rules(validate(res, today=TODAY), Severity.warning)
    assert "low_confidence" not in rules(validate(res, today=TODAY, confidence_threshold=0.6))


FIRE = {
    "incident_number": "2026-18890",
    "fire_department": "Springfield Fire Department",
    "date_of_loss": "2026-02-11",
    "alarm_time": "2026-02-11T01:35",
    "arrival_time": "2026-02-11T01:40",
    "insured_name": "Anthony Larson",
    "cause_of_loss": "fire",
}


def test_fire_report_times_and_owner():
    rep = validate(_result("fire_report", FIRE, confidence={"insured_name": 0.7}), today=TODAY)
    assert not rep.needs_review  # owner/occupant uncertainty is informational, not a review trigger
    assert "owner_may_not_be_insured" in rules(rep, Severity.info)
    rep = validate(
        _result("fire_report", {**FIRE, "arrival_time": "2026-02-11T01:20"}), today=TODAY
    )
    assert "arrival_before_alarm" in rules(rep, Severity.error)
