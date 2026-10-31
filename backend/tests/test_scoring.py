from app.schemas import ExtractionResult
from app.scoring import field_matches, score_document, summarize


def _s(v):
    return None if v is None else str(v)


def _pred(fields: dict, lines: list[dict], doc_type="invoice") -> ExtractionResult:
    return ExtractionResult.model_validate(
        {
            "doc_type": doc_type,
            "doc_type_confidence": 0.95,
            "fields": [
                {"name": k, "value": _s(v), "page": 1, "source_text": _s(v), "confidence": 0.95}
                for k, v in fields.items()
            ],
            "line_items": [
                {"category": None, "unit": "EA", "page": 1, "confidence": 0.9, **li} for li in lines
            ],
            "notes": None,
        }
    )


TRUTH = {
    "doc_type": "invoice",
    "scan": False,
    "fields": {
        "claim_number": "CLM-100",
        "policy_number": None,
        "invoice_date": "2026-04-01",
        "total": 1234.5,
        "cause_of_loss_detail": "Water Damage - burst supply line",
    },
    "line_items": [
        {"description": "Drywall", "quantity": 10, "unit_price": 2.5, "amount": 25.0},
        {"description": "Paint", "quantity": 4, "unit_price": 30.0, "amount": 120.0},
        {"description": "Labor", "quantity": 2, "unit_price": 65.0, "amount": 130.0},
    ],
}


def test_field_matching_rules():
    assert field_matches("total", 1234.5, "$1,234.50")
    assert not field_matches("total", 1234.5, "1234.60")
    assert field_matches("invoice_date", "2026-04-01", "2026-04-01")
    assert field_matches("alarm_time", "2026-02-11T01:35", "2026-02-11 01:35")
    assert field_matches("claim_number", "CLM-100", "clm-100")
    assert field_matches(
        "cause_of_loss_detail",
        "Water Damage - burst supply line",
        "Water damage, burst supply line",
    )
    assert not field_matches("insured_name", "Jane Doe", "John Doe")


def test_narrative_graded_on_content_not_wording():
    truth = (
        "Engine and ladder companies responded to a reported structure fire at the above address. "
        "On arrival crews found smoke showing from the bedroom. Fire was contained to the area of "
        "origin and extinguished with one handline. Occupants were evacuated prior to arrival. "
        "Scene turned over to the owner after overhaul."
    )
    condensed = (
        "Engine and ladder companies responded to a structure fire; smoke showing from the bedroom "
        "on arrival. Fire was contained to the area of origin and extinguished with one handline. "
        "Occupants evacuated prior to arrival. Scene turned over to owner after overhaul."
    )
    assert field_matches("narrative_summary", truth, condensed)
    assert not field_matches("narrative_summary", truth, "Kitchen fire, two injured, total loss.")


def test_outcomes_correct_missed_hallucinated_wrong():
    pred = _pred(
        {
            "claim_number": "CLM-100",  # correct
            "policy_number": "HO-999",  # hallucinated (truth is null)
            "invoice_date": None,  # missed
            "total": "1200.00",  # wrong
            "cause_of_loss_detail": "Water damage - burst supply line",  # correct (fuzzy)
        },
        TRUTH["line_items"],
    )
    ds = score_document("inv-1", TRUTH, pred)
    outcomes = {f.name: f.outcome for f in ds.fields}
    assert outcomes == {
        "claim_number": "correct",
        "policy_number": "hallucinated",
        "invoice_date": "missed",
        "total": "wrong",
        "cause_of_loss_detail": "correct",
    }
    assert ds.doc_type_ok


def test_line_alignment_tolerates_dropped_row():
    lines = [TRUTH["line_items"][0], TRUTH["line_items"][2]]  # middle row dropped
    ds = score_document("inv-1", TRUTH, _pred({}, lines))
    assert (ds.lines_truth, ds.lines_pred, ds.lines_matched) == (3, 2, 2)
    assert ds.line_detail_errors == 0
    assert not ds.lines_sum_ok


def test_summary_aggregates():
    good = score_document(
        "a", TRUTH, _pred({k: v for k, v in TRUTH["fields"].items()}, TRUTH["line_items"])
    )
    s = summarize([good])
    assert s["field_accuracy"] == 1.0
    assert s["line_items"]["recall"] == 1.0
    assert s["line_items"]["docs_sum_reconciles"] == 1
    assert s["outcomes"]["hallucinated"] == 0


def test_per_trap_targets_only_relevant_fields():
    truth = {**TRUTH, "traps": ["claim_in_text"]}
    fields = dict(TRUTH["fields"], claim_number=None)  # missed the claim number in the "Re:" line
    s = summarize([score_document("a", truth, _pred(fields, TRUTH["line_items"]))])
    trap = s["per_trap"]["claim_in_text"]
    # targets claim_number + policy_number: one missed, one correctly null
    assert trap == {
        "docs": 1,
        "target_field_accuracy": 0.5,
        "target_fields": 2,
        "line_recall": None,
    }
