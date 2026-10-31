"""Score an ExtractionResult against ground truth. No API calls; used by scripts/eval.py.

Truth format (written by scripts/gen_synthetic.py):
    {"doc_type": str, "fields": {name: value|None}, "line_items": [{description, quantity, unit,
     unit_price, amount, category}], "scan": bool}
"""

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from app.schemas import ExtractionResult
from app.values import parse_number as _num

NUMERIC = {
    "subtotal",
    "tax",
    "total",
    "amount_due",
    "overhead_and_profit",
    "injuries",
    "fatalities",
}
DATES = {"date_of_loss", "estimate_date", "invoice_date"}
DATETIMES = {"alarm_time", "arrival_time"}
ENUMS = {"cause_of_loss"}
# Free text the model may reasonably paraphrase or trim: similarity threshold instead of equality
FUZZY = {
    "cause_of_loss_detail": 0.8,
    "cause_determination": 0.8,
    "area_of_origin": 0.8,
}
# Summaries: graded on how many of the source's content words survive, not on wording
SUMMARY = {"narrative_summary": 0.6}
STOPWORDS = {
    "the",
    "and",
    "with",
    "from",
    "that",
    "this",
    "were",
    "was",
    "for",
    "prior",
    "after",
    "above",
}

MONEY_TOL = 0.011

# The fields each hard-profile trap is designed to confuse (see scripts/gen_synthetic.py).
# "lines" means the trap targets line item extraction.
TRAP_TARGETS = {
    "xactimate": {"subtotal", "tax", "overhead_and_profit", "total", "lines"},
    "long_table": {"lines"},
    "math_error": {"total", "amount_due"},
    "conflicting_dates": {"date_of_loss", "estimate_date", "invoice_date"},
    "stamp": {"estimate_date", "invoice_date"},
    "label_variants": {
        "insured_name",
        "claim_number",
        "policy_number",
        "date_of_loss",
        "amount_due",
    },
    "claim_in_text": {"claim_number", "policy_number"},
    "credit_line": {"subtotal", "total", "lines"},
    "time_distractors": {"alarm_time", "arrival_time"},
    "time_format": {"alarm_time", "arrival_time"},
}


def _norm(v) -> str:
    s = str(v).casefold()
    s = re.sub(r"[.,#:;'\"]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _empty(v) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


def similarity(a, b) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _content_words(v) -> set[str]:
    return {
        w for w in re.findall(r"[a-z0-9]+", str(v).casefold()) if len(w) > 3 and w not in STOPWORDS
    }


def token_recall(truth, pred) -> float:
    """Share of the truth's content words that appear in the prediction."""
    t = _content_words(truth)
    return len(t & _content_words(pred)) / len(t) if t else 1.0


def field_matches(name: str, truth, pred) -> bool:
    if name in NUMERIC:
        t, p = _num(truth), _num(pred)
        return t is not None and p is not None and abs(t - p) <= MONEY_TOL
    if name in DATES:
        return str(pred).strip()[:10] == str(truth)[:10]
    if name in DATETIMES:
        return str(pred).strip().replace(" ", "T")[:16] == str(truth)[:16]
    if name in ENUMS:
        return _norm(pred) == _norm(truth)
    if name in SUMMARY:
        return token_recall(truth, pred) >= SUMMARY[name]
    if name in FUZZY:
        return similarity(truth, pred) >= FUZZY[name]
    return _norm(pred) == _norm(truth)


@dataclass
class FieldScore:
    name: str
    truth: object
    pred: object
    confidence: float | None
    outcome: str  # correct | wrong | missed | hallucinated

    @property
    def ok(self) -> bool:
        return self.outcome == "correct"


@dataclass
class DocScore:
    doc_id: str
    doc_type: str
    scan: bool
    doc_type_ok: bool
    traps: list[str] = field(default_factory=list)
    fields: list[FieldScore] = field(default_factory=list)
    lines_truth: int = 0
    lines_pred: int = 0
    lines_matched: int = 0
    line_detail_errors: int = 0
    lines_sum_ok: bool = True

    @property
    def field_accuracy(self) -> float:
        return sum(f.ok for f in self.fields) / len(self.fields) if self.fields else 1.0


def score_fields(truth: dict, pred: ExtractionResult) -> list[FieldScore]:
    by_name = {f.name.value: f for f in pred.fields}
    out = []
    for name, t in truth["fields"].items():
        pf = by_name.get(name)
        p = pf.value if pf else None
        conf = pf.confidence if pf else None
        if _empty(t):
            outcome = "correct" if _empty(p) else "hallucinated"
        elif _empty(p):
            outcome = "missed"
        else:
            outcome = "correct" if field_matches(name, t, p) else "wrong"
        out.append(FieldScore(name, t, p, conf, outcome))
    return out


def _amount_key(v) -> str:
    n = _num(v)
    return f"{n:.2f}" if n is not None else "?"


def score_lines(truth_items: list[dict], pred: ExtractionResult) -> tuple[int, int, bool]:
    """Align predicted to true line items by amount sequence (tolerant of a dropped or extra
    row), then check description / quantity / unit price on aligned pairs.
    Returns (matched, detail_errors, sum_ok)."""
    t_keys = [_amount_key(li["amount"]) for li in truth_items]
    p_keys = [_amount_key(li.amount) for li in pred.line_items]
    matched, detail_errors = 0, 0
    sm = SequenceMatcher(None, t_keys, p_keys, autojunk=False)
    for block in sm.get_matching_blocks():
        for k in range(block.size):
            t = truth_items[block.a + k]
            p = pred.line_items[block.b + k]
            matched += 1
            if (
                similarity(t["description"], p.description) < 0.8
                or not field_matches("total", t["quantity"], p.quantity)
                or not field_matches("total", t["unit_price"], p.unit_price)
            ):
                detail_errors += 1
    t_sum = sum(_num(li["amount"]) or 0 for li in truth_items)
    p_sum = sum(_num(li.amount) or 0 for li in pred.line_items)
    return matched, detail_errors, abs(t_sum - p_sum) <= MONEY_TOL * max(1, len(truth_items))


def score_document(doc_id: str, truth: dict, pred: ExtractionResult) -> DocScore:
    ds = DocScore(
        doc_id=doc_id,
        doc_type=truth["doc_type"],
        scan=bool(truth.get("scan")),
        traps=list(truth.get("traps", [])),
        doc_type_ok=pred.doc_type.value == truth["doc_type"],
        fields=score_fields(truth, pred),
        lines_truth=len(truth["line_items"]),
        lines_pred=len(pred.line_items),
    )
    ds.lines_matched, ds.line_detail_errors, ds.lines_sum_ok = score_lines(
        truth["line_items"], pred
    )
    return ds


def summarize(scores: list[DocScore]) -> dict:
    """Aggregate metrics for the report."""

    def acc(fs: list[FieldScore]) -> float | None:
        return round(sum(f.ok for f in fs) / len(fs), 4) if fs else None

    all_fields = [f for s in scores for f in s.fields]
    per_field: dict[str, list[FieldScore]] = {}
    for f in all_fields:
        per_field.setdefault(f.name, []).append(f)
    per_type: dict[str, list[DocScore]] = {}
    for s in scores:
        per_type.setdefault(s.doc_type, []).append(s)

    lt = sum(s.lines_truth for s in scores)
    lp = sum(s.lines_pred for s in scores)
    lm = sum(s.lines_matched for s in scores)
    hi = [f for f in all_fields if f.confidence is not None and f.confidence >= 0.9]
    lo = [f for f in all_fields if f.confidence is not None and f.confidence < 0.9]
    per_trap = {}
    for trap in sorted({t for s in scores for t in s.traps}):
        docs = [s for s in scores if trap in s.traps]
        targets = TRAP_TARGETS.get(trap, set())
        target_fields = [f for s in docs for f in s.fields if f.name in targets]
        lt_t = sum(s.lines_truth for s in docs)
        per_trap[trap] = {
            "docs": len(docs),
            "target_field_accuracy": acc(target_fields),
            "target_fields": len(target_fields),
            "line_recall": round(sum(s.lines_matched for s in docs) / lt_t, 4)
            if "lines" in targets and lt_t
            else None,
        }

    return {
        "documents": len(scores),
        "doc_type_accuracy": acc(
            [FieldScore("t", 0, 0, None, "correct" if s.doc_type_ok else "wrong") for s in scores]
        ),
        "field_accuracy": acc(all_fields),
        "field_accuracy_clean": acc([f for s in scores if not s.scan for f in s.fields]),
        "field_accuracy_scan": acc([f for s in scores if s.scan for f in s.fields]),
        "per_doc_type": {
            t: acc([f for s in ss for f in s.fields]) for t, ss in sorted(per_type.items())
        },
        "per_field": {n: acc(fs) for n, fs in sorted(per_field.items())},
        "outcomes": {
            o: sum(f.outcome == o for f in all_fields)
            for o in ("correct", "wrong", "missed", "hallucinated")
        },
        "line_items": {
            "truth": lt,
            "predicted": lp,
            "recall": round(lm / lt, 4) if lt else None,
            "precision": round(lm / lp, 4) if lp else None,
            "detail_accuracy": round(1 - sum(s.line_detail_errors for s in scores) / lm, 4)
            if lm
            else None,
            "docs_sum_reconciles": sum(s.lines_sum_ok for s in scores if s.lines_truth),
            "docs_with_lines": sum(1 for s in scores if s.lines_truth),
        },
        "per_trap": per_trap,
        "calibration": {
            "high_conf_fields": len(hi),
            "high_conf_accuracy": acc(hi),
            "low_conf_fields": len(lo),
            "low_conf_accuracy": acc(lo),
        },
    }
