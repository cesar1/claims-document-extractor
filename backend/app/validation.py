"""Deterministic checks on an ExtractionResult, run after extraction and before human review.

The extraction prompt deliberately copies what is printed and never computes; this module is
where arithmetic, dates and completeness get checked. Each finding is a Flag the review UI can
attach to the fields it names. Severity:
    error    the document is internally inconsistent or a value is unusable
    warning  a human should look (missing data, low confidence, odd but possible)
    info     context for the reviewer; does not by itself require review
"""

from datetime import date
from enum import Enum

from pydantic import BaseModel

from app.schemas import CAUSES_OF_LOSS, DocType, ExtractionResult, FieldName
from app.values import parse_date, parse_datetime, parse_number

# Below this a field needs a human look. Chosen from the eval: 0.9 flagged over half the documents
# for fields that were all correct; the printed-math-error totals it alone caught at 0.80-0.85 are
# caught deterministically by totals_mismatch.
CONFIDENCE_THRESHOLD = 0.8
NUMERIC_FIELDS = {
    FieldName.subtotal,
    FieldName.tax,
    FieldName.overhead_and_profit,
    FieldName.total,
    FieldName.amount_due,
    FieldName.injuries,
    FieldName.fatalities,
}
DATE_FIELDS = {FieldName.date_of_loss, FieldName.estimate_date, FieldName.invoice_date}
DATETIME_FIELDS = {FieldName.alarm_time, FieldName.arrival_time}

# Fields the downstream claims system can't do without, per doc type
REQUIRED: dict[DocType, list[FieldName]] = {
    DocType.contractor_estimate: [
        FieldName.contractor_name,
        FieldName.estimate_date,
        FieldName.total,
    ],
    DocType.invoice: [
        FieldName.vendor_name,
        FieldName.invoice_number,
        FieldName.invoice_date,
        FieldName.total,
    ],
    DocType.fire_report: [
        FieldName.incident_number,
        FieldName.fire_department,
        FieldName.date_of_loss,
        FieldName.alarm_time,
    ],
    DocType.other: [],
}
DOC_DATE = {
    DocType.contractor_estimate: FieldName.estimate_date,
    DocType.invoice: FieldName.invoice_date,
}


class Severity(str, Enum):
    error = "error"
    warning = "warning"
    info = "info"


class Flag(BaseModel):
    rule: str
    severity: Severity
    message: str
    fields: list[str] = []
    line_indices: list[int] = []


class ValidationReport(BaseModel):
    flags: list[Flag]
    needs_review: bool
    errors: int
    warnings: int


def money_tolerance(n_terms: int = 1) -> float:
    """Allow a cent of rounding per summed term, never less than 5 cents."""
    return max(0.05, 0.01 * n_terms)


def _close(a: float, b: float, n_terms: int = 1) -> bool:
    return abs(a - b) <= money_tolerance(n_terms)


def _fmt(x: float) -> str:
    return f"${x:,.2f}"


class _Checker:
    def __init__(self, result: ExtractionResult, today: date, threshold: float):
        self.r = result
        self.today = today
        self.threshold = threshold
        self.flags: list[Flag] = []

    def flag(self, rule: str, severity: Severity, message: str, fields=(), lines=()) -> None:
        self.flags.append(
            Flag(
                rule=rule,
                severity=severity,
                message=message,
                fields=[f.value if isinstance(f, FieldName) else f for f in fields],
                line_indices=list(lines),
            )
        )

    def num(self, name: FieldName) -> float | None:
        return parse_number(self.r.value(name))

    def run(self) -> list[Flag]:
        self.check_doc_type()
        self.check_parseable()
        self.check_required()
        if self.r.doc_type in (DocType.contractor_estimate, DocType.invoice):
            self.check_totals()
            self.check_line_items()
        self.check_dates()
        if self.r.doc_type is DocType.fire_report:
            self.check_fire_times()
        self.check_confidence()
        if self.r.notes:
            self.flag(
                "extractor_notes", Severity.info, "The extractor left notes for the reviewer."
            )
        return self.flags

    def check_doc_type(self) -> None:
        if self.r.doc_type is DocType.other:
            self.flag(
                "doc_type_other",
                Severity.warning,
                "Not recognized as an estimate, invoice or fire report.",
            )
        elif self.r.doc_type_confidence < self.threshold:
            self.flag(
                "doc_type_low_confidence",
                Severity.warning,
                f"Document type '{self.r.doc_type.value}' at confidence {self.r.doc_type_confidence:.2f}.",
            )

    def check_parseable(self) -> None:
        for f in self.r.fields:
            if f.value is None:
                continue
            bad = (
                (f.name in NUMERIC_FIELDS and parse_number(f.value) is None)
                or (f.name in DATE_FIELDS and parse_date(f.value) is None)
                or (f.name in DATETIME_FIELDS and parse_datetime(f.value) is None)
            )
            if bad:
                self.flag(
                    "unparseable",
                    Severity.error,
                    f"Can't read {f.name.value} value {f.value!r}.",
                    [f.name],
                )
        cause = self.r.value(FieldName.cause_of_loss)
        if cause is not None and cause not in CAUSES_OF_LOSS:
            self.flag(
                "cause_not_in_list",
                Severity.warning,
                f"Cause of loss {cause!r} is not one of the allowed values.",
                [FieldName.cause_of_loss],
            )

    def check_required(self) -> None:
        for name in REQUIRED[self.r.doc_type]:
            if self.r.value(name) is None:
                self.flag("required_missing", Severity.warning, f"{name.value} is missing.", [name])
        if self.r.doc_type in DOC_DATE and self.r.value(FieldName.claim_number) is None:
            self.flag(
                "claim_unmatched",
                Severity.warning,
                "No claim number; this document must be matched to a claim manually.",
                [FieldName.claim_number],
            )

    def check_totals(self) -> None:
        sub, tax, op, total = (
            self.num(FieldName.subtotal),
            self.num(FieldName.tax),
            self.num(FieldName.overhead_and_profit),
            self.num(FieldName.total),
        )
        if sub is not None and total is not None:
            t0, o0 = tax or 0.0, op or 0.0
            # Layouts differ on whether the printed subtotal already includes tax (Xactimate does)
            candidates = {sub + t0 + o0, sub + o0, sub + t0}
            if not any(_close(c, total, 3) for c in candidates):
                parts = " + ".join(
                    f"{label} {_fmt(v)}"
                    for label, v in (("subtotal", sub), ("tax", tax), ("O&P", op))
                    if v is not None
                )
                self.flag(
                    "totals_mismatch",
                    Severity.error,
                    f"Printed total {_fmt(total)} doesn't match its components ({parts}).",
                    [
                        FieldName.total,
                        FieldName.subtotal,
                        FieldName.tax,
                        FieldName.overhead_and_profit,
                    ],
                )
        due = self.num(FieldName.amount_due)
        if due is not None and total is not None:
            if due > total + money_tolerance():
                self.flag(
                    "amount_due_exceeds_total",
                    Severity.error,
                    f"Amount due {_fmt(due)} is more than the total {_fmt(total)}.",
                    [FieldName.amount_due, FieldName.total],
                )
            elif not _close(due, total):
                self.flag(
                    "credits_applied",
                    Severity.info,
                    f"Amount due is {_fmt(total - due)} less than the total (deposit or credit applied).",
                    [FieldName.amount_due],
                )

    def check_line_items(self) -> None:
        lines = self.r.line_items
        if not lines:
            return
        missing = [i for i, li in enumerate(lines) if li.amount is None]
        if missing:
            self.flag(
                "line_amount_missing",
                Severity.warning,
                f"{len(missing)} line item(s) have no amount.",
                ["line_items"],
                missing,
            )
        line_sum = sum(li.amount for li in lines if li.amount is not None)
        n = len(lines) - len(missing)
        sub, tax, total = (
            self.num(FieldName.subtotal),
            self.num(FieldName.tax),
            self.num(FieldName.total),
        )

        basis = None
        if sub is not None and _close(line_sum, sub, n):
            basis = "subtotal"
        elif total is not None and _close(line_sum, total, n):
            basis = "total"
        elif sub is not None and tax is not None and _close(line_sum, sub - tax, n):
            basis = "subtotal_before_tax"
        if basis == "total":
            self.flag(
                "lines_include_tax_op",
                Severity.info,
                "Line amounts reconcile to the total, so they include tax and/or O&P (Xactimate style).",
                ["line_items", FieldName.total],
            )
        elif basis is None and (sub is not None or total is not None):
            targets = ", ".join(
                f"{k} {_fmt(v)}" for k, v in (("subtotal", sub), ("total", total)) if v is not None
            )
            self.flag(
                "lines_mismatch",
                Severity.error,
                f"{n} line items sum to {_fmt(line_sum)}, which matches neither {targets}. "
                "A line may be missing, duplicated or misread.",
                ["line_items", FieldName.subtotal, FieldName.total],
            )

        # Quantity x unit price vs amount. When lines reconcile to the total, an amount above
        # qty x price is explained by per-line tax/O&P; only amounts below it are suspicious.
        # Nearly every line off means a pricing pattern, not misreads.
        checkable = [
            (i, li)
            for i, li in enumerate(lines)
            if None not in (li.quantity, li.unit_price, li.amount)
        ]
        off = []
        for i, li in checkable:
            diff = li.amount - li.quantity * li.unit_price
            tol = max(0.02, 0.005 * abs(li.amount))
            if abs(diff) > tol and not (basis == "total" and diff > 0):
                off.append(i)
        if checkable and off:
            if len(off) / len(checkable) >= 0.8:
                self.flag(
                    "line_pricing_pattern",
                    Severity.info,
                    f"Quantity x unit price differs from the amount on {len(off)} of {len(checkable)} lines; "
                    "the amounts likely include per-line tax/O&P or use another price column.",
                    ["line_items"],
                )
            else:
                self.flag(
                    "line_math",
                    Severity.warning,
                    f"Quantity x unit price doesn't equal the amount on {len(off)} of {len(checkable)} lines; "
                    "check those lines for misreads.",
                    ["line_items"],
                    off[:25],
                )

        low = [i for i, li in enumerate(lines) if li.confidence < self.threshold]
        if low:
            self.flag(
                "line_low_confidence",
                Severity.warning,
                f"{len(low)} line item(s) below confidence {self.threshold:.2f}.",
                ["line_items"],
                low[:25],
            )

    def check_dates(self) -> None:
        loss = parse_date(self.r.value(FieldName.date_of_loss))
        if loss is not None:
            if loss > self.today:
                self.flag(
                    "loss_in_future",
                    Severity.error,
                    f"Date of loss {loss} is in the future.",
                    [FieldName.date_of_loss],
                )
            elif (self.today - loss).days > 3 * 365:
                self.flag(
                    "loss_old",
                    Severity.warning,
                    f"Date of loss {loss} is more than 3 years ago.",
                    [FieldName.date_of_loss],
                )
        doc_field = DOC_DATE.get(self.r.doc_type)
        doc_date = parse_date(self.r.value(doc_field)) if doc_field else None
        if doc_date is not None:
            if loss is not None and doc_date < loss:
                self.flag(
                    "dated_before_loss",
                    Severity.error,
                    f"{doc_field.value} {doc_date} is before the date of loss {loss}.",
                    [doc_field, FieldName.date_of_loss],
                )
            if doc_date > self.today:
                self.flag(
                    "doc_date_in_future",
                    Severity.warning,
                    f"{doc_field.value} {doc_date} is in the future.",
                    [doc_field],
                )

    def check_fire_times(self) -> None:
        alarm = parse_datetime(self.r.value(FieldName.alarm_time))
        arrival = parse_datetime(self.r.value(FieldName.arrival_time))
        if alarm and arrival:
            minutes = (arrival - alarm).total_seconds() / 60
            if minutes < 0:
                self.flag(
                    "arrival_before_alarm",
                    Severity.error,
                    f"Arrival {arrival:%H:%M} is before the alarm {alarm:%H:%M}.",
                    [FieldName.alarm_time, FieldName.arrival_time],
                )
            elif minutes > 120:
                self.flag(
                    "slow_response",
                    Severity.warning,
                    f"Arrival is {minutes:.0f} minutes after the alarm; check the times.",
                    [FieldName.alarm_time, FieldName.arrival_time],
                )
        loss = parse_date(self.r.value(FieldName.date_of_loss))
        if alarm and loss and alarm.date() != loss:
            self.flag(
                "alarm_date_differs",
                Severity.warning,
                f"Alarm date {alarm.date()} differs from the incident date {loss}.",
                [FieldName.alarm_time, FieldName.date_of_loss],
            )

    def check_confidence(self) -> None:
        fire = self.r.doc_type is DocType.fire_report
        if fire and self.r.value(FieldName.insured_name) is not None:
            self.flag(
                "owner_may_not_be_insured",
                Severity.info,
                "Fire reports name the owner/occupant, who may not be the insured; confirm when matching to the claim.",
                [FieldName.insured_name],
            )
        for f in self.r.fields:
            if fire and f.name is FieldName.insured_name:
                continue  # uncertainty here is about who the person is, covered by the flag above
            if f.value is not None and f.confidence < self.threshold:
                self.flag(
                    "low_confidence",
                    Severity.warning,
                    f"{f.name.value} extracted at confidence {f.confidence:.2f}.",
                    [f.name],
                )


def validate(
    result: ExtractionResult,
    today: date | None = None,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> ValidationReport:
    flags = _Checker(result, today or date.today(), confidence_threshold).run()
    errors = sum(f.severity is Severity.error for f in flags)
    warnings = sum(f.severity is Severity.warning for f in flags)
    return ValidationReport(
        flags=flags, needs_review=bool(errors or warnings), errors=errors, warnings=warnings
    )
