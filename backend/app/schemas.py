"""Output schema for claim document extraction.

The schema Claude fills is deliberately flat: a list of named fields plus a list of line items.
A nested per-doc-type schema with provenance on every value compiles to a structured-output
grammar that is too large (API 400), and costs far more output tokens on long estimates.

Each header field carries provenance (page + verbatim source text) and a confidence score, so
the review UI can jump to the evidence and highlight weak fields. Native citations can't be
combined with structured outputs, so provenance lives here.
"""

from enum import Enum

from pydantic import BaseModel, Field


class DocType(str, Enum):
    contractor_estimate = "contractor_estimate"
    fire_report = "fire_report"
    invoice = "invoice"
    other = "other"


class FieldName(str, Enum):
    # common to all claim documents
    claim_number = "claim_number"
    policy_number = "policy_number"
    insured_name = "insured_name"
    property_address = "property_address"
    date_of_loss = "date_of_loss"
    cause_of_loss = "cause_of_loss"
    cause_of_loss_detail = "cause_of_loss_detail"
    # contractor estimate
    contractor_name = "contractor_name"
    contractor_license = "contractor_license"
    estimate_number = "estimate_number"
    estimate_date = "estimate_date"
    overhead_and_profit = "overhead_and_profit"
    # fire report
    incident_number = "incident_number"
    fire_department = "fire_department"
    alarm_time = "alarm_time"
    arrival_time = "arrival_time"
    area_of_origin = "area_of_origin"
    cause_determination = "cause_determination"
    injuries = "injuries"
    fatalities = "fatalities"
    narrative_summary = "narrative_summary"
    # invoice
    vendor_name = "vendor_name"
    invoice_number = "invoice_number"
    invoice_date = "invoice_date"
    amount_due = "amount_due"
    # totals (estimate and invoice)
    subtotal = "subtotal"
    tax = "tax"
    total = "total"


# Which fields each doc type is expected to have; drives prompt guidance and "missing" flags.
COMMON_FIELDS = [
    FieldName.claim_number,
    FieldName.policy_number,
    FieldName.insured_name,
    FieldName.property_address,
    FieldName.date_of_loss,
    FieldName.cause_of_loss,
    FieldName.cause_of_loss_detail,
]
EXPECTED_FIELDS: dict[DocType, list[FieldName]] = {
    DocType.contractor_estimate: COMMON_FIELDS
    + [
        FieldName.contractor_name,
        FieldName.contractor_license,
        FieldName.estimate_number,
        FieldName.estimate_date,
        FieldName.subtotal,
        FieldName.tax,
        FieldName.overhead_and_profit,
        FieldName.total,
    ],
    DocType.fire_report: COMMON_FIELDS
    + [
        FieldName.incident_number,
        FieldName.fire_department,
        FieldName.alarm_time,
        FieldName.arrival_time,
        FieldName.area_of_origin,
        FieldName.cause_determination,
        FieldName.injuries,
        FieldName.fatalities,
        FieldName.narrative_summary,
    ],
    DocType.invoice: COMMON_FIELDS
    + [
        FieldName.vendor_name,
        FieldName.invoice_number,
        FieldName.invoice_date,
        FieldName.subtotal,
        FieldName.tax,
        FieldName.total,
        FieldName.amount_due,
    ],
    DocType.other: [],
}

CAUSES_OF_LOSS = [
    "fire",
    "smoke",
    "water",
    "wind",
    "hail",
    "theft",
    "vandalism",
    "other",
    "unknown",
]


class ExtractedField(BaseModel):
    name: FieldName
    value: str | None = Field(description="Normalized value as a string; null if not present")
    page: int | None = Field(description="1-indexed page where the value appears")
    source_text: str | None = Field(description="Short verbatim snippet supporting the value")
    confidence: float = Field(description="0.0-1.0; how sure you are the value is correct")


class LineItem(BaseModel):
    description: str
    quantity: float | None
    unit: str | None
    unit_price: float | None
    amount: float | None = Field(description="Line total in dollars as printed")
    category: str | None = Field(description="Section/room/trade heading the line sits under")
    page: int | None
    confidence: float


class ExtractionResult(BaseModel):
    doc_type: DocType
    doc_type_confidence: float
    fields: list[ExtractedField]
    line_items: list[LineItem]
    notes: str | None = Field(
        description="Anything a reviewer should know: illegible areas, conflicting values, etc."
    )

    def field(self, name: FieldName) -> ExtractedField | None:
        return next((f for f in self.fields if f.name == name), None)

    def value(self, name: FieldName) -> str | None:
        f = self.field(name)
        return f.value if f else None
