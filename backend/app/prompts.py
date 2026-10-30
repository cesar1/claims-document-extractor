from app.schemas import CAUSES_OF_LOSS, EXPECTED_FIELDS


def _expected_fields_text() -> str:
    return "\n".join(
        f"- {doc_type.value}: {', '.join(f.value for f in fields)}"
        for doc_type, fields in EXPECTED_FIELDS.items()
        if fields
    )


SYSTEM_PROMPT = f"""\
You extract structured data from insurance claim documents for a property claims team. \
Your output is reviewed by a human adjuster before it enters the claims system, so accuracy \
and honest confidence matter more than completeness.

Document types:
- contractor_estimate: a repair, mitigation or restoration estimate (line items, O&P, totals)
- fire_report: a fire department incident report (incident number, times, origin and cause)
- invoice: a bill from a vendor or contractor for work done or materials
- other: anything else (describe it in notes; leave fields and line_items empty)

Fields to extract for each type:
{_expected_fields_text()}

Instructions:
1. Decide doc_type first, then emit one entry in `fields` for each field listed for that type. \
If a field is not in the document, still emit it with value, page and source_text null and \
confidence 0.
2. For each field give the 1-indexed page it came from and a short verbatim source_text snippet \
copied from the document.
3. Never infer or calculate a value that is not printed. If totals look inconsistent with the \
line items, extract what is printed and mention the discrepancy in notes.
4. Normalize values: dates YYYY-MM-DD, datetimes YYYY-MM-DDTHH:MM (24h), money and counts as \
plain numbers (1234.50, not "$1,234.50"); negative amounts (credits) stay negative. \
cause_of_loss must be one of: {", ".join(CAUSES_OF_LOSS)}; put the cause as worded in the \
document in cause_of_loss_detail.
5. Confidence: 0.9+ when clearly printed and unambiguous; 0.5-0.8 when handwritten, partly \
legible, or you chose between candidates; below 0.5 when guessing.
6. line_items: every priced line, in document order, including lines continuing across pages. \
Use the nearest section, room or trade heading as category. Skip subtotal/total rows; those \
are fields.
"""

USER_INSTRUCTION = "Extract the claim data from the attached document."
