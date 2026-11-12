"""Apply a reviewer's corrections to a model extraction and record what changed."""

from difflib import SequenceMatcher
from itertools import zip_longest
from typing import Any, Literal

from pydantic import BaseModel

from app.schemas import DocType, ExtractedField, ExtractionResult, FieldName, LineItem


class ReviewedLineItem(BaseModel):
    description: str
    quantity: float | None = None
    unit: str | None = None
    unit_price: float | None = None
    amount: float | None = None
    category: str | None = None


class ReviewSubmission(BaseModel):
    """The reviewer's corrections, always relative to the model output (not to an earlier draft).

    Anything left out keeps the model's value: `fields` only needs the fields that differ, and
    `line_items`, when given, replaces the whole list.
    """

    doc_type: DocType | None = None
    fields: dict[FieldName, str | None] = {}
    line_items: list[ReviewedLineItem] | None = None
    approve: bool = False


class Change(BaseModel):
    target: Literal["doc_type", "field", "line_item"]
    name: str | None = None  # field name, for target="field"
    before_index: int | None = None  # line position in the model output (None = added)
    after_index: int | None = None  # line position in the corrected list (None = removed)
    before: Any = None
    after: Any = None


def _content(item: LineItem | ReviewedLineItem) -> tuple:
    return (
        item.description,
        item.quantity,
        item.unit,
        item.unit_price,
        item.amount,
        item.category,
    )


def _plain(item: LineItem | None) -> dict | None:
    if item is None:
        return None
    return ReviewedLineItem.model_validate(item, from_attributes=True).model_dump()


def apply_review(
    result: ExtractionResult, submission: ReviewSubmission
) -> tuple[ExtractionResult, list[Change]]:
    """Return the corrected result and the list of changes from the model output.

    Corrected values get confidence 1.0 (a human entered them) and lose their source_text, which
    no longer supports the value; the page is kept so the UI can still jump to the right place.
    """
    changes: list[Change] = []

    doc_type, doc_type_confidence = result.doc_type, result.doc_type_confidence
    if submission.doc_type is not None and submission.doc_type is not result.doc_type:
        changes.append(
            Change(target="doc_type", before=result.doc_type.value, after=submission.doc_type.value)
        )
        doc_type, doc_type_confidence = submission.doc_type, 1.0

    fields: dict[FieldName, ExtractedField] = {}
    for f in result.fields:
        fields.setdefault(f.name, f)
    for name, raw in submission.fields.items():
        value = (raw or "").strip() or None
        old = fields.get(name)
        before = old.value if old else None
        if value == before:
            continue
        changes.append(Change(target="field", name=name.value, before=before, after=value))
        fields[name] = ExtractedField(
            name=name,
            value=value,
            page=old.page if old else None,
            source_text=None,
            confidence=1.0,
        )

    lines = result.line_items
    if submission.line_items is not None:
        old_lines, new_lines = result.line_items, submission.line_items
        lines = []
        matcher = SequenceMatcher(
            a=[_content(li) for li in old_lines],
            b=[_content(li) for li in new_lines],
            autojunk=False,
        )
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                lines.extend(old_lines[i1:i2])
                continue
            for i, j in zip_longest(range(i1, i2), range(j1, j2)):
                before = old_lines[i] if i is not None else None
                after = new_lines[j] if j is not None else None
                changes.append(
                    Change(
                        target="line_item",
                        before_index=i,
                        after_index=j,
                        before=_plain(before),
                        after=after.model_dump() if after else None,
                    )
                )
                if after is not None:
                    lines.append(
                        LineItem(
                            **after.model_dump(),
                            page=before.page if before else None,
                            confidence=1.0,
                        )
                    )

    corrected = ExtractionResult(
        doc_type=doc_type,
        doc_type_confidence=doc_type_confidence,
        fields=list(fields.values()),
        line_items=lines,
        notes=result.notes,
    )
    return corrected, changes
