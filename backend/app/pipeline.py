"""Background job for one queued document: extract with Claude, validate, store the outcome."""

import logging
from collections.abc import Callable

from sqlmodel import Session

from app import extract
from app.db import DocStatus, Document, ExtractionRow, utcnow
from app.storage import Storage
from app.validation import validate

log = logging.getLogger(__name__)

Extractor = Callable[[bytes], extract.Extraction]


def _fail(engine, doc_id: str, error: str) -> None:
    with Session(engine) as session:
        doc = session.get(Document, doc_id)
        doc.status = DocStatus.failed
        doc.error = error[:500]
        doc.updated_at = utcnow()
        session.commit()


def process_document(engine, storage: Storage, extractor: Extractor, doc_id: str) -> None:
    with Session(engine) as session:
        doc = session.get(Document, doc_id)
        if doc is None or doc.status is not DocStatus.queued:
            return
        doc.status = DocStatus.processing
        doc.updated_at = utcnow()
        session.commit()

    # Any failure has to land in the row; otherwise the document sits in "processing" forever.
    try:
        extraction = extractor(storage.read(doc_id))
        report = validate(extraction.result)
    except Exception as e:
        log.exception("Extraction failed for document %s", doc_id)
        _fail(engine, doc_id, f"{type(e).__name__}: {e}")
        return

    with Session(engine) as session:
        doc = session.get(Document, doc_id)
        session.add(
            ExtractionRow(
                document_id=doc_id,
                result=extraction.result.model_dump(mode="json"),
                validation=report.model_dump(mode="json"),
                model=extraction.model,
                effort=extract.EFFORT,
                input_tokens=extraction.input_tokens,
                output_tokens=extraction.output_tokens,
                cost_usd=extraction.cost_usd,
                message_id=extraction.message_id,
            )
        )
        doc.status = DocStatus.needs_review
        doc.error = None
        doc.doc_type = extraction.result.doc_type.value
        doc.errors = report.errors
        doc.warnings = report.warnings
        doc.updated_at = utcnow()
        session.commit()
