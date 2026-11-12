"""HTTP API: upload claim PDFs, list the review queue, save reviews, export approved data.

Run from backend/:  python -m uvicorn app.main:create_app --factory --reload
Uploading or re-extracting a document makes a paid Claude call (about $0.10 per document).
"""

import hashlib
import os
import re
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import AfterValidator, BaseModel, ConfigDict
from sqlmodel import Session, select

from app.db import (
    DocStatus,
    Document,
    Review,
    fail_interrupted,
    get_review,
    init_db,
    latest_extraction,
    make_engine,
    utcnow,
)
from app.extract import extract_pdf
from app.pipeline import Extractor, process_document
from app.review import Change, ReviewedLineItem, ReviewSubmission, apply_review
from app.schemas import ExtractionResult
from app.storage import Storage
from app.validation import Flag, Severity, ValidationReport, validate

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # the API takes 32 MB per request, and base64 adds a third

# SQLite hands datetimes back without a timezone; they are stored as UTC, so say so in responses.
UtcDatetime = Annotated[
    datetime, AfterValidator(lambda d: d if d.tzinfo else d.replace(tzinfo=UTC))
]


class DocumentSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    size_bytes: int
    status: DocStatus
    error: str | None
    doc_type: str | None
    errors: int
    warnings: int
    created_at: UtcDatetime
    updated_at: UtcDatetime


class ExtractionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    result: ExtractionResult
    validation: ValidationReport
    model: str
    effort: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    created_at: UtcDatetime


class ReviewOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    extraction_id: int
    data: ExtractionResult
    changes: list[Change]
    validation: ValidationReport
    approved_at: UtcDatetime | None
    updated_at: UtcDatetime


class DocumentDetail(DocumentSummary):
    pdf_url: str
    extraction: ExtractionOut | None
    review: ReviewOut | None


class ExportSource(BaseModel):
    model: str
    message_id: str
    extraction_id: int


class ExportPayload(BaseModel):
    """The system-ready record: reviewed values only, without per-field provenance."""

    document_id: str
    filename: str
    sha256: str
    doc_type: str
    fields: dict[str, str | None]
    line_items: list[ReviewedLineItem]
    open_flags: list[Flag]  # errors/warnings still true of the approved data
    corrections: int
    approved_at: UtcDatetime
    source: ExportSource


def create_app(data_dir: Path | None = None, extractor: Extractor = extract_pdf) -> FastAPI:
    data_dir = Path(data_dir or os.getenv("EXTRACTOR_DATA_DIR") or DEFAULT_DATA_DIR)
    storage = Storage(data_dir / "uploads")
    engine = make_engine(data_dir / "extractor.db")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        init_db(engine)
        fail_interrupted(engine)
        yield
        engine.dispose()

    app = FastAPI(title="Claims Document Extractor", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("EXTRACTOR_CORS_ORIGINS", "http://localhost:5173").split(","),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_session():
        with Session(engine) as session:
            yield session

    SessionDep = Annotated[Session, Depends(get_session)]

    def get_document(doc_id: str, session: SessionDep) -> Document:
        doc = session.get(Document, doc_id)
        if doc is None:
            raise HTTPException(404, "Document not found")
        return doc

    DocumentDep = Annotated[Document, Depends(get_document)]

    def detail(session: Session, doc: Document) -> DocumentDetail:
        extraction = latest_extraction(session, doc.id)
        review = get_review(session, doc.id)
        return DocumentDetail(
            **DocumentSummary.model_validate(doc).model_dump(),
            pdf_url=f"/documents/{doc.id}/pdf",
            extraction=ExtractionOut.model_validate(extraction) if extraction else None,
            review=ReviewOut.model_validate(review) if review else None,
        )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/documents", status_code=201, response_model=DocumentSummary)
    def upload_document(
        file: UploadFile, background: BackgroundTasks, session: SessionDep, response: Response
    ):
        """Store a PDF and queue it for extraction. A file that was already uploaded (same
        bytes) is not extracted again: the existing document comes back with status 200."""
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"PDF is larger than {MAX_UPLOAD_BYTES // 2**20} MB")
        if b"%PDF-" not in data[:1024]:
            raise HTTPException(415, "Only PDF files are accepted")
        sha256 = hashlib.sha256(data).hexdigest()
        existing = session.exec(
            select(Document).where(Document.sha256 == sha256).order_by(Document.created_at)
        ).first()
        if existing is not None:
            response.status_code = 200
            return existing
        doc = Document(
            filename=re.split(r"[\\/]", file.filename or "document.pdf")[-1][:200],
            sha256=sha256,
            size_bytes=len(data),
        )
        storage.save(doc.id, data)
        session.add(doc)
        session.commit()
        session.refresh(doc)
        background.add_task(process_document, engine, storage, extractor, doc.id)
        return doc

    @app.get("/documents", response_model=list[DocumentSummary])
    def list_documents(session: SessionDep, status: DocStatus | None = None):
        query = select(Document).order_by(Document.created_at.desc())
        if status is not None:
            query = query.where(Document.status == status)
        return session.exec(query).all()

    @app.get("/documents/{doc_id}", response_model=DocumentDetail)
    def read_document(doc: DocumentDep, session: SessionDep):
        return detail(session, doc)

    @app.get("/documents/{doc_id}/pdf")
    def read_pdf(doc: DocumentDep):
        return FileResponse(
            storage.path(doc.id),
            media_type="application/pdf",
            filename=doc.filename,
            content_disposition_type="inline",
        )

    @app.put("/documents/{doc_id}/review", response_model=DocumentDetail)
    def save_review(doc: DocumentDep, submission: ReviewSubmission, session: SessionDep):
        """Save corrections as a draft, or approve them. Saving an approved document with
        `approve: false` reopens it."""
        if doc.status not in (DocStatus.needs_review, DocStatus.approved):
            raise HTTPException(409, f"Document is {doc.status.value}; there is nothing to review")
        extraction = latest_extraction(session, doc.id)
        corrected, changes = apply_review(
            ExtractionResult.model_validate(extraction.result), submission
        )
        # An approved document can still carry errors: a total that is misprinted on the page
        # stays wrong however carefully it is read. They travel with the export as open_flags.
        report = validate(corrected)
        now = utcnow()

        review = get_review(session, doc.id) or Review(
            document_id=doc.id, extraction_id=extraction.id, data={}, changes=[], validation={}
        )
        review.extraction_id = extraction.id
        review.data = corrected.model_dump(mode="json")
        review.changes = [c.model_dump(mode="json") for c in changes]
        review.validation = report.model_dump(mode="json")
        review.approved_at = now if submission.approve else None
        review.updated_at = now
        session.add(review)

        doc.status = DocStatus.approved if submission.approve else DocStatus.needs_review
        doc.doc_type = corrected.doc_type.value
        doc.errors, doc.warnings = report.errors, report.warnings
        doc.updated_at = now
        session.commit()
        return detail(session, doc)

    @app.post("/documents/{doc_id}/reextract", status_code=202, response_model=DocumentSummary)
    def reextract(doc: DocumentDep, background: BackgroundTasks, session: SessionDep):
        """Run the model again (a new paid call). Any draft review is discarded."""
        if doc.status is DocStatus.approved:
            raise HTTPException(409, "Document is approved; reopen it before re-extracting")
        if doc.status in (DocStatus.queued, DocStatus.processing):
            raise HTTPException(409, "Extraction is already in progress")
        review = get_review(session, doc.id)
        if review is not None:
            session.delete(review)
        doc.status = DocStatus.queued
        doc.error = None
        doc.updated_at = utcnow()
        session.commit()
        session.refresh(doc)
        background.add_task(process_document, engine, storage, extractor, doc.id)
        return doc

    @app.get("/documents/{doc_id}/export", response_model=ExportPayload)
    def export_document(doc: DocumentDep, session: SessionDep):
        review = get_review(session, doc.id)
        if doc.status is not DocStatus.approved or review is None:
            raise HTTPException(409, "Only approved documents can be exported")
        extraction = latest_extraction(session, doc.id)
        data = ExtractionResult.model_validate(review.data)
        report = ValidationReport.model_validate(review.validation)
        return ExportPayload(
            document_id=doc.id,
            filename=doc.filename,
            sha256=doc.sha256,
            doc_type=data.doc_type.value,
            fields={f.name.value: f.value for f in data.fields},
            line_items=[
                ReviewedLineItem.model_validate(li, from_attributes=True) for li in data.line_items
            ],
            open_flags=[f for f in report.flags if f.severity is not Severity.info],
            corrections=len(review.changes),
            approved_at=review.approved_at,
            source=ExportSource(
                model=extraction.model,
                message_id=extraction.message_id,
                extraction_id=extraction.id,
            ),
        )

    return app
