"""SQLite persistence: one row per uploaded document, one per extraction run, one review."""

from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from uuid import uuid4

from sqlalchemy import JSON, Column
from sqlmodel import Field, Session, SQLModel, create_engine, select


class DocStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    needs_review = "needs_review"
    approved = "approved"
    failed = "failed"


def utcnow() -> datetime:
    return datetime.now(UTC)


class Document(SQLModel, table=True):
    id: str = Field(default_factory=lambda: uuid4().hex, primary_key=True)
    filename: str
    sha256: str
    size_bytes: int
    status: DocStatus = DocStatus.queued
    error: str | None = None
    # Denormalized from the latest extraction (or the review, once one exists) for the queue view
    doc_type: str | None = None
    errors: int = 0
    warnings: int = 0
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class ExtractionRow(SQLModel, table=True):
    """One model run. Re-extracting adds a row; the newest one is current."""

    __tablename__ = "extraction"

    id: int | None = Field(default=None, primary_key=True)
    document_id: str = Field(foreign_key="document.id", index=True)
    result: dict = Field(sa_column=Column(JSON, nullable=False))  # ExtractionResult
    validation: dict = Field(sa_column=Column(JSON, nullable=False))  # ValidationReport
    model: str
    effort: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    message_id: str
    created_at: datetime = Field(default_factory=utcnow)


class Review(SQLModel, table=True):
    """The reviewer's corrected data for a document, as a draft or approved."""

    id: int | None = Field(default=None, primary_key=True)
    document_id: str = Field(foreign_key="document.id", unique=True)
    extraction_id: int = Field(foreign_key="extraction.id")
    data: dict = Field(sa_column=Column(JSON, nullable=False))  # corrected ExtractionResult
    changes: list = Field(sa_column=Column(JSON, nullable=False))  # diff vs. the model output
    validation: dict = Field(sa_column=Column(JSON, nullable=False))  # rules re-run on `data`
    approved_at: datetime | None = None
    updated_at: datetime = Field(default_factory=utcnow)


def make_engine(db_path: Path):
    return create_engine(
        f"sqlite:///{db_path.as_posix()}", connect_args={"check_same_thread": False}
    )


def init_db(engine) -> None:
    SQLModel.metadata.create_all(engine)


def latest_extraction(session: Session, document_id: str) -> ExtractionRow | None:
    return session.exec(
        select(ExtractionRow)
        .where(ExtractionRow.document_id == document_id)
        .order_by(ExtractionRow.id.desc())
    ).first()


def get_review(session: Session, document_id: str) -> Review | None:
    return session.exec(select(Review).where(Review.document_id == document_id)).first()


def fail_interrupted(engine) -> int:
    """Mark documents left queued/processing by a previous server run as failed.

    They are not re-run automatically: every extraction is a paid API call, so a retry has to be
    asked for.
    """
    with Session(engine) as session:
        stuck = session.exec(
            select(Document).where(Document.status.in_([DocStatus.queued, DocStatus.processing]))
        ).all()
        for doc in stuck:
            doc.status = DocStatus.failed
            doc.error = "The server stopped before extraction finished. Re-extract to retry."
            doc.updated_at = utcnow()
        session.commit()
        return len(stuck)
