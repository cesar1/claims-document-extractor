"""Classify and extract a claim PDF in one Claude call.

Usage:  python -m app.extract path/to/doc.pdf [more.pdf ...]
"""

import base64
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from app.prompts import SYSTEM_PROMPT, USER_INSTRUCTION
from app.schemas import ExtractionResult
from app.validation import validate

load_dotenv()

MODEL = os.getenv("EXTRACTOR_MODEL", "claude-opus-5-5")
EFFORT = os.getenv("EXTRACTOR_EFFORT", "medium")
MAX_TOKENS = 64000  # long estimates with per-field provenance get big; streaming avoids timeouts
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# $/MTok for claude-opus-5-5; cache writes are 1.25x input
PRICE_INPUT = 4.00
PRICE_OUTPUT = 20.00
PRICE_CACHE_READ = 0.20
PRICE_CACHE_WRITE = 5.00


class ExtractionError(Exception):
    pass


@dataclass
class Extraction:
    result: ExtractionResult
    input_tokens: int
    output_tokens: int
    cost_usd: float
    message_id: str
    model: str  # the model that answered; differs from MODEL when a server-side fallback ran


def cost_usd(usage, batch: bool = False) -> float:
    cache_read = usage.cache_read_input_tokens or 0
    cache_write = usage.cache_creation_input_tokens or 0
    full = (
        usage.input_tokens * PRICE_INPUT
        + cache_read * PRICE_CACHE_READ
        + cache_write * PRICE_CACHE_WRITE
        + usage.output_tokens * PRICE_OUTPUT
    ) / 1_000_000
    return full * 0.5 if batch else full


def _base_request(pdf_bytes: bytes) -> dict:
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": base64.standard_b64encode(pdf_bytes).decode("ascii"),
                        },
                    },
                    {"type": "text", "text": USER_INSTRUCTION},
                ],
            }
        ],
    }


def build_request(pdf_bytes: bytes) -> dict:
    """Kwargs for the live streaming path (SDK parses into ExtractionResult)."""
    return {
        **_base_request(pdf_bytes),
        "output_config": {"effort": EFFORT},
        "output_format": ExtractionResult,
    }


def build_batch_params(pdf_bytes: bytes) -> dict:
    """Params for one Message Batches request. Batches reject `fallbacks` and take the raw
    JSON schema rather than a Pydantic type."""
    return {
        **_base_request(pdf_bytes),
        "output_config": {
            "effort": EFFORT,
            "format": {
                "type": "json_schema",
                "schema": anthropic.transform_schema(ExtractionResult),
            },
        },
    }


def check_message(message) -> None:
    """Raise ExtractionError for responses that carry no usable extraction."""
    if message.stop_reason == "refusal":
        raise ExtractionError(f"Model declined the document: {message.stop_details}")
    if message.stop_reason == "max_tokens":
        raise ExtractionError("Output hit max_tokens before the extraction was complete")


def parse_message_text(message) -> ExtractionResult:
    """Parse a non-SDK-parsed response (e.g. a batch result) into ExtractionResult."""
    check_message(message)
    text = next((b.text for b in message.content if b.type == "text"), None)
    if text is None:
        raise ExtractionError(f"No text output (stop_reason={message.stop_reason})")
    return ExtractionResult.model_validate_json(text)


def extract_pdf(pdf_bytes: bytes, client: anthropic.Anthropic | None = None) -> Extraction:
    client = client or anthropic.Anthropic()
    with client.beta.messages.stream(
        **build_request(pdf_bytes),
        fallbacks="default",
        betas=[FALLBACK_BETA],
    ) as stream:
        message = stream.get_final_message()

    check_message(message)
    if message.parsed_output is None:
        raise ExtractionError(f"No structured output (stop_reason={message.stop_reason})")

    return Extraction(
        result=message.parsed_output,
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
        cost_usd=cost_usd(message.usage),
        message_id=message.id,
        model=message.model,
    )


def main(paths: list[str]) -> int:
    if not paths:
        print(__doc__)
        return 2
    total = 0.0
    failed = 0
    for p in paths:
        try:
            ex = extract_pdf(Path(p).read_bytes())
        except (ExtractionError, anthropic.APIError) as e:
            print(f"[FAIL] {p}: {e}", file=sys.stderr)
            failed += 1
            continue
        total += ex.cost_usd
        print(json.dumps(ex.result.model_dump(mode="json"), indent=2))
        print(
            f"[ok] {p}: {ex.result.doc_type.value} "
            f"in={ex.input_tokens} out={ex.output_tokens} ${ex.cost_usd:.4f}",
            file=sys.stderr,
        )
        report = validate(ex.result)
        status = "NEEDS REVIEW" if report.needs_review else "clean"
        print(
            f"  validation: {status} ({report.errors} errors, {report.warnings} warnings)",
            file=sys.stderr,
        )
        for f in report.flags:
            if f.severity.value != "info":
                print(f"    [{f.severity.value}] {f.rule}: {f.message}", file=sys.stderr)
    print(f"[total] ${total:.4f}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
