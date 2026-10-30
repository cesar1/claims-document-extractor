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


def _cost(usage) -> float:
    cache_read = usage.cache_read_input_tokens or 0
    cache_write = usage.cache_creation_input_tokens or 0
    return (
        usage.input_tokens * PRICE_INPUT
        + cache_read * PRICE_CACHE_READ
        + cache_write * PRICE_CACHE_WRITE
        + usage.output_tokens * PRICE_OUTPUT
    ) / 1_000_000


def build_request(pdf_bytes: bytes) -> dict:
    """Request kwargs, shared by the live path and (later) the Batches eval path."""
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
        "output_config": {"effort": EFFORT},
        "output_format": ExtractionResult,
    }


def extract_pdf(pdf_bytes: bytes, client: anthropic.Anthropic | None = None) -> Extraction:
    client = client or anthropic.Anthropic()
    with client.beta.messages.stream(
        **build_request(pdf_bytes),
        fallbacks="default",
        betas=[FALLBACK_BETA],
    ) as stream:
        message = stream.get_final_message()

    if message.stop_reason == "refusal":
        raise ExtractionError(f"Model declined the document: {message.stop_details}")
    if message.stop_reason == "max_tokens":
        raise ExtractionError("Output hit max_tokens before the extraction was complete")
    if message.parsed_output is None:
        raise ExtractionError(f"No structured output (stop_reason={message.stop_reason})")

    return Extraction(
        result=message.parsed_output,
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
        cost_usd=_cost(message.usage),
        message_id=message.id,
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
    print(f"[total] ${total:.4f}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
