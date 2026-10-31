"""Parse the normalized string values Claude returns (see the prompt's normalization rules)."""

import re
from datetime import date, datetime


def parse_number(v) -> float | None:
    """'1,234.50', '$1234.5', '(115.42)' -> float; None if absent or unparseable."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = re.sub(r"[,$\s]", "", str(v))
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    try:
        return float(s)
    except ValueError:
        return None


def parse_date(v) -> date | None:
    """ISO 'YYYY-MM-DD' (a trailing time part is ignored); None if absent or unparseable."""
    if v is None:
        return None
    try:
        return date.fromisoformat(str(v).strip()[:10])
    except ValueError:
        return None


def parse_datetime(v) -> datetime | None:
    """ISO 'YYYY-MM-DDTHH:MM' (a space instead of T is accepted)."""
    if v is None:
        return None
    try:
        return datetime.fromisoformat(str(v).strip().replace(" ", "T")[:16])
    except ValueError:
        return None
