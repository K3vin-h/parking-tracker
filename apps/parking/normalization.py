"""Canonical identifiers shared by parking models, forms, and services."""


def canonicalize_plate(raw_text: str | None) -> str:
    """Collapse whitespace and case so one physical plate has one database key."""
    if not raw_text:
        return ""
    return "".join(str(raw_text).split()).upper()
