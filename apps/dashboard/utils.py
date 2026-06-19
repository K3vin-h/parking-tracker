"""Shared dashboard presentation helpers used by views and APIs."""


def confidence_band(score: float) -> str:
    """Map confidence to the fixed green/yellow/red bands required by the UI."""
    if score >= 0.8:
        return "good"
    if score >= 0.6:
        return "warning"
    return "error"
