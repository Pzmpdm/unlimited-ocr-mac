"""Page-quality routing: decide how to process a PDF page.

Native — the embedded text layer is trustworthy and rich; use layout
         extraction directly (fastest and most exact for digital PDFs).
Hybrid — the text layer is good but the page has complex regions
         (tables, figures, images, multi-column) that benefit from
         selective region-level OCR (see hybrid_ocr).
VLM    — the page is scanned or its text layer is unusable; run the
         visual model on the whole page.

The heuristics stay simple and data-driven: they only use values already
computed by pdf_converter.extract_native_page() so the per-page
preprocessing cost does not grow.
"""
from dataclasses import dataclass
from typing import Literal

PageRoute = Literal["native", "hybrid", "vlm"]

# Thresholds are module constants, not scattered magic numbers.
MIN_NATIVE_TEXT_CHARS = 80  # below this the text layer is too sparse
MAX_SUSPICIOUS_CHAR_RATIO = 0.03  # above this the text layer is corrupted
HYBRID_IMAGE_COVERAGE = 0.10  # image area fraction that triggers hybrid


@dataclass
class PageAnalysis:
    text_chars: int = 0
    text_blocks: int = 0
    image_count: int = 0
    image_coverage: float = 0.0
    drawing_count: int = 0
    table_count: int = 0
    figure_count: int = 0
    suspicious_char_ratio: float = 0.0
    column_count: int = 1  # 1 | 2 | 0 (unknown)
    native_markdown_chars: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> "PageAnalysis":
        """Build from an analysis dict, ignoring unknown keys."""
        if not data:
            return cls()
        known = {name: data[name] for name in cls.__dataclass_fields__ if name in data}
        return cls(**known)


def suspicious_character_ratio(text: str) -> float:
    """Fraction of characters that look like a broken text layer.

    Counts replacement chars (U+FFFD), un-normalized Private Use Area code
    points and stray control characters that should not appear in real text.
    """
    if not text:
        return 0.0
    total = 0
    suspicious = 0
    for ch in text:
        code = ord(ch)
        total += 1
        if ch == "\ufffd" or 0xE000 <= code <= 0xF8FF or (code < 0x20 and ch not in "\t\n\r"):
            suspicious += 1
    return suspicious / total if total else 0.0


def estimate_column_count(lines) -> int:
    """Rough column estimate from the text-line x distribution.

    Splits sorted x0 coordinates in half; if both sides hold at least two
    lines, the inter-cluster gap is a large fraction of the total span and
    each cluster stays narrow, treat the page as two-column. Returns 1 or 2
    (0 is reserved for genuinely unknown pages, not used today).
    """
    x0s = sorted(float(line["bbox"][0]) for line in lines if line.get("bbox"))
    if len(x0s) < 6:
        return 1
    mid = len(x0s) // 2
    left = x0s[:mid]
    right = x0s[mid:]
    if not left or not right:
        return 1
    span = max(1.0, right[-1] - left[0])
    gap = right[0] - left[-1]
    if gap > 0.25 * span and (left[-1] - left[0]) < 0.5 * span and (right[-1] - right[0]) < 0.5 * span:
        return 2
    return 1


def choose_page_route(a: PageAnalysis) -> PageRoute:
    """Pick native / hybrid / vlm from a page analysis.

    Order matters: VLM wins for sparse or corrupted text layers; hybrid wins
    when the text is usable but the page contains tables, figures, heavy
    images or multiple columns; everything else stays native.
    """
    if a.text_chars < MIN_NATIVE_TEXT_CHARS or a.suspicious_char_ratio > MAX_SUSPICIOUS_CHAR_RATIO:
        return "vlm"
    if (
        a.table_count > 0
        or a.figure_count > 0
        or a.image_coverage >= HYBRID_IMAGE_COVERAGE
        or a.column_count >= 2
    ):
        return "hybrid"
    return "native"
