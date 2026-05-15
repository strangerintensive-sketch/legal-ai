"""
extract_pdf_text.py — Reading-order text extraction for PDFs.

Public API:
    extract(pdf_path) -> str

Handled per-page layouts (classified independently, no global split):
    - single column
    - two columns
    - two columns with an optional full-width header band

Cross-page assembly order:
    [deduped headers, page order]
    [page1 left  + page2 left  + ...]
    [single-column pages, in page order]
    [page1 right + page2 right + ...]

Backed by PyMuPDF (fitz). Output is cleaned for NER tokenization consistency:
NFKC normalize, strip zero-width / soft-hyphen, fold typographic dashes /
quotes / ellipsis, collapse intra-line whitespace.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import fitz


# ─── Tunables ────────────────────────────────────────────────────────────────

_HEADER_WIDTH_RATIO = 0.85     # block width >= 85% page width => header candidate
                                # (lower values misclassify wide main-column
                                # paragraphs in left-heavy 2-col layouts)
_SPLIT_SEARCH_LO = 0.15        # search split-x in middle 70% of page
_SPLIT_SEARCH_HI = 0.85
_SPLIT_STEP = 5
_SPLIT_MARGIN = 10             # px tolerance for left/right/crossing classification
_MIN_BLOCKS_FOR_TWO_COL = 4    # too few blocks => treat as single column


# ─── Data model ──────────────────────────────────────────────────────────────

@dataclass
class Block:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2


@dataclass
class PageLayout:
    page_num: int
    width: float
    headers: list[Block] = field(default_factory=list)
    left: list[Block] = field(default_factory=list)
    right: list[Block] = field(default_factory=list)
    full_width: list[Block] = field(default_factory=list)
    split_x: float | None = None

    @property
    def is_single(self) -> bool:
        return self.split_x is None


# ─── PyMuPDF block extraction ────────────────────────────────────────────────

def _extract_blocks(page: fitz.Page) -> list[Block]:
    raw = page.get_text("dict")
    blocks: list[Block] = []
    for b in raw.get("blocks", []):
        if b.get("type", 0) != 0:  # skip image blocks
            continue
        lines: list[str] = []
        for line in b.get("lines", []):
            text = "".join(s.get("text", "") for s in line.get("spans", [])).strip()
            if text:
                lines.append(text)
        if not lines:
            continue
        x0, y0, x1, y1 = b["bbox"]
        blocks.append(
            Block(
                text="\n".join(lines),
                x0=float(x0), y0=float(y0),
                x1=float(x1), y1=float(y1),
            )
        )
    return blocks


# ─── Split detection (per page, body blocks only) ────────────────────────────

def _find_split_x(body_blocks: list[Block], page_width: float) -> float | None:
    if len(body_blocks) < _MIN_BLOCKS_FOR_TWO_COL:
        return None

    best_split: float | None = None
    best_score = 0
    lo = int(page_width * _SPLIT_SEARCH_LO)
    hi = int(page_width * _SPLIT_SEARCH_HI)

    for split_x in range(lo, hi, _SPLIT_STEP):
        left = sum(1 for b in body_blocks if b.x1 < split_x + _SPLIT_MARGIN)
        right = sum(1 for b in body_blocks if b.x0 > split_x - _SPLIT_MARGIN)
        crossing = sum(
            1 for b in body_blocks
            if b.x0 < split_x - _SPLIT_MARGIN and b.x1 > split_x + _SPLIT_MARGIN
        )
        if left >= 2 and right >= 2 and crossing <= 1:
            score = left + right - crossing * 5
            if score > best_score:
                best_score = score
                best_split = float(split_x)

    return best_split


# ─── Per-page classification ─────────────────────────────────────────────────

def _classify_page(blocks: list[Block], page_width: float, page_num: int) -> PageLayout:
    layout = PageLayout(page_num=page_num, width=page_width)
    if not blocks:
        return layout

    wide_threshold = page_width * _HEADER_WIDTH_RATIO
    narrow_blocks = [b for b in blocks if b.width < wide_threshold]

    split_x = _find_split_x(narrow_blocks, page_width)

    if split_x is None:
        # Single column: whole page goes into full_width bucket, in y-order.
        layout.full_width = sorted(blocks, key=lambda b: (b.y0, b.x0))
        return layout

    # Two columns. Header = wide block OR block that crosses the split.
    # Header detection is optional — list stays empty if no block qualifies.
    headers: list[Block] = []
    body: list[Block] = []
    for b in blocks:
        spans_split = b.x0 < split_x - _SPLIT_MARGIN and b.x1 > split_x + _SPLIT_MARGIN
        if b.width >= wide_threshold or spans_split:
            headers.append(b)
        else:
            body.append(b)

    layout.split_x = split_x
    layout.headers = sorted(headers, key=lambda b: b.y0)
    layout.left = sorted(
        [b for b in body if b.cx < split_x], key=lambda b: (b.y0, b.x0)
    )
    layout.right = sorted(
        [b for b in body if b.cx >= split_x], key=lambda b: (b.y0, b.x0)
    )
    return layout


# ─── Assembly ────────────────────────────────────────────────────────────────

def _dedupe_headers(headers: list[Block]) -> list[Block]:
    seen: set[str] = set()
    out: list[Block] = []
    for h in headers:
        key = " ".join(h.text.split()).lower()
        if key and key not in seen:
            seen.add(key)
            out.append(h)
    return out


def _join_blocks(blocks: list[Block]) -> str:
    return "\n\n".join(b.text for b in blocks if b.text)


def _assemble(pages: list[PageLayout]) -> str:
    all_headers: list[Block] = []
    left_bucket: list[Block] = []
    right_bucket: list[Block] = []
    full_width_pages: list[Block] = []

    for p in pages:
        all_headers.extend(p.headers)
        if p.is_single:
            full_width_pages.extend(p.full_width)
        else:
            left_bucket.extend(p.left)
            right_bucket.extend(p.right)

    parts: list[str] = []
    deduped = _dedupe_headers(all_headers)
    if deduped:
        parts.append(_join_blocks(deduped))
    if left_bucket:
        parts.append(_join_blocks(left_bucket))
    if full_width_pages:
        parts.append(_join_blocks(full_width_pages))
    if right_bucket:
        parts.append(_join_blocks(right_bucket))

    return "\n\n".join(parts)


# ─── Text cleanup (for NER tokenization consistency) ────────────────────────

# Zero-width space, ZWNJ, ZWJ, BOM, soft hyphen — invisible chars that PDFs
# (especially justified text) leak into extracted strings and that wreck
# tokenization consistency.
_INVISIBLE_CHARS = re.compile("[​‌‍﻿­]")
_INTRA_LINE_SPACES = re.compile(r"[ \t]+")

# NFKC does not fold typographic dashes, smart quotes, or ellipsis — they're
# Unicode-distinct, not compatibility characters. Mixed ASCII/typography
# variants across a multi-source corpus would split equivalent tokens.
_TYPOGRAPHY_FOLD = str.maketrans({
    "–": "-",  # en dash
    "—": "-",  # em dash
    "−": "-",  # minus sign
    "‐": "-",  # hyphen
    "‑": "-",  # non-breaking hyphen
    "“": '"',
    "”": '"',
    "„": '"',
    "‟": '"',
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
    "…": "...",
})


def _clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE_CHARS.sub("", text)
    text = text.translate(_TYPOGRAPHY_FOLD)
    text = _INTRA_LINE_SPACES.sub(" ", text)
    return text.strip()


# ─── Public API ──────────────────────────────────────────────────────────────

def extract(pdf_path: str | Path) -> str:
    """Extract reading-order text from a pdf PDF."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {pdf_path.suffix}")

    pages: list[PageLayout] = []
    with fitz.open(pdf_path) as doc:
        for page_idx, page in enumerate(doc):
            blocks = _extract_blocks(page)
            layout = _classify_page(blocks, float(page.rect.width), page_idx + 1)
            pages.append(layout)

    return _clean_text(_assemble(pages))
    # return _assemble(pages)


# ─── CLI ─────────────────────────────────────────────────────────────────────
#
# Single file:  prints extracted text to stdout.
# Folder:       writes <name>.txt next to each <name>.pdf in the folder.

def _process_folder(folder: Path) -> int:
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {folder}", file=sys.stderr)
        return 1
    failed = 0
    for pdf in pdfs:
        out = pdf.with_suffix(".txt")
        try:
            text = extract(pdf)
            out.write_text(text, encoding="utf-8")
            print(f"[OK]   {pdf.name} -> {out.name} ({len(text):,} chars)")
        except Exception as e:
            print(f"[FAIL] {pdf.name}: {e}", file=sys.stderr)
            failed += 1
    print(f"\n{len(pdfs) - failed}/{len(pdfs)} succeeded")
    return 1 if failed else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_pdf_text.py <pdf-or-folder>", file=sys.stderr)
        sys.exit(1)

    target = Path(sys.argv[1])
    if not target.exists():
        print(f"Not found: {target}", file=sys.stderr)
        sys.exit(1)

    if target.is_dir():
        sys.exit(_process_folder(target))

    print(extract(target))


# python extract_pdf_text.py path/to/pdf.pdf      # → stdout
# python extract_pdf_text.py path/to/folder/         # → <name>.txt files in same folder
