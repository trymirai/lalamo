from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from lalamo.data import PdfRenderConfig, pdf_image_parts, render_pdf_pages


def _write_pdf(path: Path, page_texts: list[str]) -> None:
    """A US-letter PDF (612 x 792 points) with one big line of black text per page."""
    pages = []
    for text in page_texts:
        page = Image.new("RGB", (612, 792), "white")
        ImageDraw.Draw(page).text((60, 300), text, fill="black", font=ImageFont.load_default(size=48))
        pages.append(page)
    first, *rest = pages
    # Pillow's PDF writer stores RGB pages as JPEG, and plugins register lazily: load them all first.
    Image.init()
    first.save(path, save_all=True, append_images=rest, resolution=72.0)


def test_renders_every_page_at_the_requested_resolution(tmp_path: Path) -> None:
    path = tmp_path / "doc.pdf"
    _write_pdf(path, ["first", "second", "third"])
    pages = render_pdf_pages(path, PdfRenderConfig(dpi=144.0))
    assert len(pages) == 3
    assert all(page.size == (1224, 1584) and page.mode == "RGB" for page in pages)
    # Text survives rendering: the page is mostly white but has dark ink.
    gray = np.asarray(pages[0].convert("L"))
    assert gray.min() < 64
    assert gray.max() == 255


def test_accepts_bytes_and_builds_message_parts(tmp_path: Path) -> None:
    path = tmp_path / "doc.pdf"
    _write_pdf(path, ["only page"])
    parts = pdf_image_parts(path.read_bytes(), PdfRenderConfig(dpi=72.0))
    assert len(parts) == 1
    assert parts[0].image.size == (612, 792)


def test_refuses_documents_over_the_page_limit(tmp_path: Path) -> None:
    path = tmp_path / "doc.pdf"
    _write_pdf(path, ["a", "b", "c"])
    with pytest.raises(ValueError, match="max_pages"):
        render_pdf_pages(path, PdfRenderConfig(max_pages=2))
