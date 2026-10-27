from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

from lalamo.models.chat_codec import ImagePart

__all__ = [
    "PdfRenderConfig",
    "pdf_image_parts",
    "render_pdf_pages",
]

# PDF geometry is expressed in points; one point is 1/72 inch.
POINTS_PER_INCH = 72.0


@dataclass(frozen=True)
class PdfRenderConfig:
    """How PDF pages become images for a vision-language model.

    Qwen-style models have no PDF input of their own: a document is a sequence of page images. Each
    image costs about one token per 32x32 pixels (patch 16, merged 2x2), so the resolution is a direct
    trade-off between legibility and context length. At the default 144 DPI a US-letter page renders
    at 1224x1584 pixels, which is roughly 1900 tokens.
    """

    dpi: float = 144.0
    max_pages: int | None = None

    def __post_init__(self) -> None:
        if self.dpi <= 0:
            raise ValueError("dpi must be positive.")
        if self.max_pages is not None and self.max_pages < 1:
            raise ValueError("max_pages must be at least 1.")


def render_pdf_pages(
    source: Path | str | bytes,
    config: PdfRenderConfig = PdfRenderConfig(),
) -> tuple[Image.Image, ...]:
    """Render every page of a PDF to an RGB image on a white background."""
    pdf_input = source if isinstance(source, bytes) else str(Path(source))
    with pdfium.PdfDocument(pdf_input) as document:
        num_pages = len(document)
        if config.max_pages is not None and num_pages > config.max_pages:
            raise ValueError(f"The PDF has {num_pages} pages, more than max_pages={config.max_pages}.")
        return tuple(
            document[page_index].render(scale=config.dpi / POINTS_PER_INCH).to_pil().convert("RGB")
            for page_index in range(num_pages)
        )


def pdf_image_parts(
    source: Path | str | bytes,
    config: PdfRenderConfig = PdfRenderConfig(),
) -> tuple[ImagePart, ...]:
    """PDF pages as message content, ready to be followed by a ``TextPart`` with the question."""
    return tuple(ImagePart(page) for page in render_pdf_pages(source, config))
