"""Native lalamo multimodal inference, used the way an application would: bfloat16, default settings, reply().

Everything runs in lalamo: preprocessing, the vision encoder, splicing image embeddings into the prompt,
M-RoPE positions, prefill and decoding. Exact parity with HF is covered by the tracer tests; these tests
check that real questions about real inputs get the right answers.
"""

import json
import logging
import time
from functools import cache
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFont

from lalamo.data import PdfRenderConfig, pdf_image_parts
from lalamo.model_import.common import import_model
from lalamo.models import GenerationConfig, VisionLanguageModel
from lalamo.models.chat_codec import ContentPart, ImagePart, ReasoningEffort, TextPart, UserMessage
from lalamo.module import Keychain
from lalamo.utils.sharding import ShardingConfig

log = logging.getLogger(__name__)

REPOS = [pytest.param("Qwen/Qwen3.5-0.8B", id="0.8B"), pytest.param("Qwen/Qwen3.5-27B", id="27B")]


@cache
def _model(repo: str) -> VisionLanguageModel:
    sharding_config = ShardingConfig.replicated(jax.devices()[:1])
    model = import_model(repo, sharding_config=sharding_config, dtype=jnp.bfloat16, include_vision=True).model
    assert isinstance(model, VisionLanguageModel)
    return model


def _ask(repo: str, parts: tuple[ContentPart, ...], max_output_length: int = 48) -> str:
    model = _model(repo)
    question = next(part.text for part in parts if isinstance(part, TextPart))
    start = time.perf_counter()
    reply = model.reply(
        [UserMessage(parts)],
        generation_config=GenerationConfig(temperature=0.0),
        max_output_length=max_output_length,
        # These tests check short factual answers. Qwen3.5-27B reasons by default (0.8B does not), and
        # with reasoning on the token budget is spent inside the <think> block before any answer.
        reasoning_effort=ReasoningEffort.NO_REASONING,
        keychain=Keychain.init(0, sharding_config=model.sharding_config),
    )
    answer = reply.response.strip()
    record = {
        "model": repo,
        "images": sum(isinstance(part, ImagePart) for part in parts),
        "question": question,
        "answer": answer,
        "seconds": round(time.perf_counter() - start, 2),
    }
    log.info("ANSWER %s", json.dumps(record))
    return answer


def _photo(filename: str) -> Image.Image:
    return Image.open(hf_hub_download("huggingface/documentation-images", filename, repo_type="dataset"))


@pytest.mark.parametrize("repo", REPOS)
@pytest.mark.parametrize(("filename", "expected_word"), [("bee.jpg", "bee"), ("pipeline-cat-chonk.jpeg", "cat")])
def test_answers_questions_about_photos(repo: str, filename: str, expected_word: str) -> None:
    question = TextPart("What animal is in this picture? Answer in one sentence.")
    answer = _ask(repo, (ImagePart(_photo(filename)), question))
    assert expected_word in answer.lower()


@pytest.mark.parametrize("repo", REPOS)
def test_answers_questions_about_two_images_at_once(repo: str) -> None:
    answer = _ask(
        repo,
        (
            ImagePart(_photo("bee.jpg")),
            ImagePart(_photo("pipeline-cat-chonk.jpeg")),
            TextPart("Name the animal in each of the two pictures, in order."),
        ),
        # The model describes each picture before naming the animals, so it needs more room.
        max_output_length=256,
    )
    assert "bee" in answer.lower()
    assert "cat" in answer.lower()
    assert answer.lower().index("bee") < answer.lower().index("cat")


def write_invoice_pdf(path: Path) -> None:
    font = ImageFont.load_default(size=40)
    pages = []
    for lines in (["ACME SUPPLIES", "Invoice number: 4721"], ["Payment summary", "Total due: 93 dollars"]):
        page = Image.new("RGB", (612, 792), "white")
        draw = ImageDraw.Draw(page)
        for line_index, line in enumerate(lines):
            draw.text((50, 120 + 80 * line_index), line, fill="black", font=font)
        pages.append(page)
    first, *rest = pages
    Image.init()  # Pillow's PDF writer needs the lazily registered JPEG plugin.
    first.save(path, save_all=True, append_images=rest, resolution=72.0)


@pytest.mark.parametrize("repo", REPOS)
@pytest.mark.parametrize(
    ("question", "expected"),
    [("What is the invoice number?", "4721"), ("What is the total amount due, in dollars?", "93")],
)
def test_answers_questions_about_a_pdf(repo: str, tmp_path: Path, question: str, expected: str) -> None:
    pdf_path = tmp_path / "invoice.pdf"
    write_invoice_pdf(pdf_path)
    pages = pdf_image_parts(pdf_path, PdfRenderConfig(dpi=144.0))
    answer = _ask(repo, (*pages, TextPart(f"{question} Answer briefly.")))
    assert expected in answer
