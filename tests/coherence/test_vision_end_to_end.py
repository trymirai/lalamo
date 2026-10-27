"""End-to-end check of the lalamo vision path on a real photo.

The pipeline is: lalamo preprocessing, then the lalamo vision encoder, then HF's Qwen3.5 text decoder.
lalamo cannot generate from images by itself yet (M-RoPE and embedding splicing are phase 2), so the
encoder is checked the way that matters: the language model must still understand the picture.

HF's vision tower is swapped for a shim that returns lalamo's embeddings. Everything else is stock HF:
the tokenizer, the chat template with its ``<|image_pad|>`` placeholders, M-RoPE and the decoder.

``AutoProcessor`` cannot be used because its video sub-processor needs torchvision, which this project
does not install. ``_hf_inputs`` reproduces the image path of ``Qwen3VLProcessor.__call__`` instead,
using HF's own tokenizer, chat template and PIL image processor.
"""

import logging
from collections.abc import Callable
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoTokenizer, PreTrainedTokenizerBase, Qwen3_5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil

from lalamo.model_import.vision import import_qwen3_5_vision_encoder
from lalamo.module import Keychain
from lalamo.modules.vision import ImageGrid, VisionForwardPassConfig
from lalamo.utils.sharding import ShardingConfig
from tests.tracer.tracer import configure_precision_for_tests

log = logging.getLogger(__name__)

REPO = "Qwen/Qwen3.5-0.8B"
IMAGE_TOKEN = "<|image_pad|>"
PROMPT = "What animal is in this picture? Answer in one short sentence."


class PhotoCase(NamedTuple):
    filename: str
    expected_word: str


PHOTO_CASES = [
    pytest.param(PhotoCase("bee.jpg", "bee"), id="bee"),
    pytest.param(PhotoCase("pipeline-cat-chonk.jpeg", "cat"), id="cat"),
]


def _tokenizer() -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(REPO)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    return tokenizer


def _photo(filename: str) -> Image.Image:
    path = hf_hub_download("huggingface/documentation-images", filename, repo_type="dataset")
    return Image.open(path).convert("RGB")


def _lalamo_image_embeddings(image: Image.Image) -> tuple[torch.Tensor, ImageGrid]:
    sharding_config = ShardingConfig.replicated(jax.devices()[:1])
    imported = import_qwen3_5_vision_encoder(REPO, sharding_config=sharding_config, dtype=jnp.float32)
    preprocessed = imported.preprocessing_config.preprocess([image])
    patches = jax.device_put(jnp.asarray(preprocessed.patches), sharding_config.make_sharding((None, None)))
    result = imported.encoder(
        patches,
        preprocessed.grids,
        VisionForwardPassConfig.for_tracer_tests(),
        keychain=Keychain.init(0, sharding_config=sharding_config),
    )
    (grid,) = preprocessed.grids
    return torch.from_numpy(np.asarray(jax.device_get(result.outputs)).copy()), grid


def _vision_shim(embeddings: torch.Tensor, expected_grid: ImageGrid) -> Callable[..., BaseModelOutputWithPooling]:
    def forward(_pixel_values: torch.Tensor, grid_thw: torch.Tensor, **_kwargs: object) -> BaseModelOutputWithPooling:
        (hf_grid,) = grid_thw.tolist()
        assert ImageGrid(*hf_grid) == expected_grid, "HF and lalamo preprocessing produced different grids"
        hidden_states = cast("torch.FloatTensor", embeddings)
        return BaseModelOutputWithPooling(last_hidden_state=hidden_states, pooler_output=hidden_states)

    return forward


def _hf_inputs(image: Image.Image) -> dict[str, torch.Tensor]:
    tokenizer = _tokenizer()
    image_processor = Qwen2VLImageProcessorPil.from_pretrained(REPO)
    image_inputs = image_processor(images=[image], return_tensors="pt")

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    # HF annotates conversations as text-only, but multimodal content parts are a list.
    prompt = tokenizer.apply_chat_template(
        cast("Any", messages),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert isinstance(prompt, str)
    (grid,) = image_inputs["image_grid_thw"]
    num_image_tokens = int(grid.prod()) // image_processor.merge_size**2
    prompt = prompt.replace(IMAGE_TOKEN, IMAGE_TOKEN * num_image_tokens, 1)

    text_inputs = tokenizer([prompt], return_tensors="pt")
    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    mm_token_type_ids = (text_inputs["input_ids"] == image_token_id).long()
    return {**text_inputs, **image_inputs, "mm_token_type_ids": mm_token_type_ids}


def _describe(model: Qwen3_5ForConditionalGeneration, inputs: dict[str, torch.Tensor]) -> str:
    with torch.no_grad():
        # HF types generate() against a narrower base class than the multimodal model it lives on.
        generated = cast("Any", model).generate(**inputs, max_new_tokens=32, do_sample=False)
    prompt_length = inputs["input_ids"].shape[1]
    description = _tokenizer().decode(generated[0, prompt_length:], skip_special_tokens=True)
    assert isinstance(description, str)
    return description.strip()


@pytest.mark.parametrize("case", PHOTO_CASES)
def test_lalamo_vision_embeddings_drive_correct_description(case: PhotoCase) -> None:
    configure_precision_for_tests()
    image = _photo(case.filename)
    inputs = _hf_inputs(image)

    model = Qwen3_5ForConditionalGeneration.from_pretrained(REPO, dtype=torch.float32).eval()
    with torch.no_grad():
        hf_embeddings = model.model.visual(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"]).pooler_output
    hf_description = _describe(model, inputs)

    lalamo_embeddings, grid = _lalamo_image_embeddings(image)
    model.model.visual.forward = _vision_shim(lalamo_embeddings, grid)
    lalamo_description = _describe(model, inputs)

    # Both sides preprocess identically (PIL backend), so any gap comes from float32 arithmetic.
    cosine = torch.nn.functional.cosine_similarity(lalamo_embeddings, hf_embeddings, dim=-1)
    log.info("Image %s, grid %s", case.filename, grid)
    log.info("Per-token cosine similarity: mean %.5f, min %.5f", cosine.mean(), cosine.min())
    log.info("HF vision tower:     %s", hf_description)
    log.info("lalamo vision tower: %s", lalamo_description)

    assert cosine.min() > 0.9999
    assert lalamo_description == hf_description
    assert case.expected_word in lalamo_description.lower()
    assert case.expected_word in hf_description.lower()
