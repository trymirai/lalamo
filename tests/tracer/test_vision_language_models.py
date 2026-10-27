"""Full multimodal parity: lalamo's VisionLanguageModel against HF Qwen3_5ForConditionalGeneration.

Both sides get the same images and question. lalamo runs its own preprocessing, vision encoder, embedding
splice, M-RoPE positions and decoder; HF runs its stock model on identical token ids and pixels. Logits
are compared at every prompt position, and greedy generation must agree token for token.

Each comparison logs one ``PARITY {...}`` JSON line with its metrics, so results can be collected.
"""

import json
import logging
import tempfile
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import Qwen3_5ForConditionalGeneration

from lalamo.data import PdfRenderConfig, render_pdf_pages
from lalamo.model_import.common import import_model
from lalamo.models import GenerationConfig, VisionLanguageModel
from lalamo.models.chat_codec import ImagePart, TextPart, UserMessage
from lalamo.module import Keychain
from lalamo.modules import DecoderForwardPassConfig
from lalamo.modules.vision import VisionForwardPassConfig
from lalamo.utils.sharding import ShardingConfig
from tests.coherence.test_vision_language_end_to_end import write_invoice_pdf
from tests.common import assert_close
from tests.tracer.tracer import FRACTION_OF_ALLOWED_VIOLATIONS, configure_precision_for_tests
from tests.tracer.tracer_vision import synthetic_image

log = logging.getLogger(__name__)

NUM_GENERATED_TOKENS = 24
# Bounds on the relative RMS error of the prompt logits (float32 end to end). They come from measuring
# how much HF's *own* logits move when its vision embeddings get noise the size of a float32 vision
# tower's error (3.7e-5): 5e-5 for 0.8B, but up to 5.6e-3 for 27B, which amplifies such perturbations
# 10-150x. A fixed bound calibrated on 0.8B is therefore the wrong yardstick for 27B. Identical greedy
# generation and near-total top-1 agreement remain the decisive checks for every model.
MAX_RELATIVE_RMS_ERROR = {"Qwen/Qwen3.5-0.8B": 1e-3, "Qwen/Qwen3.5-27B": 5e-3}
MIN_TOP1_AGREEMENT = 0.995
# Plain greedy decoding. The Qwen3.5 spec adds a repetition penalty by default, and the 27B checkpoint's
# generation_config.json could do the same on the HF side, so both are disabled explicitly.
GREEDY = GenerationConfig(temperature=0.0, repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0)


@dataclass(frozen=True)
class VisionLanguageTestSpec:
    model_repo: str
    inputs: str
    question: str

    @property
    def test_id(self) -> str:
        return f"{self.model_repo}/{self.inputs}"


SPECS = [
    VisionLanguageTestSpec(repo, inputs, question)
    for repo in ("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-27B")
    for inputs, question in (
        ("photo", "Describe this image in one sentence."),
        ("synthetic", "Describe this image in one sentence."),
        ("two-page-pdf", "What is the invoice number and the total due?"),
    )
]


def _images(inputs: str) -> tuple[Image.Image, ...]:
    match inputs:
        case "photo":
            path = hf_hub_download("huggingface/documentation-images", "pipeline-cat-chonk.jpeg", repo_type="dataset")
            return (Image.open(path).convert("RGB"),)
        case "synthetic":
            return (synthetic_image(448, 320, seed=3),)
        case "two-page-pdf":
            with tempfile.TemporaryDirectory() as directory:
                pdf_path = Path(directory) / "invoice.pdf"
                write_invoice_pdf(pdf_path)
                return render_pdf_pages(pdf_path, PdfRenderConfig(dpi=144.0))
        case _:
            raise ValueError(inputs)


def _sharding_config() -> ShardingConfig:
    # One device: the shared tracer mesh replicates weights to every GPU, i.e. eight float32 copies of 27B.
    return ShardingConfig.replicated(jax.devices()[:1])


@cache
def _lalamo_model(repo: str) -> VisionLanguageModel:
    model = import_model(repo, sharding_config=_sharding_config(), dtype=jnp.float32, include_vision=True).model
    assert isinstance(model, VisionLanguageModel)
    return model


@cache
def _hf_model(repo: str) -> Qwen3_5ForConditionalGeneration:
    return Qwen3_5ForConditionalGeneration.from_pretrained(repo, dtype=torch.float32).eval()


def _relative_rms_error(result: jax.Array, reference: jax.Array) -> float:
    return float(jnp.sqrt(jnp.mean(jnp.square(result - reference))) / jnp.sqrt(jnp.mean(jnp.square(reference))))


@pytest.mark.parametrize("spec", SPECS, ids=[spec.test_id for spec in SPECS])
def test_vision_language_model_matches_huggingface(spec: VisionLanguageTestSpec) -> None:
    configure_precision_for_tests()
    model = _lalamo_model(spec.model_repo)
    images = _images(spec.inputs)
    prompt = model.build_prompt([UserMessage((*(ImagePart(image) for image in images), TextPart(spec.question)))])
    assert prompt.images is not None
    keychain = Keychain.init(0, sharding_config=model.sharding_config)
    decoder_config = DecoderForwardPassConfig.for_tracer_tests()
    inputs = model.prompt_inputs(prompt, decoder_config, VisionForwardPassConfig.for_tracer_tests(), keychain=keychain)
    lalamo_logits = model.decoder(
        inputs.token_ids,
        inputs.positions,
        input_embeddings=inputs.embeddings,
        forward_pass_config=decoder_config,
        keychain=keychain,
    ).logits[0]

    hf_model = _hf_model(spec.model_repo)
    input_ids = torch.from_numpy(prompt.token_ids.astype(np.int64))[None]
    grids = prompt.images.grids
    hf_inputs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        # lalamo's preprocessing is verified bit-exact against HF's PIL processor in the unit tests.
        "pixel_values": torch.from_numpy(prompt.images.patches),
        "image_grid_thw": torch.tensor([[g.num_frames, g.height_in_patches, g.width_in_patches] for g in grids]),
        "mm_token_type_ids": (input_ids == model.vision_config.image_token_id).long(),
    }
    with torch.no_grad():
        hf_logits = jnp.asarray(hf_model(**hf_inputs).logits[0].numpy())

    lalamo_reply = model.generate_tokens(
        inputs.token_ids,
        generation_config=GREEDY,
        max_output_length=NUM_GENERATED_TOKENS,
        prefill_forward_pass_config=decoder_config,
        decode_forward_pass_config=decoder_config,
        prompt_embeddings=inputs.embeddings,
        prompt_positions=inputs.positions,
        keychain=keychain,
    ).token_ids[0]
    with torch.no_grad():
        hf_generated = cast("Any", hf_model).generate(
            **hf_inputs,
            max_new_tokens=NUM_GENERATED_TOKENS,
            do_sample=False,
            repetition_penalty=1.0,
        )
    hf_reply = hf_generated[0, input_ids.shape[1] :].tolist()
    stop_token_ids = set(model.config.generation_config.stop_token_ids)
    lalamo_tokens = lalamo_reply.tolist()[: len(hf_reply)]
    compared_length = next((index for index, token in enumerate(hf_reply) if token in stop_token_ids), len(hf_reply))
    compared_length = min(compared_length + 1, len(hf_reply))

    relative_rms_error = _relative_rms_error(lalamo_logits, hf_logits)
    metrics = {
        "test": spec.test_id,
        "images": len(images),
        "grids": [[g.height_in_patches, g.width_in_patches] for g in grids],
        "prompt_tokens": len(prompt.token_ids),
        "image_tokens": int((prompt.token_ids == model.vision_config.image_token_id).sum()),
        "last_position": int(prompt.positions[-1].max()),
        "logits_relative_rms_error": relative_rms_error,
        "logits_max_abs_error": float(jnp.max(jnp.abs(lalamo_logits - hf_logits))),
        "top1_agreement": float(jnp.mean(jnp.argmax(lalamo_logits, -1) == jnp.argmax(hf_logits, -1))),
        "generated_tokens_compared": compared_length,
        "generation_identical": lalamo_tokens[:compared_length] == hf_reply[:compared_length],
        "lalamo_text": model.token_codec.decode_tokens(model.config.generation_config.trim_stop_tokens(lalamo_tokens)),
        "hf_text": model.token_codec.decode_tokens(model.config.generation_config.trim_stop_tokens(hf_reply)),
    }
    log.info("PARITY %s", json.dumps(metrics))

    assert_close(
        result=lalamo_logits,
        reference=hf_logits,
        operation_name="Prompt logits",
        fraction_of_allowed_violations=FRACTION_OF_ALLOWED_VIOLATIONS,
    )
    bound = MAX_RELATIVE_RMS_ERROR[spec.model_repo]
    assert relative_rms_error < bound, f"Prompt logits relative RMS error {relative_rms_error:.2e} exceeds {bound:.0e}"
    assert metrics["top1_agreement"] >= MIN_TOP1_AGREEMENT, f"Top-1 agreement {metrics['top1_agreement']:.4f}"
    assert metrics["generation_identical"], f"lalamo: {metrics['lalamo_text']!r}\nHF:     {metrics['hf_text']!r}"
