from pathlib import Path
from typing import cast

import torch
from fish_speech.models.text2semantic.llama import DualARModelArgs
from fish_speech.tokenizer import FishTokenizer
from jax import numpy as jnp

from lalamo import FileSpec
from lalamo.initializer import EmptyInitializer
from lalamo.model_import.loaders.fishaudio_loaders import (
    load_tokenizer_from_fishaudio_tiktoken,
    load_transformer_block,
)
from lalamo.model_import.model_specs.fishaudio import FISHAUDIO_TTS_MODELS
from lalamo.module import Keychain
from lalamo.modules.utils import call_vmapped
from lalamo.utils.torch_interop import torch_to_jax
from tests.common import assert_close
from tests.helpers import make_test_sharding_config
from tests.tts.fishaudio.fishaudio_thin_wrapper import (
    FishAudioTextDecoder_Foreign,
)
from tests.tts.fishaudio.fishaudio_torch_stuff import FishAudioFromTorch

from .fishaudio_torch_stuff import ConfigMapping, prepare_state_dict_for_lalamo_loaders


def test_fishaudio_text_tokenization(fish_audio_local_model_path: Path) -> None:
    model_spec = next(model for model in FISHAUDIO_TTS_MODELS if model.origin.description == "fishaudio/s1-mini")
    assert isinstance(model_spec.configs.tokenizer, FileSpec)
    tokenizer_path = fish_audio_local_model_path / model_spec.configs.tokenizer.filename
    tokenizer_special_tokens_path = fish_audio_local_model_path / "special_tokens.json"

    lalamo_tokenizer, _ = load_tokenizer_from_fishaudio_tiktoken(tokenizer_path, tokenizer_special_tokens_path)
    fish_tokenizer = FishTokenizer.from_pretrained(str(fish_audio_local_model_path))

    test_text = "red brown dog jumped over a lazy fox 01234567890 $#!@-+_"

    tokens_fish = jnp.asarray(fish_tokenizer.encode(test_text))
    tokens_lalamo = jnp.asarray(lalamo_tokenizer.encode(test_text).ids)
    assert jnp.all(tokens_fish == tokens_lalamo)


@torch.no_grad
def test_single_text_transformer_layer(fish_audio_local_model_path: Path) -> None:
    pytorch_tts_generator = FishAudioFromTorch.build_foreign_fish_audio_tts_generator(fish_audio_local_model_path)
    assert isinstance(pytorch_tts_generator.text_decoder, FishAudioTextDecoder_Foreign)
    fish_model = pytorch_tts_generator.text_decoder.fish_model
    config = fish_model.config
    assert isinstance(config, DualARModelArgs)

    transformer_cfg, _ = ConfigMapping.lalamo_transformer_cfg_from_fish_text_decoder_cfg(config)
    lalamo_transformer = transformer_cfg.init(
        EmptyInitializer(default_dtype=jnp.bfloat16, sharding_config=make_test_sharding_config()),
    )

    weights_dict = prepare_state_dict_for_lalamo_loaders(fish_model.state_dict())
    lalamo_transformer = load_transformer_block(lalamo_transformer, weights_dict)

    batch_size = 1
    seq_length = 16
    model_dim = config.dim

    torch.manual_seed(42)
    embedded_input_torch = torch.randn(batch_size, seq_length, model_dim, dtype=torch.bfloat16)
    embedded_input_lalamo = torch_to_jax(embedded_input_torch)

    max_seq_len = seq_length
    input_pos = torch.arange(max_seq_len, device=embedded_input_torch.device)
    input_pos_lalamo = torch_to_jax(input_pos)[None, :]

    causal_mask = cast("torch.Tensor", fish_model.causal_mask)
    freqs_cis_table = cast("torch.Tensor", fish_model.freqs_cis)
    mask = causal_mask[None, None, input_pos, :max_seq_len]
    freqs_cis = freqs_cis_table[input_pos]

    fish_layer = fish_model.layers[0]
    fish_layer_result = fish_layer(embedded_input_torch, freqs_cis, mask, input_pos=input_pos)

    (global_rope,) = lalamo_transformer.ropes
    pos_emb_lalamo = call_vmapped(global_rope, input_pos_lalamo)
    lalamo_layer = lalamo_transformer.layers[0]
    lalamo_layer_result = lalamo_layer(
        embedded_input_lalamo,
        pos_emb_lalamo,
        keychain=Keychain.init(0, sharding_config=make_test_sharding_config()),
    )

    fish_output_jax = torch_to_jax(fish_layer_result)
    lalamo_output = lalamo_layer_result.outputs

    assert_close(
        result=lalamo_output,
        reference=fish_output_jax,
        atol=1e-1,
        rtol=1e-5,
        operation_name="test_single_text_transformer_layer",
    )
