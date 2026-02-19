import numpy as np
import torch
import jax
from jax import numpy as jnp
import types
import sys
import json
import math

from lalamo.common import ParameterPath
from lalamo.model_import.loaders import load_audio_decoder
from lalamo.model_import.model_configs.huggingface.qwen3_tts import Qwen3TTSTokenizer12HzConfig
from lalamo.model_import.loaders.qwen3_tts_loaders import (
    load_qwen3_tts_convnext_block,
    load_qwen3_tts_decoder_block,
    load_qwen3_tts_pre_transformer,
    load_qwen3_tts_residual_unit,
    load_qwen3_tts_split_rvq,
    load_qwen3_tts_text_decoder,
)
from lalamo.modules.audio.qwen3_tts import (
    apply_rotary_pos_emb,
    default_qwen3_tts_audio_decoder_config,
    default_qwen3_tts_text_decoder_config,
    rotate_half,
)
from lalamo.modules.audio.qwen3_tts.qwen3_tts_text_decoding import (
    Qwen3TTSTextDecoder,
    Qwen3TTSTextDecoderConfig,
)
from lalamo.modules.torch_interop import torch_to_jax
from lalamo.sampling import GreedyPolicy
from tests.common import assert_close
from tests.tts.qwen3.reference.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2DecoderConfig,
)
from tests.tts.qwen3.reference.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2ConvNeXtBlock,
    Qwen3TTSTokenizerV2Decoder,
    Qwen3TTSTokenizerV2DecoderDecoderBlock,
    Qwen3TTSTokenizerV2DecoderDecoderResidualUnit,
    Qwen3TTSTokenizerV2DecoderTransformerModel,
    SplitResidualVectorQuantizer,
)
from tests.tts.utils import prepare_state_dict_for_lalamo_loaders


def _assert_very_close(result: jnp.ndarray, reference: jnp.ndarray, operation_name: str) -> None:
    assert_close(
        result=result.astype(jnp.float32),
        reference=reference.astype(jnp.float32),
        atol=1e-6,
        rtol=1e-5,
        operation_name=operation_name,
    )


def _nct_to_nsc(x: np.ndarray) -> np.ndarray:
    return np.transpose(x, (0, 2, 1))


def _nsc_to_nct(x: np.ndarray) -> np.ndarray:
    return np.transpose(x, (0, 2, 1))


def _tiny_decoder_config() -> Qwen3TTSTokenizerV2DecoderConfig:
    config = Qwen3TTSTokenizerV2DecoderConfig(
        codebook_size=64,
        hidden_size=32,
        latent_dim=32,
        max_position_embeddings=256,
        rope_theta=10000,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_bias=True,
        sliding_window=16,
        intermediate_size=96,
        hidden_act="silu",
        layer_scale_initial_scale=0.01,
        rms_norm_eps=1e-5,
        num_hidden_layers=2,
        num_quantizers=4,
        upsample_rates=(2, 2),
        upsampling_ratios=(2, 2),
        decoder_dim=32,
        attention_dropout=0.0,
        head_dim=8,
        num_semantic_quantizers=1,
    )
    config.codebook_dim = 32
    config._attn_implementation = "eager"
    return config


def test_rotate_half_matches_reference() -> None:
    x = torch.randn(2, 3, 8)
    expected = torch.cat((-x[..., 4:], x[..., :4]), dim=-1)
    result = rotate_half(torch_to_jax(x))
    _assert_very_close(result, torch_to_jax(expected), "rotate_half")


def test_apply_rotary_pos_emb_matches_reference_formula() -> None:
    q = torch.randn(2, 4, 6, 8)
    k = torch.randn(2, 4, 6, 8)
    cos = torch.randn(2, 6, 8)
    sin = torch.randn(2, 6, 8)

    cos_e = cos.unsqueeze(1)
    sin_e = sin.unsqueeze(1)
    q_expected = (q * cos_e) + (torch.cat((-q[..., 4:], q[..., :4]), dim=-1) * sin_e)
    k_expected = (k * cos_e) + (torch.cat((-k[..., 4:], k[..., :4]), dim=-1) * sin_e)

    q_result, k_result = apply_rotary_pos_emb(
        torch_to_jax(q),
        torch_to_jax(k),
        torch_to_jax(cos),
        torch_to_jax(sin),
    )

    _assert_very_close(q_result, torch_to_jax(q_expected), "apply_rotary_pos_emb.q")
    _assert_very_close(k_result, torch_to_jax(k_expected), "apply_rotary_pos_emb.k")


@torch.no_grad()
def test_convnext_block_matches_reference() -> None:
    torch.manual_seed(42)
    block = Qwen3TTSTokenizerV2ConvNeXtBlock(dim=16)
    block.eval()

    lalamo_cfg = default_qwen3_tts_audio_decoder_config(
        hidden_size=32,
        latent_dim=32,
        codebook_dim=32,
        decoder_dim=32,
        num_quantizers=4,
        codebook_size=64,
        upsample_rates=(2, 2),
        upsampling_ratios=(2, 2),
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=96,
        max_position_embeddings=256,
        sliding_window=16,
        num_semantic_quantizers=1,
        enable_debug=False,
    )

    lalamo_block = lalamo_cfg.upsample_block_config.convnext_config.empty(16)
    weights_dict = prepare_state_dict_for_lalamo_loaders(block.state_dict())
    lalamo_block = load_qwen3_tts_convnext_block(lalamo_block, weights_dict, ParameterPath())

    x_nct = torch.randn(2, 16, 24)
    torch_output_nct = block(x_nct)

    x_nsc = _nct_to_nsc(x_nct.numpy())
    lalamo_output_nsc = lalamo_block(jnp.asarray(x_nsc, dtype=jnp.float32))
    lalamo_output_nct = _nsc_to_nct(np.asarray(lalamo_output_nsc))

    _assert_very_close(jnp.asarray(lalamo_output_nct), torch_to_jax(torch_output_nct), "convnext_block")


@torch.no_grad()
def test_decoder_residual_unit_matches_reference() -> None:
    torch.manual_seed(7)
    unit = Qwen3TTSTokenizerV2DecoderDecoderResidualUnit(dim=12, dilation=3)
    unit.eval()

    lalamo_cfg = default_qwen3_tts_audio_decoder_config(
        hidden_size=32,
        latent_dim=32,
        codebook_dim=32,
        decoder_dim=32,
        num_quantizers=4,
        codebook_size=64,
        upsample_rates=(2, 2),
        upsampling_ratios=(2, 2),
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=96,
        max_position_embeddings=256,
        sliding_window=16,
        num_semantic_quantizers=1,
        enable_debug=False,
    )

    lalamo_unit = lalamo_cfg.decoder_block_config.residual_unit_config.empty(dim=12, dilation=3)
    weights_dict = prepare_state_dict_for_lalamo_loaders(unit.state_dict())
    lalamo_unit = load_qwen3_tts_residual_unit(lalamo_unit, weights_dict, ParameterPath())

    x_nct = torch.randn(2, 12, 21)
    torch_output_nct = unit(x_nct)

    x_nsc = _nct_to_nsc(x_nct.numpy())
    lalamo_output_nsc = lalamo_unit(jnp.asarray(x_nsc, dtype=jnp.float32))
    lalamo_output_nct = _nsc_to_nct(np.asarray(lalamo_output_nsc))

    _assert_very_close(jnp.asarray(lalamo_output_nct), torch_to_jax(torch_output_nct), "decoder_residual_unit")


@torch.no_grad()
def test_decoder_block_matches_reference() -> None:
    torch.manual_seed(11)
    config = _tiny_decoder_config()
    block = Qwen3TTSTokenizerV2DecoderDecoderBlock(config, layer_idx=0)
    block.eval()

    lalamo_cfg = default_qwen3_tts_audio_decoder_config(
        hidden_size=config.hidden_size,
        latent_dim=config.latent_dim,
        codebook_dim=config.codebook_dim,
        decoder_dim=config.decoder_dim,
        num_quantizers=config.num_quantizers,
        codebook_size=config.codebook_size,
        upsample_rates=config.upsample_rates,
        upsampling_ratios=config.upsampling_ratios,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        intermediate_size=config.intermediate_size,
        max_position_embeddings=config.max_position_embeddings,
        sliding_window=config.sliding_window,
        attention_bias=config.attention_bias,
        num_semantic_quantizers=config.num_semantic_quantizers,
        enable_debug=False,
    )

    in_dim = config.decoder_dim
    out_dim = config.decoder_dim // 2
    upsample_rate, *_ = config.upsample_rates

    lalamo_block = lalamo_cfg.decoder_block_config.empty(in_dim=in_dim, out_dim=out_dim, upsample_rate=upsample_rate)
    weights_dict = prepare_state_dict_for_lalamo_loaders(block.state_dict())
    lalamo_block = load_qwen3_tts_decoder_block(lalamo_block, weights_dict, ParameterPath())

    x_nct = torch.randn(2, in_dim, 8)
    torch_output_nct = block(x_nct)

    x_nsc = _nct_to_nsc(x_nct.numpy())
    lalamo_output_nsc = lalamo_block(jnp.asarray(x_nsc, dtype=jnp.float32))
    lalamo_output_nct = _nsc_to_nct(np.asarray(lalamo_output_nsc))

    _assert_very_close(jnp.asarray(lalamo_output_nct), torch_to_jax(torch_output_nct), "decoder_block")


@torch.no_grad()
def test_split_rvq_matches_reference() -> None:
    torch.manual_seed(13)
    rvq = SplitResidualVectorQuantizer(
        dimension=16,
        n_q=4,
        n_q_semantic=1,
        bins=32,
        input_dimension=32,
        output_dimension=32,
    )
    rvq.eval()

    lalamo_cfg = default_qwen3_tts_audio_decoder_config(
        hidden_size=32,
        latent_dim=32,
        codebook_dim=32,
        decoder_dim=32,
        num_quantizers=4,
        codebook_size=32,
        upsample_rates=(2, 2),
        upsampling_ratios=(2, 2),
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=96,
        max_position_embeddings=256,
        sliding_window=16,
        num_semantic_quantizers=1,
        enable_debug=False,
    )

    lalamo_rvq = lalamo_cfg.quantizer_config.empty(
        dimension=16,
        n_q=4,
        bins=32,
        input_dimension=32,
        output_dimension=32,
    )

    weights_dict = prepare_state_dict_for_lalamo_loaders(rvq.state_dict())
    lalamo_rvq = load_qwen3_tts_split_rvq(lalamo_rvq, weights_dict, ParameterPath())

    codes_torch = torch.randint(low=0, high=32, size=(2, 4, 7), dtype=torch.int64)
    torch_output_nct = rvq.decode(codes_torch)

    codes_jax = torch_to_jax(codes_torch).astype(jnp.int32)
    lalamo_output_nct = lalamo_rvq.decode(codes_jax)

    _assert_very_close(lalamo_output_nct, torch_to_jax(torch_output_nct), "split_rvq")


@torch.no_grad()
def test_pre_transformer_matches_reference() -> None:
    torch.manual_seed(17)
    config = _tiny_decoder_config()
    torch_model = Qwen3TTSTokenizerV2DecoderTransformerModel(config)
    torch_model.eval()

    lalamo_cfg = default_qwen3_tts_audio_decoder_config(
        hidden_size=config.hidden_size,
        latent_dim=config.latent_dim,
        codebook_dim=config.codebook_dim,
        decoder_dim=config.decoder_dim,
        num_quantizers=config.num_quantizers,
        codebook_size=config.codebook_size,
        upsample_rates=config.upsample_rates,
        upsampling_ratios=config.upsampling_ratios,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        intermediate_size=config.intermediate_size,
        max_position_embeddings=config.max_position_embeddings,
        sliding_window=config.sliding_window,
        rope_theta=config.rope_theta,
        layer_scale_initial_scale=config.layer_scale_initial_scale,
        rms_norm_eps=config.rms_norm_eps,
        attention_bias=config.attention_bias,
        num_semantic_quantizers=config.num_semantic_quantizers,
        enable_debug=False,
    )

    lalamo_model = lalamo_cfg.pre_transformer_config.empty()
    weights_dict = prepare_state_dict_for_lalamo_loaders(torch_model.state_dict())
    lalamo_model = load_qwen3_tts_pre_transformer(lalamo_model, weights_dict, ParameterPath())

    x_torch = torch.randn(2, 11, config.latent_dim)
    torch_output = torch_model(inputs_embeds=x_torch).last_hidden_state

    lalamo_output = lalamo_model(torch_to_jax(x_torch))

    _assert_very_close(lalamo_output, torch_to_jax(torch_output), "pre_transformer")


@torch.no_grad()
def test_audio_decoder_matches_reference() -> None:
    torch.manual_seed(23)
    config = _tiny_decoder_config()
    torch_decoder = Qwen3TTSTokenizerV2Decoder(config)
    torch_decoder.eval()

    lalamo_cfg = default_qwen3_tts_audio_decoder_config(
        hidden_size=config.hidden_size,
        latent_dim=config.latent_dim,
        codebook_dim=config.codebook_dim,
        decoder_dim=config.decoder_dim,
        num_quantizers=config.num_quantizers,
        codebook_size=config.codebook_size,
        upsample_rates=config.upsample_rates,
        upsampling_ratios=config.upsampling_ratios,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        intermediate_size=config.intermediate_size,
        max_position_embeddings=config.max_position_embeddings,
        sliding_window=config.sliding_window,
        rope_theta=config.rope_theta,
        layer_scale_initial_scale=config.layer_scale_initial_scale,
        rms_norm_eps=config.rms_norm_eps,
        attention_bias=config.attention_bias,
        num_semantic_quantizers=config.num_semantic_quantizers,
        enable_debug=False,
    )

    lalamo_decoder = lalamo_cfg.empty()
    weights_dict = prepare_state_dict_for_lalamo_loaders(torch_decoder.state_dict())
    lalamo_decoder = load_audio_decoder(lalamo_decoder, weights_dict, ParameterPath())

    codes_torch = torch.randint(
        low=0,
        high=config.codebook_size,
        size=(2, config.num_quantizers, 12),
        dtype=torch.int64,
    )

    torch_output_nct = torch_decoder(codes_torch)

    codes_jax = torch_to_jax(codes_torch).astype(jnp.int32)
    lalamo_output_nsc = lalamo_decoder(codes_jax)
    lalamo_output_nct = _nsc_to_nct(np.asarray(lalamo_output_nsc))

    _assert_very_close(jnp.asarray(lalamo_output_nct), torch_to_jax(torch_output_nct), "audio_decoder")


@torch.no_grad()
def test_audio_decoder_chunked_decode_matches_reference() -> None:
    torch.manual_seed(29)
    config = _tiny_decoder_config()
    torch_decoder = Qwen3TTSTokenizerV2Decoder(config)
    torch_decoder.eval()

    lalamo_cfg = default_qwen3_tts_audio_decoder_config(
        hidden_size=config.hidden_size,
        latent_dim=config.latent_dim,
        codebook_dim=config.codebook_dim,
        decoder_dim=config.decoder_dim,
        num_quantizers=config.num_quantizers,
        codebook_size=config.codebook_size,
        upsample_rates=config.upsample_rates,
        upsampling_ratios=config.upsampling_ratios,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        intermediate_size=config.intermediate_size,
        max_position_embeddings=config.max_position_embeddings,
        sliding_window=config.sliding_window,
        rope_theta=config.rope_theta,
        layer_scale_initial_scale=config.layer_scale_initial_scale,
        rms_norm_eps=config.rms_norm_eps,
        attention_bias=config.attention_bias,
        num_semantic_quantizers=config.num_semantic_quantizers,
        enable_debug=False,
    )

    lalamo_decoder = lalamo_cfg.empty()
    weights_dict = prepare_state_dict_for_lalamo_loaders(torch_decoder.state_dict())
    lalamo_decoder = load_audio_decoder(lalamo_decoder, weights_dict, ParameterPath())

    codes_torch = torch.randint(
        low=0,
        high=config.codebook_size,
        size=(1, config.num_quantizers, 31),
        dtype=torch.int64,
    )

    torch_output_nct = torch_decoder.chunked_decode(codes_torch, chunk_size=10, left_context_size=3)

    codes_jax = torch_to_jax(codes_torch).astype(jnp.int32)
    lalamo_output_nsc = lalamo_decoder.chunked_decode(codes_jax, chunk_size=10, left_context_size=3)
    lalamo_output_nct = _nsc_to_nct(np.asarray(lalamo_output_nsc))

    _assert_very_close(jnp.asarray(lalamo_output_nct), torch_to_jax(torch_output_nct), "audio_decoder.chunked_decode")


def _import_reference_talker_class():
    module_name = "tests.tts.qwen3.reference.inference.qwen3_tts_tokenizer"
    if module_name not in sys.modules:
        module = types.ModuleType(module_name)

        class _DummyTokenizer:
            pass

        module.Qwen3TTSTokenizer = _DummyTokenizer
        sys.modules[module_name] = module

    from tests.tts.qwen3.reference.core.models.modeling_qwen3_tts import Qwen3TTSTalkerForConditionalGeneration

    return Qwen3TTSTalkerForConditionalGeneration


def _tiny_talker_config():
    from tests.tts.qwen3.reference.core.models.configuration_qwen3_tts import (
        Qwen3TTSTalkerCodePredictorConfig,
        Qwen3TTSTalkerConfig,
    )

    predictor_config = Qwen3TTSTalkerCodePredictorConfig(
        vocab_size=48,
        hidden_size=12,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=3,
        num_key_value_heads=3,
        head_dim=4,
        max_position_embeddings=128,
        attention_bias=True,
        use_sliding_window=False,
        sliding_window=32,
        num_code_groups=4,
    )
    return Qwen3TTSTalkerConfig(
        code_predictor_config=predictor_config,
        vocab_size=48,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=128,
        attention_bias=True,
        rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 0], "interleaved": False},
        use_sliding_window=False,
        sliding_window=32,
        num_code_groups=4,
        text_vocab_size=96,
        text_hidden_size=16,
        codec_pad_id=40,
        codec_bos_id=41,
        codec_eos_token_id=42,
        codec_think_id=43,
        codec_nothing_id=44,
        codec_think_bos_id=45,
        codec_think_eos_id=46,
    )


def _build_lalamo_text_decoder(torch_talker) -> tuple[object, object]:
    talker_cfg = torch_talker.config
    predictor_cfg = talker_cfg.code_predictor_config
    lalamo_cfg = default_qwen3_tts_text_decoder_config(
        precision=jnp.float32,
        talker_vocab_size=talker_cfg.vocab_size,
        text_vocab_size=talker_cfg.text_vocab_size,
        talker_hidden_size=talker_cfg.hidden_size,
        text_hidden_size=talker_cfg.text_hidden_size,
        talker_intermediate_size=talker_cfg.intermediate_size,
        talker_num_hidden_layers=talker_cfg.num_hidden_layers,
        talker_num_attention_heads=talker_cfg.num_attention_heads,
        talker_num_key_value_heads=talker_cfg.num_key_value_heads,
        talker_head_dim=talker_cfg.head_dim,
        talker_max_position_embeddings=talker_cfg.max_position_embeddings,
        talker_rope_theta=talker_cfg.rope_theta,
        talker_rms_norm_eps=talker_cfg.rms_norm_eps,
        talker_attention_bias=talker_cfg.attention_bias,
        talker_sliding_window_sizes=(None,) * talker_cfg.num_hidden_layers,
        predictor_hidden_size=predictor_cfg.hidden_size,
        predictor_intermediate_size=predictor_cfg.intermediate_size,
        predictor_num_hidden_layers=predictor_cfg.num_hidden_layers,
        predictor_num_attention_heads=predictor_cfg.num_attention_heads,
        predictor_num_key_value_heads=predictor_cfg.num_key_value_heads,
        predictor_head_dim=predictor_cfg.head_dim,
        predictor_max_position_embeddings=predictor_cfg.max_position_embeddings,
        predictor_rope_theta=predictor_cfg.rope_theta,
        predictor_rms_norm_eps=predictor_cfg.rms_norm_eps,
        predictor_attention_bias=predictor_cfg.attention_bias,
        predictor_sliding_window_sizes=(None,) * predictor_cfg.num_hidden_layers,
        predictor_vocab_size=predictor_cfg.vocab_size,
        num_code_groups=talker_cfg.num_code_groups,
        max_new_tokens=16,
        codec_bos_id=talker_cfg.codec_bos_id,
        codec_eos_token_id=talker_cfg.codec_eos_token_id,
        codec_pad_id=talker_cfg.codec_pad_id,
        codec_think_id=talker_cfg.codec_think_id,
        codec_nothing_id=talker_cfg.codec_nothing_id,
        codec_think_bos_id=talker_cfg.codec_think_bos_id,
        codec_think_eos_id=talker_cfg.codec_think_eos_id,
        tts_bos_token_id=90,
        tts_eos_token_id=91,
        tts_pad_token_id=92,
    )

    weights_dict = prepare_state_dict_for_lalamo_loaders(torch_talker.state_dict(), prefix="talker")
    lalamo_text_decoder = load_qwen3_tts_text_decoder(lalamo_cfg.empty(), weights_dict, ParameterPath())
    return lalamo_cfg, lalamo_text_decoder


@torch.no_grad()
def test_text_decoder_projection_matches_reference() -> None:
    torch.manual_seed(101)
    TalkerClass = _import_reference_talker_class()
    torch_talker = TalkerClass(_tiny_talker_config())
    torch_talker.eval()

    _, lalamo_text_decoder = _build_lalamo_text_decoder(torch_talker)

    text_tokens = torch.randint(low=0, high=torch_talker.config.text_vocab_size, size=(1, 9), dtype=torch.int64)

    torch_projected = torch_talker.text_projection(torch_talker.get_text_embeddings()(text_tokens))
    lalamo_projected = lalamo_text_decoder._project_text_embeddings(torch_to_jax(text_tokens).astype(jnp.int32))

    _assert_very_close(lalamo_projected, torch_to_jax(torch_projected), "text_decoder.text_projection")


@torch.no_grad()
def test_text_decoder_decode_utterance_matches_reference_algorithm() -> None:
    torch.manual_seed(103)
    TalkerClass = _import_reference_talker_class()
    torch_talker = TalkerClass(_tiny_talker_config())
    torch_talker.eval()

    lalamo_cfg, lalamo_text_decoder = _build_lalamo_text_decoder(torch_talker)

    text_tokens = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]], dtype=torch.int64)
    greedy = GreedyPolicy()
    lalamo_codes = lalamo_text_decoder.decode_utterance(
        torch_to_jax(text_tokens).astype(jnp.int32),
        sampling_policy=greedy,
        key=jax.random.PRNGKey(0),
    )

    def project_text(x: torch.Tensor) -> torch.Tensor:
        return torch_talker.text_projection(torch_talker.get_text_embeddings()(x))

    text_hidden = project_text(text_tokens)
    tts_special = project_text(torch.tensor([[90, 91, 92]], dtype=torch.int64))
    tts_bos_embed = tts_special[:, 0:1]
    tts_eos_embed = tts_special[:, 1:2]
    tts_pad_embed = tts_special[:, 2:3]

    codec_prefill_ids = torch.tensor(
        [[
            torch_talker.config.codec_nothing_id,
            torch_talker.config.codec_think_bos_id,
            torch_talker.config.codec_think_eos_id,
            torch_talker.config.codec_pad_id,
            torch_talker.config.codec_bos_id,
        ]],
        dtype=torch.int64,
    )
    codec_prefill_embed = torch_talker.get_input_embeddings()(codec_prefill_ids)

    role_hidden = text_hidden[:, :3]
    first_text_hidden = text_hidden[:, 3:4]
    trailing_text_hidden = torch.cat([text_hidden[:, 4:], tts_eos_embed], dim=1)
    codec_prompt = torch.cat([tts_pad_embed.repeat(1, 3, 1), tts_bos_embed], dim=1) + codec_prefill_embed[:, :-1]
    talker_inputs = torch.cat([role_hidden, codec_prompt, first_text_hidden + codec_prefill_embed[:, -1:]], dim=1)

    generated_steps: list[torch.Tensor] = []
    for step in range(lalamo_cfg.max_new_tokens):
        talker_hidden = torch_talker.model(inputs_embeds=talker_inputs, use_cache=False).last_hidden_state
        last_hidden = talker_hidden[:, -1:]
        first_logits = torch_talker.codec_head(last_hidden)[:, 0]
        first_codec = torch.argmax(first_logits, dim=-1)
        if int(first_codec.item()) == torch_talker.config.codec_eos_token_id:
            break

        first_codec_embed = torch_talker.get_input_embeddings()(first_codec[:, None])
        predictor_inputs = torch.cat([last_hidden, first_codec_embed], dim=1)
        step_ids = [first_codec]
        step_embeds = [first_codec_embed]

        for idx in range(torch_talker.config.num_code_groups - 1):
            predictor_hidden = torch_talker.code_predictor.model(
                inputs_embeds=torch_talker.code_predictor.small_to_mtp_projection(predictor_inputs),
                use_cache=False,
            ).last_hidden_state
            predictor_logits = torch_talker.code_predictor.lm_head[idx](predictor_hidden[:, -1])
            next_codec = torch.argmax(predictor_logits, dim=-1)
            next_codec_embed = torch_talker.code_predictor.model.codec_embedding[idx](next_codec[:, None])

            step_ids.append(next_codec)
            step_embeds.append(next_codec_embed)
            if idx + 1 < torch_talker.config.num_code_groups - 1:
                predictor_inputs = torch.cat([predictor_inputs, next_codec_embed], dim=1)

        generated_steps.append(torch.stack(step_ids, dim=1)[0])

        next_input = torch.stack([embed[:, 0] for embed in step_embeds], dim=1).sum(dim=1, keepdim=True)
        if step < trailing_text_hidden.shape[1]:
            next_input = next_input + trailing_text_hidden[:, step : step + 1]
        else:
            next_input = next_input + tts_pad_embed
        talker_inputs = torch.cat([talker_inputs, next_input], dim=1)

    if generated_steps:
        torch_codes = torch.stack(generated_steps, dim=1)
    else:
        torch_codes = torch.zeros((torch_talker.config.num_code_groups, 0), dtype=torch.int64)

    _assert_very_close(lalamo_codes, torch_to_jax(torch_codes), "text_decoder.decode_utterance")


def test_qwen3_config_uses_real_text_decoder_when_talker_present(tmp_path) -> None:
    decoder_cfg = _tiny_decoder_config()
    talker_cfg = _tiny_talker_config()

    raw_config = {
        "model_type": "qwen3_tts",
        "torch_dtype": "float32",
        "decode_upsample_rate": math.prod((*decoder_cfg.upsample_rates, *decoder_cfg.upsampling_ratios)),
        "encode_downsample_rate": math.prod((*decoder_cfg.upsample_rates, *decoder_cfg.upsampling_ratios)),
        "encoder_valid_num_quantizers": decoder_cfg.num_quantizers,
        "input_sample_rate": 24000,
        "output_sample_rate": 24000,
        "tts_pad_token_id": 92,
        "tts_bos_token_id": 90,
        "tts_eos_token_id": 91,
        "decoder_config": {
            "attention_dropout": decoder_cfg.attention_dropout,
            "attention_bias": decoder_cfg.attention_bias,
            "codebook_dim": decoder_cfg.codebook_dim,
            "codebook_size": decoder_cfg.codebook_size,
            "decoder_dim": decoder_cfg.decoder_dim,
            "head_dim": decoder_cfg.head_dim,
            "hidden_act": decoder_cfg.hidden_act,
            "hidden_size": decoder_cfg.hidden_size,
            "intermediate_size": decoder_cfg.intermediate_size,
            "latent_dim": decoder_cfg.latent_dim,
            "layer_scale_initial_scale": decoder_cfg.layer_scale_initial_scale,
            "max_position_embeddings": decoder_cfg.max_position_embeddings,
            "num_attention_heads": decoder_cfg.num_attention_heads,
            "num_hidden_layers": decoder_cfg.num_hidden_layers,
            "num_key_value_heads": decoder_cfg.num_key_value_heads,
            "num_quantizers": decoder_cfg.num_quantizers,
            "num_semantic_quantizers": decoder_cfg.num_semantic_quantizers,
            "rms_norm_eps": decoder_cfg.rms_norm_eps,
            "semantic_codebook_size": 32,
            "rope_theta": decoder_cfg.rope_theta,
            "sliding_window": decoder_cfg.sliding_window,
            "upsample_rates": list(decoder_cfg.upsample_rates),
            "upsampling_ratios": list(decoder_cfg.upsampling_ratios),
            "vector_quantization_hidden_dimension": 32,
        },
        "talker_config": talker_cfg.to_dict(),
    }

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(raw_config))

    foreign_cfg = Qwen3TTSTokenizer12HzConfig.from_json(config_path)
    tts_cfg = foreign_cfg.to_tts_config(None, jnp.float32, jnp.float32)

    assert isinstance(tts_cfg.text_decoder_config, Qwen3TTSTextDecoderConfig)
    tts_model = tts_cfg.empty()
    assert isinstance(tts_model.text_decoder, Qwen3TTSTextDecoder)
