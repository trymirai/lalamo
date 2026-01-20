import base64
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig
from tiktoken.core import Encoding as TikokenEncoding
from tokenizers import Tokenizer
from transformers.integrations.tiktoken import convert_tiktoken_to_fast

from lalamo.sampling import SamplingPolicy, make_policy
from lalamo.utils import setup_custom_logger

DEFAULT_FISHAUDIO_RANDOM_SEED: int = 123


@dataclass(frozen=True)
class FishaudioConsts:
    """
    Consts and configuration elements that were stored in Fishaudio
    codebase as either global variables or magic consts
    """

    DEFAULT_FISH_AUDIO_SAMPLING_TEMPERATURE = 0.8008
    DEFAULT_FISH_AUDIO_SAMPLING_TOP_P = 0.8008
    DEFAULT_FISH_AUDIO_REPETITION_PENALTY: float = 1.1016
    SHORT_LOGITS_SIZE: int = 1024
    REPEAT_WINDOW_SIZE: int = 16
    FISH_TIKTOKEN_PATTERN = "|".join(  # noqa: FLY002
        [
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
            r"\p{P}",
            r"[^\r\n\p{L}\p{N}]?\p{L}+",
            r"\p{N}",
            r" ?[^\s\p{L}\p{N}]+[\r\n]*",
            r"\s*[\r\n]+",
            r"\s+(\?!\S)",
            r"\s+",
        ],
    )
    IM_END_TOKEN = "<|im_end|>"


# NOTE: in fish-speech repo this was a YAML config stored right in the code
_default_audio_codec_config = {
    "_target_": "fish_speech.models.dac.modded_dac.DAC",
    "sample_rate": 44100,
    "encoder_dim": 64,
    "encoder_rates": [2, 4, 8, 8],
    "decoder_dim": 1536,
    "decoder_rates": [8, 8, 4, 2],
    "encoder_transformer_layers": [0, 0, 0, 4],
    "decoder_transformer_layers": [4, 0, 0, 0],
    "transformer_general_config": {
        "_target_": "fish_speech.models.dac.modded_dac.ModelArgs",
        "_partial_": True,
        "block_size": 16384,
        "n_local_heads": -1,
        "head_dim": 64,
        "rope_base": 10000,
        "norm_eps": 1e-5,
        "dropout_rate": 0.1,
        "attn_dropout_rate": 0.1,
        "channels_first": True,
    },
    "quantizer": {
        "_target_": "fish_speech.models.dac.rvq.DownsampleResidualVectorQuantize",
        "input_dim": 1024,
        "n_codebooks": 9,
        "codebook_size": 1024,
        "codebook_dim": 8,
        "quantizer_dropout": 0.5,
        "downsample_factor": [2, 2],
        "post_module": {
            "_target_": "fish_speech.models.dac.modded_dac.WindowLimitedTransformer",
            "causal": True,
            "window_size": 128,
            "input_dim": 1024,
            "config": {
                "_target_": "fish_speech.models.dac.modded_dac.ModelArgs",
                "block_size": 4096,
                "n_layer": 8,
                "n_head": 16,
                "dim": 1024,
                "intermediate_size": 3072,
                "n_local_heads": -1,
                "head_dim": 64,
                "rope_base": 10000,
                "norm_eps": 1e-5,
                "dropout_rate": 0.1,
                "attn_dropout_rate": 0.1,
                "channels_first": True,
            },
        },
        "pre_module": {
            "_target_": "fish_speech.models.dac.modded_dac.WindowLimitedTransformer",
            "causal": True,
            "window_size": 128,
            "input_dim": 1024,
            "config": {
                "_target_": "fish_speech.models.dac.modded_dac.ModelArgs",
                "block_size": 4096,
                "n_layer": 8,
                "n_head": 16,
                "dim": 1024,
                "intermediate_size": 3072,
                "n_local_heads": -1,
                "head_dim": 64,
                "rope_base": 10000,
                "norm_eps": 1e-5,
                "dropout_rate": 0.1,
                "attn_dropout_rate": 0.1,
                "channels_first": True,
            },
        },
        "semantic_codebook_size": 4096,
    },
}

fishaudio_logger = setup_custom_logger(logger_name="fishaudio")


def default_fishaudio_sampling_policy() -> SamplingPolicy:
    return make_policy(
        temperature=FishaudioConsts.DEFAULT_FISH_AUDIO_SAMPLING_TEMPERATURE,
        top_p=FishaudioConsts.DEFAULT_FISH_AUDIO_SAMPLING_TOP_P,
    )


@dataclass(frozen=True)
class FishAudioSpecialInferenceTokens:
    semantic_begin_id: int
    semantic_end_id: int
    im_end_token_id: int


def get_default_fishaudio_dac_config() -> DictConfig:
    return DictConfig(_default_audio_codec_config)


def _load_fishaudio_tiktoken_data(
    tiktoken_path: Path,
    special_tokens: dict[str, int],
) -> tuple[TikokenEncoding, FishAudioSpecialInferenceTokens]:
    def load_tiktoken_bpe(tiktoken_bpe_file: Path) -> dict[bytes, int]:
        data = {}
        with open(tiktoken_bpe_file) as token_file:
            for line in token_file.read().splitlines():
                if not line:
                    continue
                token, rank = line.split()
                if token == "=":
                    continue
                data[base64.b64decode(token)] = int(rank)
        return data

    mergeable_ranks = load_tiktoken_bpe(tiktoken_path)
    special_token_begin = len(mergeable_ranks)
    all_special_tokens_with_ids = {token: special_token_begin + i for i, token in enumerate(special_tokens)}

    semantic_id_to_token_id = {}
    end_idx = 0
    for token in special_tokens:
        if token.startswith("<|semantic:"):
            match_results = re.match(r"<\|semantic:(\d+)\|>", token)
            assert match_results is not None
            idx = int(match_results.group(1))
            semantic_id_to_token_id[idx] = all_special_tokens_with_ids[token]
            end_idx = max(end_idx, idx)

    semantic_begin_id = semantic_id_to_token_id[0]
    semantic_end_id = semantic_id_to_token_id[end_idx]

    tkt_model = TikokenEncoding(
        name=Path(tiktoken_path).stem,
        pat_str=FishaudioConsts.FISH_TIKTOKEN_PATTERN,
        mergeable_ranks=mergeable_ranks,
        special_tokens=all_special_tokens_with_ids,
    )

    inference_special_tokens = FishAudioSpecialInferenceTokens(
        semantic_begin_id=semantic_begin_id,
        semantic_end_id=semantic_end_id,
        im_end_token_id=all_special_tokens_with_ids[FishaudioConsts.IM_END_TOKEN],
    )

    return tkt_model, inference_special_tokens


def load_tokenizer_from_fishaudio_tiktoken(
    path_to_tokenizer: Path,
    path_to_special_tokens: Path,
) -> tuple[Tokenizer, FishAudioSpecialInferenceTokens]:
    output_temp_dir = tempfile.mkdtemp()
    try:
        if path_to_special_tokens.exists():
            with open(path_to_special_tokens) as f:
                all_special_tokens_with_ids = json.load(f)
        else:
            all_special_tokens_with_ids = {}

        tkt_model, special_inference_tokens = _load_fishaudio_tiktoken_data(
            path_to_tokenizer,
            all_special_tokens_with_ids,
        )

        convert_tiktoken_to_fast(tkt_model, output_temp_dir)
        tokenizer = Tokenizer.from_file(output_temp_dir + "/tokenizer.json")
        return tokenizer, special_inference_tokens
    finally:
        shutil.rmtree(output_temp_dir)
