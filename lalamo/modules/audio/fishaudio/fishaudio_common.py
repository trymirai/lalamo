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

from .fishaudio_consts import (
    DEFAULT_FISH_AUDIO_SAMPLING_TEMPERATURE,
    DEFAULT_FISH_AUDIO_SAMPLING_TOP_P,
    FISH_TIKTOKEN_PATTERN,
    IM_END_TOKEN,
    _default_audio_codec_config,
)


def default_fishaudio_sampling_policy() -> SamplingPolicy:
    return make_policy(
        temperature=DEFAULT_FISH_AUDIO_SAMPLING_TEMPERATURE,
        top_p=DEFAULT_FISH_AUDIO_SAMPLING_TOP_P,
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
        pat_str=FISH_TIKTOKEN_PATTERN,
        mergeable_ranks=mergeable_ranks,
        special_tokens=all_special_tokens_with_ids,
    )

    inference_special_tokens = FishAudioSpecialInferenceTokens(
        semantic_begin_id=semantic_begin_id,
        semantic_end_id=semantic_end_id,
        im_end_token_id=all_special_tokens_with_ids[IM_END_TOKEN],
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
