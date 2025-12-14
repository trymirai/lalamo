import shutil
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from fish_speech.models.text2semantic.llama import (
    BaseModelArgs,
    DualARModelArgs,
    DualARTransformer,
    NaiveModelArgs,
)
from fish_speech.tokenizer import IM_END_TOKEN, FishTokenizer
from jaxtyping import Array, Float, Int
from tokenizers import Tokenizer
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.integrations.tiktoken import convert_tiktoken_to_fast

from lalamo.modules import (
    AudioDecoder,
)
from lalamo.modules.audio.audio_decoder import AudioDecoderConfig
from lalamo.modules.audio.text_decoder import TextDecoder, TextDecoderConfig
from lalamo.modules.torch_interop import jax_to_torch, torch_to_jax

dac_config = {
    "_target_": "fish_speech.models.dac.modded_dac.DAC",
    "decoder_dim": 1536,
    "decoder_rates": [8, 8, 4, 2],
    "decoder_transformer_layers": [4, 0, 0, 0],
    "encoder_dim": 64,
    "encoder_rates": [2, 4, 8, 8],
    "encoder_transformer_layers": [0, 0, 0, 4],
    "quantizer": {
        "_target_": "fish_speech.models.dac.rvq.DownsampleResidualVectorQuantize",
        "codebook_dim": 8,
        "codebook_size": 1024,
        "downsample_factor": [2, 2],
        "input_dim": 1024,
        "n_codebooks": 9,
        "post_module": {
            "_target_": "fish_speech.models.dac.modded_dac.WindowLimitedTransformer",
            "causal": True,
            "config": {
                "_target_": "fish_speech.models.dac.modded_dac.ModelArgs",
                "attn_dropout_rate": 0.1,
                "block_size": 4096,
                "channels_first": True,
                "dim": 1024,
                "dropout_rate": 0.1,
                "head_dim": 64,
                "intermediate_size": 3072,
                "n_head": 16,
                "n_layer": 8,
                "n_local_heads": -1,
                "norm_eps": "1e-5",
                "rope_base": 10000,
            },
            "input_dim": 1024,
            "window_size": 128,
        },
        "pre_module": {
            "_target_": "fish_speech.models.dac.modded_dac.WindowLimitedTransformer",
            "causal": True,
            "config": {
                "_target_": "fish_speech.models.dac.modded_dac.ModelArgs",
                "attn_dropout_rate": 0.1,
                "block_size": 4096,
                "channels_first": True,
                "dim": 1024,
                "dropout_rate": 0.1,
                "head_dim": 64,
                "intermediate_size": 3072,
                "n_head": 16,
                "n_layer": 8,
                "n_local_heads": -1,
                "norm_eps": "1e-5",
                "rope_base": 10000,
            },
            "input_dim": 1024,
            "window_size": 128,
        },
        "quantizer_dropout": 0.5,
        "semantic_codebook_size": 4096,
    },
    "sample_rate": 44100,
    "transformer_general_config": {
        "_partial_": True,
        "_target_": "fish_speech.models.dac.modded_dac.ModelArgs",
        "attn_dropout_rate": 0.1,
        "block_size": 16384,
        "channels_first": True,
        "dropout_rate": 0.1,
        "head_dim": 64,
        "n_local_heads": -1,
        "norm_eps": "1e-5",
        "rope_base": 10000,
    },
}


def load_tokenizer_from_fish_audio(path_to_chkpt: str) -> Tokenizer:
    output_temp_dir = tempfile.mkdtemp()
    try:
        fishspeech_tokenizer = FishTokenizer.from_pretrained(path_to_chkpt)

        convert_tiktoken_to_fast(fishspeech_tokenizer.tkt_model, output_temp_dir)
        tokenizer = Tokenizer.from_file(output_temp_dir + "/tokenizer.json")
        return tokenizer
    finally:
        shutil.rmtree(output_temp_dir)


def logits_to_probs(
    logits,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    repetition_penalty: torch.Tensor,
    previous_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Apply repetition penalty
    if previous_tokens is not None:
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=-1, index=previous_tokens)
        score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
        logits.scatter_(dim=-1, index=previous_tokens, src=score)

    # Apply top-p sampling
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[0] = False  # keep at least one option
    indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
    logits = logits.masked_fill(indices_to_remove, -float("Inf"))
    logits = logits / torch.clip(temperature, min=1e-5)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample(
    logits,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    repetition_penalty: torch.Tensor,
    previous_tokens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    probs = logits_to_probs(
        logits=logits[0, -1],
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        previous_tokens=previous_tokens,
    )
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def decode_one_token_ar_fishaudio(
    model: DualARTransformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    repetition_penalty: torch.Tensor,
    previous_tokens: Optional[torch.Tensor] = None,
    argmax_decoding=False,
) -> torch.Tensor:
    # print(x, torch.count_nonzero(vq_masks))
    forward_result = model.forward_generate(
        x,
        input_pos,
    )
    logits = forward_result.logits  # [:, -1:]
    hidden_states = forward_result.hidden_states  # [:, -1:]

    codebooks = [
        sample(
            logits,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            previous_tokens=(previous_tokens[:, 0] if previous_tokens is not None else None),
        )[0]
    ]

    # Only clear cache for fast_layers, avoid clearing main model cache
    for layer in model.fast_layers:
        if hasattr(layer, "attention") and hasattr(layer.attention, "kv_cache"):
            layer.attention.kv_cache.k_cache.fill_(0)
            layer.attention.kv_cache.v_cache.fill_(0)

    input_pos = torch.tensor([0], device=hidden_states.device, dtype=torch.long)
    model.forward_generate_fast(hidden_states, input_pos)
    a = codebooks[0] - model.tokenizer.semantic_begin_id
    a[a < 0] = 0
    hidden_states = model.fast_embeddings(a)
    codebooks.append(a)

    for codebook_idx in range(1, model.config.num_codebooks):
        input_pos = torch.tensor([codebook_idx], device=hidden_states.device, dtype=torch.long)
        logits = model.forward_generate_fast(hidden_states, input_pos)

        short_logits = logits[:, :, :1024]

        # Convert logits to probs
        if argmax_decoding:
            a = short_logits.argmax(dim=2)[0]
        else:
            a = sample(
                short_logits,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                previous_tokens=(previous_tokens[codebook_idx + 1] if previous_tokens is not None else None),
            )[0]

        hidden_states = model.fast_embeddings(a)
        codebooks.append(a)

    codebooks = torch.stack(codebooks, dim=1)

    # Only delete references, let Python GC handle cleanup
    del logits, hidden_states, forward_result

    return codebooks.T


def decode_n_tokens(
    model: DualARTransformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    repetition_penalty: torch.Tensor,
):
    previous_tokens = torch.zeros(
        (model.config.num_codebooks + 1, model.config.max_seq_len),
        dtype=torch.int,
        device=cur_token.device,
    )

    for i in range(num_new_tokens):
        # MY_DBG
        print(f"generating token {i}")

        # We need to get windowed repeat penalty
        win_size = 16
        if i < win_size:
            window = previous_tokens[:, :win_size]
        else:
            window = previous_tokens[:, i - win_size : i]

        with sdpa_kernel(SDPBackend.MATH):  # Actually better for Inductor to codegen attention here
            next_token = decode_one_token_ar_fishaudio(
                model=model,
                x=cur_token,
                input_pos=input_pos,
                previous_tokens=window,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            ).clone()

        input_pos += 1
        cur_token = next_token.view(1, model.config.num_codebooks + 1, -1)
        previous_tokens[:, i : i + 1] = next_token.view(model.config.num_codebooks + 1, -1)

        if cur_token[0, 0, -1] == model.tokenizer.get_token_id(IM_END_TOKEN):
            break

    # Only clean up the large tensor
    del cur_token

    return previous_tokens[:, : i + 1]


@torch.no_grad()
@torch.inference_mode()
def generate(
    *,
    model: DualARTransformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    num_samples: int = 1,
    **sampling_kwargs,
):
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(1)
    prompt = prompt[None].repeat(num_samples, 1, 1)

    if T >= model.config.max_seq_len:
        raise ValueError(f"Input sequence length {T} exceeds max_seq_len {model.config.max_seq_len}")

    if max_new_tokens:
        if T + max_new_tokens > model.config.max_seq_len:
            max_new_tokens = model.config.max_seq_len - T

        T_new = T + max_new_tokens
    else:
        T_new = model.config.max_seq_len
        max_new_tokens = T_new - T

    device, dtype = prompt.device, prompt.dtype

    # Critical fix: Only set up cache on first run or when necessary
    if not hasattr(model, "_cache_setup_done") or not model._cache_setup_done:
        with torch.device(device):
            model.setup_caches(
                max_batch_size=1,  # Fixed to 1, avoid dynamic changes
                max_seq_len=model.config.max_seq_len,
                dtype=next(model.parameters()).dtype,
            )
        model._cache_setup_done = True

    codebook_dim = 1 + model.config.num_codebooks

    # Create new tensor each time, but try to reuse memory
    input_pos = torch.arange(0, T, device=device, dtype=torch.long)
    empty = torch.empty((codebook_dim, model.config.max_seq_len), dtype=dtype, device=device)
    empty[:, :T] = prompt
    seq = empty

    # Use pre-created fixed parameter tensors
    temperature = getattr(model, "fixed_temperature", torch.tensor(0.8, device=device, dtype=torch.float))
    top_p = getattr(model, "fixed_top_p", torch.tensor(0.8, device=device, dtype=torch.float))
    repetition_penalty = getattr(
        model,
        "fixed_repetition_penalty",
        torch.tensor(1.1, device=device, dtype=torch.float),
    )

    # If different parameter values are needed, directly modify existing tensors
    temp_val = sampling_kwargs.get("temperature", 0.7)
    top_p_val = sampling_kwargs.get("top_p", 0.7)
    rep_val = sampling_kwargs.get("repetition_penalty", 1.5)

    if abs(temperature.item() - temp_val) > 1e-6:
        temperature.fill_(temp_val)
    if abs(top_p.item() - top_p_val) > 1e-6:
        top_p.fill_(top_p_val)
    if abs(repetition_penalty.item() - rep_val) > 1e-6:
        repetition_penalty.fill_(rep_val)

    first_token = decode_one_token_ar_fishaudio(
        model,
        prompt.view(1, codebook_dim, -1),
        input_pos,
        temperature,
        top_p,
        repetition_penalty,
    )
    seq[:, T : T + 1] = first_token

    # Recreate input_pos
    input_pos = torch.tensor([T], device=device, dtype=torch.int)

    x = decode_n_tokens(
        model,
        first_token.view(1, codebook_dim, -1),
        input_pos,
        max_new_tokens - 1,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    seq = seq[:, : T + 1 + x.size(1)]
    seq[:, T + 1 :] = x

    # Clean up temporary variables
    del first_token, x, prompt, empty, input_pos

    return seq


def default_fish_audio_audio_decoder_config() -> "FishAudioAudioDecoderConfig":
    return FishAudioAudioDecoderConfig()


def load_fish_audio_audio_decoder(
    chkpt_path: str | Path,
) -> "FishAudioAudioDecoder":
    config = default_fish_audio_audio_decoder_config()
    decoder = FishAudioAudioDecoder(config)

    return decoder


@dataclass(frozen=True)
class FishAudioAudioDecoderConfig(AudioDecoderConfig):
    pass


class FishAudioAudioDecoder(AudioDecoder):
    pass


@dataclass(frozen=True)
class FishAudioTextDecoderConfig_Foreign(TextDecoderConfig):
    fish_config: DualARModelArgs

    @classmethod
    def from_config_file(cls, path_to_config: str | Path) -> "FishAudioTextDecoderConfig_Foreign":
        config: DualARModelArgs | NaiveModelArgs = BaseModelArgs.from_pretrained(str(path_to_config))
        assert isinstance(config, DualARModelArgs)
        return FishAudioTextDecoderConfig_Foreign(fish_config=config)

    def _load_weights_to_fish_model(self, fish_model: DualARTransformer, path_to_model: str | Path) -> None:
        weights = torch.load(
            Path(path_to_model) / "model.pth",
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )

        if "state_dict" in weights:
            weights = weights["state_dict"]

        if next(iter(weights.keys())).startswith("model."):
            new_weights = OrderedDict()
            for k, v in weights.items():
                new_weights[k.replace("model.", "")] = v
            weights = new_weights

        # Remove audio related weights
        for k in list(weights.keys()):
            if "audio_" in k:
                weights.pop(k)

        fish_model.load_state_dict(weights, strict=False, assign=True)

    def load_model(
        self, path_to_model: str | Path, device: str = "cpu", precision: torch.dtype = torch.bfloat16
    ) -> "FishAudioTextDecoder_Foreign":
        tokenizer = FishTokenizer.from_pretrained(str(path_to_model))
        fish_model = DualARTransformer(self.fish_config, tokenizer)
        fish_model = fish_model.to(device=device, dtype=precision)

        self._load_weights_to_fish_model(fish_model, path_to_model)

        return FishAudioTextDecoder_Foreign(config=self, fish_model=fish_model)


class FishAudioTextDecoder_Foreign(TextDecoder):
    fish_model: DualARTransformer

    def __call__(
        self, text_tokens: Int[Array, "batch tokens"], input_pos: Int[Array, "batch tokens"] | None = None
    ) -> Float[Array, "batch_size tokens hidden_size"]:
        text_tokens_torch = jax_to_torch(text_tokens)

        _, n_tokens = text_tokens_torch.shape

        assert isinstance(self.config, FishAudioTextDecoderConfig_Foreign)
        values = torch.zeros((self.config.fish_config.num_codebooks + 1, n_tokens), dtype=torch.int)
        values[0] = text_tokens_torch

        time_steps = text_tokens_torch.shape[1]
        # input_pos = torch.arange(0, time_steps, device=text_tokens_torch.device)

        self.fish_model.setup_caches(
            max_batch_size=1,
            max_seq_len=self.fish_model.config.max_seq_len,
            dtype=next(self.fish_model.parameters()).dtype,
        )

        prompt_length = values.size(1)

        temperature = torch.tensor(0.8008, device=values.device, dtype=torch.bfloat16)
        top_p = torch.tensor(0.8008, device=values.device, dtype=torch.bfloat16)
        repetition_penalty = torch.tensor(1.1016, device=values.device, dtype=torch.bfloat16)

        y = generate(
            model=self.fish_model,
            prompt=values,
            max_new_tokens=0,
            temperature=temperature,  # 0.8008,
            top_p=top_p,  # 0.8008,
            repetition_penalty=repetition_penalty,  # 1.1016,
        )

        codes = y[1:, prompt_length:-1].clone()
        assert (codes >= 0).all(), f"Negative code found"

        return torch_to_jax(codes)
