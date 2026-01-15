from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Self

import huggingface_hub
import torch
from fish_speech.models.dac.modded_dac import DAC
from fish_speech.models.dac.rvq import ResidualVectorQuantize
from fish_speech.models.text2semantic.llama import (
    BaseModelArgs,
    BaseTransformerForwardResult,
    DualARModelArgs,
    DualARTransformer,
    NaiveModelArgs,
    TransformerForwardResult,
)
from fish_speech.tokenizer import IM_END_TOKEN, FishTokenizer
from hydra.utils import instantiate
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray
from omegaconf import DictConfig
from torch._tensor import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

from lalamo.common import ParameterTree
from lalamo.modules.audio.fishaudio.fishaudio_common import get_default_fishaudio_dac_config
from lalamo.modules.audio.fishaudio.fishaudio_sampling import FishAudioSamplingParams
from lalamo.modules.audio.text_decoder import TTSTextDecoder
from lalamo.modules.audio.text_to_speech import TTSAudioDecoder
from lalamo.modules.audio.utils import DTypeConvert
from lalamo.modules.torch_interop import jax_to_torch, torch_to_jax


def try_locate_fish_audio_model_path() -> Optional[Path]:
    # TODO: (peter.glushkov) replace this one with actual ModelSpec
    fish_audiod_repo_id = "fishaudio/openaudio-s1-mini"

    repos = huggingface_hub.scan_cache_dir().repos
    try:
        fish_audio_model_info = next(filter(lambda repo: repo.repo_id == fish_audiod_repo_id, repos))

        api = huggingface_hub.HfApi()
        cache_info = api.model_info(fish_audiod_repo_id)
        commit_hash = cache_info.sha
        return fish_audio_model_info.repo_path / "snapshots" / str(commit_hash)
    except StopIteration:
        return None


class FromFishAudioRepo:
    """
    Current class contains code taken from FishAudio repo with minor cosmetic changes
    https://github.com/fishaudio/fish-speech/blob/main/fish_speech/models/text2semantic/inference.py
    """

    @staticmethod
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
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, -float("Inf"))
        logits = logits / torch.clip(temperature, min=1e-5)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs

    @staticmethod
    def multinomial_sample_one_no_sync(
        probs_sort,
    ):  # Does multinomial sampling without a cuda synchronization
        q = torch.empty_like(probs_sort).exponential_(1)
        return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

    @staticmethod
    def sample(
        logits,
        temperature: torch.Tensor,
        top_p: torch.Tensor,
        repetition_penalty: torch.Tensor,
        previous_tokens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probs = FromFishAudioRepo.logits_to_probs(
            logits=logits[0, -1],
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            previous_tokens=previous_tokens,
        )
        idx_next = FromFishAudioRepo.multinomial_sample_one_no_sync(probs)
        return idx_next, probs

    @staticmethod
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
        forward_result: TransformerForwardResult = model.forward_generate(
            x,
            input_pos,
        )
        assert isinstance(forward_result, BaseTransformerForwardResult)
        logits = forward_result.logits  # [:, -1:]
        hidden_states = forward_result.hidden_states  # [:, -1:]

        if argmax_decoding:
            codebooks = [logits.argmax(dim=2)[0]]
        else:
            codebooks = [
                FromFishAudioRepo.sample(
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
                a = FromFishAudioRepo.sample(
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

    @staticmethod
    def decode_n_tokens(
        model: DualARTransformer,
        cur_token: torch.Tensor,
        input_pos: torch.Tensor,
        num_new_tokens: int,
        temperature: torch.Tensor,
        top_p: torch.Tensor,
        repetition_penalty: torch.Tensor,
        argmax_decoding: bool = False,
    ):
        previous_tokens = torch.zeros(
            (model.config.num_codebooks + 1, model.config.max_seq_len),
            dtype=torch.int,
            device=cur_token.device,
        )

        final_idx = 0
        for i in range(num_new_tokens):
            # We need to get windowed repeat penalty
            win_size = 16
            if i < win_size:
                window = previous_tokens[:, :win_size]
            else:
                window = previous_tokens[:, i - win_size : i]

            with sdpa_kernel(SDPBackend.MATH):  # Actually better for Inductor to codegen attention here
                next_token = FromFishAudioRepo.decode_one_token_ar_fishaudio(
                    model=model,
                    x=cur_token,
                    input_pos=input_pos,
                    previous_tokens=window,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    argmax_decoding=argmax_decoding,
                ).clone()

            input_pos += 1
            cur_token = next_token.view(1, model.config.num_codebooks + 1, -1)
            previous_tokens[:, i : i + 1] = next_token.view(model.config.num_codebooks + 1, -1)
            final_idx = i

            # TODO: remove after debugging is done
            print(f"{i} : code={cur_token[0]}")

            if cur_token[0, 0, -1] == model.tokenizer.get_token_id(IM_END_TOKEN):
                break

        # Only clean up the large tensor
        del cur_token

        return previous_tokens[:, : final_idx + 1]

    @staticmethod
    @torch.no_grad()
    @torch.inference_mode()
    def generate(
        *,
        model: DualARTransformer,
        prompt: torch.Tensor,
        max_new_tokens: int,
        num_samples: int = 1,
        argmax_decoding: bool = False,
        **sampling_kwargs,
    ) -> Tensor:
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

        first_token = FromFishAudioRepo.decode_one_token_ar_fishaudio(
            model,
            prompt.view(1, codebook_dim, -1),
            input_pos,
            temperature,
            top_p,
            repetition_penalty,
            argmax_decoding=argmax_decoding,
        )
        seq[:, T : T + 1] = first_token

        # Recreate input_pos
        input_pos = torch.tensor([T], device=device, dtype=torch.int)

        x = FromFishAudioRepo.decode_n_tokens(
            model,
            first_token.view(1, codebook_dim, -1),
            input_pos,
            max_new_tokens - 1,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            argmax_decoding=argmax_decoding,
        )
        seq = seq[:, : T + 1 + x.size(1)]
        seq[:, T + 1 :] = x

        # Clean up temporary variables
        del first_token, x, prompt, empty, input_pos

        return seq


def default_fish_audio_audio_decoder_config() -> "FishAudioAudioDecoderConfig_Foreign":
    return FishAudioAudioDecoderConfig_Foreign(dac_config=get_default_fishaudio_dac_config())


def load_fish_audio_audio_decoder(chkpt_path: Path, device: str = "cpu") -> "FishAudioAudioDecoder_Foreign":
    config = default_fish_audio_audio_decoder_config()
    dac_model = instantiate(config.dac_config)
    assert isinstance(dac_model, DAC)
    state_dict = torch.load(chkpt_path / "codec.pth", map_location=device, mmap=True, weights_only=True)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if any("generator" in k for k in state_dict):
        state_dict = {k.replace("generator.", ""): v for k, v in state_dict.items() if "generator." in k}

    dac_model.load_state_dict(state_dict, strict=False, assign=True)
    dac_model.eval()
    dac_model.to(device)

    decoder = FishAudioAudioDecoder_Foreign(config=config, dac_model=dac_model)

    return decoder


@dataclass(frozen=True)
class FishAudioAudioDecoderConfig_Foreign:
    dac_config: DictConfig


class FishAudioAudioDecoder_Foreign(TTSAudioDecoder[FishAudioAudioDecoderConfig_Foreign]):
    dac_model: DAC

    @property
    def samplerate(self) -> int:
        return self.dac_model.sample_rate

    @property
    def activation_precision(self) -> DTypeLike:
        semantic_quantizer = self.dac_model.quantizer
        assert isinstance(semantic_quantizer, ResidualVectorQuantize)
        return DTypeConvert.to_jax(semantic_quantizer.quantizers[0].codebook.weight.dtype)

    def export_weights(self) -> ParameterTree[Array]:
        return {}

    def import_weights(
        self,
        weights: ParameterTree[Array],  # noqa: ARG002
    ) -> Self:
        return self

    def __call__(self, rvq_codes: Int[Array, " codes tokens"]) -> Array:
        device = self.dac_model.device
        indices = jax_to_torch(rvq_codes).to(device).long()
        if len(indices.shape) != 2:
            raise ValueError(f"Unexpected input shape {indices.shape}")
        indices_lens = torch.tensor([indices.shape[1]], device=device, dtype=torch.long)

        # Restore
        audio_samples, _ = self.dac_model.decode(indices, indices_lens)

        audio_samples = torch_to_jax(audio_samples[0, 0])
        return audio_samples

    def audio_from_codes(self, indices: Array) -> Array:
        return self(indices)


@dataclass(frozen=True)
class FishAudioTextDecoderConfig_Foreign:
    fish_config: DualARModelArgs

    @classmethod
    def from_config_file(cls, path_to_config: str | Path) -> "FishAudioTextDecoderConfig_Foreign":
        config: DualARModelArgs | NaiveModelArgs = BaseModelArgs.from_pretrained(str(path_to_config))
        assert isinstance(config, DualARModelArgs)
        return FishAudioTextDecoderConfig_Foreign(fish_config=config)

    @staticmethod
    def _load_weights_to_fish_model(fish_model: DualARTransformer, path_to_model: str | Path) -> None:
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

    @staticmethod
    def _load_fish_model(
        path_to_model: str | Path,
        fish_config: DualARModelArgs,
        device: str = "cpu",
        precision: torch.dtype = torch.bfloat16,
    ) -> "DualARTransformer":
        tokenizer = FishTokenizer.from_pretrained(str(path_to_model))
        fish_model = DualARTransformer(fish_config, tokenizer)
        fish_model = fish_model.to(device=device, dtype=precision)

        FishAudioTextDecoderConfig_Foreign._load_weights_to_fish_model(fish_model, path_to_model)

        return fish_model

    def load_model(
        self, path_to_model: str | Path, device: str = "cpu", precision: torch.dtype = torch.bfloat16
    ) -> "FishAudioTextDecoder_Foreign":
        fish_model = FishAudioTextDecoderConfig_Foreign._load_fish_model(
            path_to_model=path_to_model, fish_config=self.fish_config, device=device, precision=precision
        )
        return FishAudioTextDecoder_Foreign(config=self, fish_model=fish_model)


class FishAudioTextDecoder_Foreign(TTSTextDecoder[FishAudioTextDecoderConfig_Foreign]):
    fish_model: DualARTransformer

    @property
    def activation_precision(self) -> DTypeLike:
        return DTypeConvert.to_jax(self.fish_model.embeddings.weight.dtype)

    def export_weights(self) -> ParameterTree[Array]:
        return {}

    def import_weights(
        self,
        weights: ParameterTree[Array],  # noqa: ARG002
    ) -> Self:
        return self

    def __call__(
        self,
        text_tokens: Int[Array, "batch tokens"],
        input_pos: Int[Array, "batch tokens"] | None = None,
        sampling_params: FishAudioSamplingParams | None = None,
    ) -> Float[Array, "batch_size tokens hidden_size"]:
        text_tokens_torch = jax_to_torch(text_tokens)

        batch_size, n_tokens = text_tokens_torch.shape

        assert isinstance(self.config, FishAudioTextDecoderConfig_Foreign)
        values = torch.zeros((batch_size, self.config.fish_config.num_codebooks + 1, n_tokens), dtype=torch.int)
        values[:, 0] = text_tokens_torch

        self.fish_model.setup_caches(
            max_batch_size=1,
            max_seq_len=self.fish_model.config.max_seq_len,
            dtype=next(self.fish_model.parameters()).dtype,
        )

        if sampling_params is None:
            sampling_params = FishAudioSamplingParams(
                temperature=0.8008, top_p=0.8008, repetition_penalty=1.1016, argmax_decoding=True
            )

        temperature = torch.tensor(sampling_params.temperature, device=values.device, dtype=torch.bfloat16)
        top_p = torch.tensor(sampling_params.top_p, device=values.device, dtype=torch.bfloat16)
        repetition_penalty = torch.tensor(
            sampling_params.repetition_penalty, device=values.device, dtype=torch.bfloat16
        )

        if input_pos is not None:
            input_pos_torch = jax_to_torch(input_pos)
        else:
            input_pos_torch = torch.arange(0, n_tokens, device=values.device, dtype=torch.long)

        new_token_codes = FromFishAudioRepo.decode_one_token_ar_fishaudio(
            self.fish_model,
            values,
            input_pos_torch,
            temperature,
            top_p,
            repetition_penalty,
            None,
            argmax_decoding=sampling_params.argmax_decoding,
        )

        return torch_to_jax(new_token_codes)

    def decode_utterance(
        self,
        text_tokens: Int[Array, "batch tokens"],
        sampling_params: FishAudioSamplingParams | None = None,
        key: PRNGKeyArray | None = None,  # noqa: ARG002
    ) -> Int[Array, "num_codebooks tokens"]:
        text_tokens_torch = jax_to_torch(text_tokens)

        _, n_tokens = text_tokens_torch.shape

        assert isinstance(self.config, FishAudioTextDecoderConfig_Foreign)
        values = torch.zeros((self.config.fish_config.num_codebooks + 1, n_tokens), dtype=torch.int)
        values[0] = text_tokens_torch

        self.fish_model.setup_caches(
            max_batch_size=1,
            max_seq_len=self.fish_model.config.max_seq_len,
            dtype=next(self.fish_model.parameters()).dtype,
        )

        prompt_length = values.size(1)

        temperature_tensor = torch.tensor(sampling_params.temperature, device=values.device, dtype=torch.bfloat16)
        top_p_tensor = torch.tensor(sampling_params.top_p, device=values.device, dtype=torch.bfloat16)
        repetition_penalty_tensor = torch.tensor(
            sampling_params.repetition_penalty, device=values.device, dtype=torch.bfloat16
        )

        y = FromFishAudioRepo.generate(
            model=self.fish_model,
            prompt=values,
            max_new_tokens=0,
            argmax_decoding=sampling_params.argmax_decoding,
            temperature=temperature_tensor,
            top_p=top_p_tensor,
            repetition_penalty=repetition_penalty_tensor,
        )

        codes = y[1:, prompt_length:-1].clone()
        assert (codes >= 0).all(), f"Negative code found"

        return torch_to_jax(codes)
