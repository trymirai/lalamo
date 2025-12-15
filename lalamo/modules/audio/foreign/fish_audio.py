import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Self

import equinox as eqx
import jax
import torch
from fish_speech.models.text2semantic.llama import (
    BaseModelArgs,
    DualARModelArgs,
    DualARTransformer,
)
from fish_speech.tokenizer import FishTokenizer
from jax import numpy as jnp
from jax import vmap
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray
from tokenizers import Tokenizer
from transformers.integrations.tiktoken import convert_tiktoken_to_fast

from lalamo.common import ParameterPath, ParameterTree
from lalamo.model_import.loaders.fish_audio_loaders import load_fish_audio_transformer
from lalamo.model_import.loaders.huggingface import load_linear, load_tied_embedding
from lalamo.model_import.model_specs.common import cast_if_float
from lalamo.modules import (
    AttentionConfig,
    DenseMLPConfig,
    EmbeddingConfig,
    ForwardPassMode,
    FullPrecisionLinear,
    FullPrecisionLinearConfig,
    Identity,
    LalamoModule,
    NormalizationConfig,
    SiLU,
    State,
    TiedEmbedding,
    TiedEmbeddingConfig,
    Transformer,
    TransformerConfig,
    TransformerLayerConfig,
    UpcastMode,
)
from lalamo.modules.audio.text_decoder import TextDecoderConfig
from lalamo.modules.rope import RoPEConfigBase
from lalamo.modules.torch_interop import jax_to_torch, torch_to_jax
from lalamo.modules.utils import vmap_twice
from lalamo.sampling import CompositePolicy, TemperaturePolicy, TopPPolicy
from lalamo.utils import MapDictValues


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
    logits: Float[Array, " vocabulary"],
    temperature: float,
    top_p: float,
    repetition_penalty: float = 1.0,
    previous_tokens: Optional[Int[Array, " tokens"]] = None,
) -> Float[Array, " vocabulary"]:
    # NOTE: repetition_penalty is not implemented yet - stub for API compatibility
    policies = []
    if top_p > 0 and top_p < 1.0:
        policies.append(TopPPolicy(p=top_p))
    if temperature > 0:
        policies.append(TemperaturePolicy(temperature=max(temperature, 1e-5)))

    if policies:
        policy = CompositePolicy(tuple(policies))
        processed_logits = policy.process_logits(logits)
    else:
        processed_logits = logits

    probs = jax.nn.softmax(processed_logits)
    return probs


def sample(
    logits: Float[Array, "batch tokens vocabulary"],
    key: PRNGKeyArray,
    temperature: float,
    top_p: float,
    repetition_penalty: float = 1.0,
    previous_tokens: Optional[Int[Array, " tokens"]] = None,
) -> tuple[Int[Array, ""], Float[Array, " vocabulary"]]:
    # Take the last token's logits from first batch
    last_logits = logits[0, -1]

    probs = logits_to_probs(
        logits=last_logits,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        previous_tokens=previous_tokens,
    )

    idx_next = jax.random.categorical(key, jnp.log(probs + 1e-10))
    return idx_next, probs


def extract_fast_transformer_params(fish_transformer_config: DualARModelArgs) -> BaseModelArgs:
    return BaseModelArgs(
        model_type=fish_transformer_config.model_type,
        vocab_size=fish_transformer_config.vocab_size,
        n_layer=fish_transformer_config.n_fast_layer,
        n_head=fish_transformer_config.fast_n_head,
        dim=fish_transformer_config.fast_dim,
        intermediate_size=fish_transformer_config.fast_intermediate_size,
        n_local_heads=fish_transformer_config.fast_n_local_heads,
        head_dim=fish_transformer_config.fast_head_dim,
        rope_base=fish_transformer_config.rope_base,
        norm_eps=fish_transformer_config.norm_eps,
        max_seq_len=fish_transformer_config.max_seq_len,
        dropout=fish_transformer_config.dropout,
        tie_word_embeddings=fish_transformer_config.tie_word_embeddings,
        attention_qkv_bias=fish_transformer_config.fast_attention_qkv_bias,
        attention_o_bias=fish_transformer_config.fast_attention_o_bias,
        attention_qk_norm=fish_transformer_config.fast_attention_qk_norm,
        codebook_size=fish_transformer_config.codebook_size,
        num_codebooks=fish_transformer_config.num_codebooks,
    )


def lalamo_transformer_cfg_from_fish(
    config: BaseModelArgs, precision: DTypeLike
) -> tuple[TransformerConfig, FullPrecisionLinearConfig]:
    global_rope_config = RoPEConfigFishAudio(
        precision=precision,
        base=config.rope_base,
        max_sequence_length=config.max_seq_len,
    )
    local_rope_config = None

    norm_config = NormalizationConfig(
        scale_precision=precision,
        accumulation_precision=precision,
        epsilon=config.norm_eps,
        scale_offset=None,
        upcast_mode=UpcastMode.ONLY_NORMALIZATION,
        subtract_mean=False,
    )

    qkv_projection_config = FullPrecisionLinearConfig(precision=precision)
    out_projection_config = FullPrecisionLinearConfig(precision=precision)
    mixer_config = AttentionConfig(
        qkv_projection_config=qkv_projection_config,
        out_projection_config=out_projection_config,
        query_norm_config=norm_config if config.attention_qk_norm else None,
        key_norm_config=norm_config if config.attention_qk_norm else None,
        num_heads=config.n_head,
        num_groups=config.n_local_heads,
        head_dim=config.head_dim,
        is_causal=True,
        scale=None,
        sliding_window_size=None,
        logit_soft_cap=None,
        has_sinks=False,
        has_qkv_biases=False,
        has_out_biases=False,
    )

    mlp_linear_config = FullPrecisionLinearConfig(precision=precision)
    mlp_use_up_biases = False
    mlp_use_down_biases = False
    mlp_config = DenseMLPConfig(
        linear_config=mlp_linear_config,
        activation=SiLU(),
        has_up_biases=mlp_use_up_biases,
        has_down_biases=mlp_use_down_biases,
        gate_clipping=None,
        up_clipping=None,
    )

    pre_mixer_norm_config = norm_config
    post_mixer_norm_config = None
    pre_mlp_norm_config = norm_config
    post_mlp_norm_config = None

    layer_config = TransformerLayerConfig(
        pre_mixer_norm_config=pre_mixer_norm_config,
        mixer_config=mixer_config,
        post_mixer_norm_config=post_mixer_norm_config,
        pre_mlp_norm_config=pre_mlp_norm_config,
        mlp_config=mlp_config,
        post_mlp_norm_config=post_mlp_norm_config,
    )
    model_dim = config.dim
    hidden_dim = config.intermediate_size
    context_length = config.max_seq_len

    transformer_cfg = TransformerConfig(
        global_rope_config=global_rope_config,
        local_rope_config=local_rope_config,
        layer_configs=tuple([layer_config] * config.n_layer),
        output_norm_config=norm_config,
        model_dim=model_dim,
        hidden_dim=hidden_dim,
        context_length=context_length,
    )
    linear_out_cfg = FullPrecisionLinearConfig(precision=precision)

    return (transformer_cfg, linear_out_cfg)


@dataclass(frozen=True)
class RoPEConfigFishAudio(RoPEConfigBase):
    @property
    def _attention_scaling_factor(self) -> float:
        return super()._attention_scaling_factor

    def _precompute_freqs_cis_orig(
        self, head_dim: int, seq_len: int
    ) -> tuple[Float[Array, "sequence head_dim"], Float[Array, "sequence head_dim"]]:
        time_steps = jnp.arange(0, head_dim // 2).astype(jnp.bfloat16) * 2 / head_dim
        freqs = 1.0 / (self.base**time_steps)
        t = jnp.arange(seq_len, device=freqs.device)
        freqs = jnp.outer(t, freqs)
        return (jnp.cos(freqs), jnp.sin(freqs))

    def init_orig(
        self,
        head_dim: int,
        num_timesteps: int,
    ) -> "RoPEFishAudio":
        cosines_cis, sines_cis = self._precompute_freqs_cis(head_dim, num_timesteps)
        cosines = jnp.zeros((num_timesteps, head_dim), self.precision)
        sines = jnp.zeros((num_timesteps, head_dim), self.precision)
        for k in range(num_timesteps):
            cosines = cosines.at[k, 0::2].set(cosines_cis[k])
            cosines = cosines.at[k, 1::2].set(cosines_cis[k])
            sines = sines.at[k, 0::2].set(sines_cis[k])
            sines = sines.at[k, 1::2].set(sines_cis[k])

        return RoPEFishAudio(config=self, cosines=cosines, sines=sines)

    def _precompute_freqs_cis(
        self, head_dim: int, seq_len: int
    ) -> tuple[Float[Array, "sequence head_dim"], Float[Array, "sequence head_dim"]]:
        # time_steps = jnp.arange(0, head_dim, 2).astype(jnp.bfloat16)[: (head_dim // 2)] / head_dim
        time_steps = jnp.repeat(jnp.arange(0, head_dim // 2).astype(jnp.bfloat16) * 2 / head_dim, 2)
        freqs = 1.0 / (self.base**time_steps)
        t = jnp.arange(seq_len, device=freqs.device)
        freqs = jnp.outer(t, freqs)
        return (jnp.cos(freqs), jnp.sin(freqs))

    def init(
        self,
        head_dim: int,
        num_timesteps: int,
    ) -> "RoPEFishAudio":
        cosines_cis, sines_cis = self._precompute_freqs_cis(head_dim, num_timesteps)
        return RoPEFishAudio(config=self, cosines=cosines_cis, sines=sines_cis)


@dataclass
class RoPEFishAudio:
    config: RoPEConfigBase
    sines: Float[Array, "tokens head_channels"]
    cosines: Float[Array, "tokens head_channels"]

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def head_dim(self) -> int:
        _, result = self.sines.shape
        return result

    @property
    def max_sequence_length(self) -> int:
        result, _ = self.sines.shape
        return result

    # @eqx.filter_jit
    def __call__(self, timesteps: Int[Array, " tokens"]) -> "PositionalEmbeddingsFishAudio":
        return PositionalEmbeddingsFishAudio(
            cosines=self.cosines[timesteps],
            sines=self.sines[timesteps],
        )


class PositionalEmbeddingsFishAudio(eqx.Module):
    cosines: Float[Array, "*batch tokens head_channels"]
    sines: Float[Array, "*batch tokens head_channels"]

    @property
    def head_dim(self) -> int:
        return self.cosines.shape[-1]

    def interleave_for_cis_rope(
        self,
        heads: Float[Array, "*batch tokens head_channels"],
    ) -> Float[Array, "*batch tokens head_channels"]:
        interleaved = jnp.zeros(heads.shape, dtype=heads.dtype)
        interleaved = interleaved.at[..., 0::2].set(-heads[..., 1::2])
        interleaved = interleaved.at[..., 1::2].set(heads[..., 0::2])
        return interleaved

    def apply(self, heads: Float[Array, "*batch tokens head_channels"]) -> Float[Array, "*batch tokens head_channels"]:
        return heads * self.cosines + self.interleave_for_cis_rope(heads) * self.sines


@dataclass
class FishAudioTextDecoderResult:
    token_codes: Float[Array, "batch codes"]
    hidden_states: Array | None
    state: State | None


@dataclass(frozen=True)
class FishAudioTextDecoderConfig(TextDecoderConfig):
    slow_embeddings_config: TiedEmbeddingConfig
    slow_model_config: TransformerConfig
    slow_readout_config: FullPrecisionLinearConfig

    fast_embeddings_config: TiedEmbeddingConfig
    fast_model_config: TransformerConfig
    fast_readout_config: FullPrecisionLinearConfig

    codebook_embeddings_config: TiedEmbeddingConfig
    fast_model_projection_config: FullPrecisionLinearConfig | None

    semantic_token_begin_id: int
    semantic_token_end_id: int
    codebook_size: int
    vocab_size: int
    slow_model_dim: int
    fast_model_dim: int
    num_codebooks: int

    scale_codebook_embeddings: bool

    precision: DTypeLike

    # NOTE: magic constants from FishAudio code
    short_logits_size: int = 1024
    repeat_window_size: int = 16

    @classmethod
    def from_fish_audio_config(
        cls,
        fish_audio_cfg: DualARModelArgs,
        tokenizer: FishTokenizer,
        precision: DTypeLike,
    ) -> "FishAudioTextDecoderConfig":
        slow_transformer_cfg, slow_readout_cfg = lalamo_transformer_cfg_from_fish(fish_audio_cfg, precision)
        fast_fish_cfg = extract_fast_transformer_params(fish_audio_cfg)
        fast_transformer_cfg, fast_readout_cfg = lalamo_transformer_cfg_from_fish(fast_fish_cfg, precision)

        slow_embedding_cfg = TiedEmbeddingConfig(input_scale=None, logit_soft_cap=None, precision=precision)
        fast_embedding_cfg = TiedEmbeddingConfig(input_scale=None, logit_soft_cap=None, precision=precision)

        codebook_embeddings_cfg = TiedEmbeddingConfig(input_scale=None, logit_soft_cap=None, precision=precision)
        if fish_audio_cfg.dim == fish_audio_cfg.fast_dim:
            fast_model_projection_config = None
        else:
            fast_model_projection_config = FullPrecisionLinearConfig(precision)

        assert fish_audio_cfg.fast_dim is not None
        return FishAudioTextDecoderConfig(
            slow_embeddings_config=slow_embedding_cfg,
            slow_model_config=slow_transformer_cfg,
            slow_readout_config=slow_readout_cfg,
            fast_embeddings_config=fast_embedding_cfg,
            fast_model_config=fast_transformer_cfg,
            fast_readout_config=fast_readout_cfg,
            codebook_embeddings_config=codebook_embeddings_cfg,
            fast_model_projection_config=fast_model_projection_config,
            semantic_token_begin_id=tokenizer.semantic_begin_id,
            semantic_token_end_id=tokenizer.semantic_end_id,
            codebook_size=fish_audio_cfg.codebook_size,
            precision=precision,
            vocab_size=fish_audio_cfg.vocab_size,
            slow_model_dim=fish_audio_cfg.dim,
            fast_model_dim=fish_audio_cfg.fast_dim,
            num_codebooks=fish_audio_cfg.num_codebooks,
            scale_codebook_embeddings=fish_audio_cfg.scale_codebook_embeddings,
        )

    def empty(self) -> "FishAudioTextDecoder":
        embeddings_slow = self.slow_embeddings_config.empty(self.vocab_size, self.slow_model_dim)
        embeddings_fast = self.fast_embeddings_config.empty(self.codebook_size, self.fast_model_dim)
        codebook_embeddings = self.codebook_embeddings_config.empty(
            self.codebook_size * self.num_codebooks, self.slow_model_dim
        )
        if self.fast_model_projection_config is not None:
            fast_model_projection = self.fast_model_projection_config.empty(
                self.slow_model_dim, (self.fast_model_dim,), False
            )
        else:
            fast_model_projection = Identity()
        assert isinstance(embeddings_slow, TiedEmbedding)
        assert isinstance(embeddings_fast, TiedEmbedding)
        assert isinstance(codebook_embeddings, TiedEmbedding)
        return FishAudioTextDecoder(
            self,
            embeddings_slow=embeddings_slow,
            transformer_slow=self.slow_model_config.empty(),
            readout_slow=self.slow_readout_config.empty(
                input_dim=self.slow_model_dim,
                output_dims=(self.vocab_size,),
                has_biases=False,
            ),
            embeddings_fast=embeddings_fast,
            transformer_fast=self.fast_model_config.empty(),
            readout_fast=self.fast_readout_config.empty(
                input_dim=self.fast_model_dim, output_dims=(self.codebook_size,), has_biases=False
            ),
            codebook_embeddings=codebook_embeddings,
            fast_model_projection=fast_model_projection,
        )

    @classmethod
    def load_model(cls, fish_model_or_path: Path | DualARTransformer, precision: DTypeLike) -> "FishAudioTextDecoder":
        if isinstance(fish_model_or_path, Path):
            fish_tokenizer = FishTokenizer(str(fish_model_or_path / "tokenizer.tiktoken"))
            with open(file=fish_model_or_path / "config.json") as config_file:
                fish_cfg_json = json.load(config_file)
            fish_model_cfg = DualARModelArgs(**fish_cfg_json)
            fish_model_dict = torch.load(fish_model_or_path / "model.pth", weights_only=True)
        else:
            fish_tokenizer = fish_model_or_path.tokenizer
            fish_model_cfg = fish_model_or_path.config
            fish_model_dict = fish_model_or_path.state_dict()

        assert isinstance(fish_model_cfg, DualARModelArgs)
        config = cls.from_fish_audio_config(fish_model_cfg, fish_tokenizer, precision)
        weights_mapping = dict(MapDictValues(lambda v: cast_if_float(torch_to_jax(v), precision), fish_model_dict))

        transformer_slow, readout_slow = load_fish_audio_transformer(
            config.slow_model_config.empty(),
            config.slow_readout_config.empty(
                input_dim=config.slow_model_dim,
                output_dims=(config.vocab_size,),
                has_biases=False,
            ),
            weights_mapping,
            fast=False,
        )
        transformer_fast, readout_fast = load_fish_audio_transformer(
            config.fast_model_config.empty(),
            config.fast_readout_config.empty(
                input_dim=config.fast_model_dim, output_dims=(config.codebook_size,), has_biases=False
            ),
            weights_mapping,
            fast=True,
        )
        embeddings_slow = load_tied_embedding(
            config.slow_embeddings_config.empty(config.vocab_size, config.slow_model_dim),
            weights_mapping,
            ParameterPath("embeddings"),
        )
        embeddings_fast = load_tied_embedding(
            config.fast_embeddings_config.empty(config.codebook_size, config.fast_model_dim),
            weights_mapping,
            ParameterPath("fast_embeddings"),
        )

        codebook_embeddings = load_tied_embedding(
            config.codebook_embeddings_config.empty(
                config.codebook_size * config.num_codebooks, config.slow_model_dim
            ),
            weights_mapping,
            ParameterPath("codebook_embeddings"),
        )

        if config.fast_model_projection_config is not None:
            fast_model_projection = load_linear(
                config.fast_model_projection_config.empty(
                    input_dim=config.fast_model_dim, output_dims=(config.codebook_size,), has_biases=False
                ),
                weights_mapping,
                ParameterPath("fast_project_in"),
            )
            assert isinstance(fast_model_projection, FullPrecisionLinear)
        else:
            fast_model_projection = Identity()

        return FishAudioTextDecoder(
            config=config,
            embeddings_slow=embeddings_slow,
            transformer_slow=transformer_slow,
            readout_slow=readout_slow,
            embeddings_fast=embeddings_fast,
            transformer_fast=transformer_fast,
            readout_fast=readout_fast,
            codebook_embeddings=codebook_embeddings,
            fast_model_projection=fast_model_projection,
        )


class FishAudioTextDecoder(LalamoModule[FishAudioTextDecoderConfig]):
    embeddings_slow: TiedEmbedding
    transformer_slow: Transformer
    readout_slow: FullPrecisionLinear

    embeddings_fast: TiedEmbedding
    transformer_fast: Transformer
    readout_fast: FullPrecisionLinear

    codebook_embeddings: TiedEmbedding
    fast_model_projection: FullPrecisionLinear | Identity

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def export_weights(self) -> ParameterTree[Array]:
        # TODO(peter.glushkov): implement me
        return {}

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        # TODO(peter.glushkov): implement me
        return self

    @property
    def semantic_begin_id(self) -> int:
        return self.config.semantic_token_begin_id

    @property
    def semantic_end_id(self) -> int:
        return self.config.semantic_token_end_id

    @property
    def num_codebooks(self) -> int:
        return self.config.num_codebooks

    def embed_slow_model(self) -> Array:
        return jnp.zeros((1, 2, 3))

    def __call__(
        self,
        text_tokens: Int[Array, "batch tokens"],
        input_pos: Int[Array, "batch tokens"] | None = None,
        state: State | None = None,
        argmax_decoding: bool = False,
        key: PRNGKeyArray | None = None,
    ) -> FishAudioTextDecoderResult:
        batch_size, seq_length = text_tokens.shape
        if input_pos is None:
            input_pos = jnp.arange(seq_length)[None, :]

        text_and_codebooks = jnp.zeros(
            (batch_size, self.config.num_codebooks + 1, seq_length), dtype=text_tokens.dtype
        )
        # NOTE: the rest of codebook lines should be filled in case audio promt is used, but
        # ignore it for now
        text_and_codebooks = text_and_codebooks.at[:, 0, :].set(text_tokens)

        embeddings = self.embed(text_and_codebooks)
        codes, updated_state = decode_next_token(
            model=self,
            x=embeddings,
            state_slow=state,
            input_pos=input_pos,
            temperature=0,
            top_p=0,
            repetition_penalty=0,
            previous_tokens=None,
            argmax_decoding=argmax_decoding,
            key=key,
        )
        return FishAudioTextDecoderResult(token_codes=codes, hidden_states=None, state=updated_state)

    def embed(
        self, inp: Int[Array, "batch codebooks tokens"], apply_codebook_embeddings: bool = False
    ) -> Float[Array, "batch tokens embedding"]:
        """
        apply_codebook_embeddings argumet should be set to 'True' if audio-prompt is used. In this
        case we expect codebook lines [1:-1] to be filled with something meaningful
        """

        vq_masks = (inp[:, 0] >= self.semantic_begin_id) & (inp[:, 0] <= self.semantic_end_id)
        embeddings = self.embeddings_slow.embed(inp[:, 0])

        if apply_codebook_embeddings or jnp.any(vq_masks):
            _, _, seq_length = inp.shape
            codebook_offsets = (jnp.arange(self.config.num_codebooks) * self.config.codebook_size).reshape(-1, 1)
            codebook_offsets = jnp.tile(codebook_offsets, (1, seq_length))
            codebook_embeds = vmap(self.codebook_embeddings.embed)(inp[:, 1:, :] + codebook_offsets)

            vq_embeds_sum = codebook_embeds.sum(axis=1)
            vq_embeds_sum = vq_embeds_sum.at[~vq_masks].set(0)
            embeddings = embeddings + vq_embeds_sum

        if self.config.scale_codebook_embeddings:
            # Expand vq_masks to match x's shape
            vq_masks_expanded = jnp.expand_dims(vq_masks, axis=-1)
            vq_masks_expanded = jnp.broadcast_to(vq_masks_expanded, embeddings.shape)
            embeddings = jnp.where(vq_masks_expanded, embeddings / jnp.sqrt(self.config.num_codebooks + 1), embeddings)
            assert isinstance(embeddings, Array)

        return embeddings


@torch.no_grad
def decode_next_token(
    model: FishAudioTextDecoder,
    x: Array,
    state_slow: State | None,
    input_pos: Array,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    previous_tokens: Array | None = None,
    argmax_decoding: bool = False,
    key: PRNGKeyArray | None = None,
) -> tuple[Int[Array, "batch codes"], State | None]:
    assert argmax_decoding or key is not None

    slow_forward_result = model.transformer_slow(
        inner_features=x,
        token_positions=input_pos,
        state=state_slow,
        return_updated_state=True,
        return_layer_results=True,
        return_positional_embeddings=False,
        lengths_without_padding=None,
        forward_pass_mode=ForwardPassMode.MULTI_TOKEN,
        forward_pass_config=None,
    )
    assert slow_forward_result.layer_results is not None
    hidden_states = slow_forward_result.layer_results[-1].outputs[:, -1:]
    (hidden_states,) = vmap(model.fast_model_projection)(hidden_states)
    hidden_states = hidden_states.reshape(hidden_states.shape[0], 1, -1)

    (logits,) = vmap_twice(model.readout_slow)(slow_forward_result.outputs)

    if argmax_decoding:
        codebooks = [logits[:, -1].argmax(axis=-1)]
    else:
        codebooks = [
            sample(
                logits,
                key,
                temperature,
                top_p,
                repetition_penalty,
                previous_tokens=(previous_tokens[:, 0] if previous_tokens is not None else None),
            )[0].reshape(1)
        ]

    batch_size, *_ = x.shape
    input_pos_fast = jnp.zeros((batch_size, 1), dtype=jnp.int32)
    fast_first_result = model.transformer_fast(
        inner_features=hidden_states,
        token_positions=input_pos_fast,
        state=None,
        return_updated_state=True,
        return_layer_results=False,
        return_positional_embeddings=False,
        lengths_without_padding=None,
        forward_pass_mode=ForwardPassMode.SINGLE_TOKEN,
        forward_pass_config=None,
    )
    state_fast = fast_first_result.updated_state
    a = codebooks[0] - model.semantic_begin_id
    a = a.at[a < 0].set(0)
    hidden_states = model.embeddings_fast.embed(a)
    codebooks.append(a)

    for codebook_idx in range(1, model.num_codebooks):
        hidden_states = hidden_states.reshape(hidden_states.shape[0], 1, -1)
        input_pos_fast = jnp.array([codebook_idx])[None, :]
        fast_result = model.transformer_fast(
            inner_features=hidden_states,
            token_positions=input_pos_fast,
            state=state_fast,
            return_updated_state=True,
            return_layer_results=False,
            return_positional_embeddings=False,
            lengths_without_padding=None,
            forward_pass_mode=ForwardPassMode.SINGLE_TOKEN,
            forward_pass_config=None,
        )
        (fast_logits,) = vmap_twice(model.readout_fast)(fast_result.outputs)
        state_fast = fast_result.updated_state

        short_logits = fast_logits[:, :, : model.config.short_logits_size]

        if argmax_decoding:
            a = short_logits[:, -1].argmax(axis=-1)
        else:
            a = sample(
                short_logits,
                key,
                temperature,
                top_p,
                repetition_penalty,
                previous_tokens=(previous_tokens[codebook_idx + 1] if previous_tokens is not None else None),
            )[0].reshape(1)

        hidden_states = model.embeddings_fast.embed(a)
        codebooks.append(a)

    codebooks = jnp.stack(codebooks, axis=1)

    return codebooks, slow_forward_result.updated_state
