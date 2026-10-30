import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import count, islice
from pathlib import Path
from typing import Literal, Self

from lalamo.modules.activations import Sigmoid, SiLU
from lalamo.modules.decoder import DecoderConfig
from lalamo.modules.embedding import UntiedEmbeddingConfig
from lalamo.modules.linear import LinearConfig
from lalamo.modules.mlp import DenseMLPConfig, MixtureOfExpertsConfig, SoftmaxRouting
from lalamo.modules.ngram_injection import NGramHashConfig, NGramInjectionConfig
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.residual import GatedResidualConfig
from lalamo.modules.rope import UnscaledRoPEConfig
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.modules.token_mixers.convolutions import SeparableCausalConvConfig
from lalamo.modules.token_mixers.deltanet import DeltaNetConfig
from lalamo.modules.token_mixers.sparse_attention import SparseAttentionConfig, SparseAttentionIndexerConfig
from lalamo.modules.transformer import TransformerConfig
from lalamo.modules.transformer_layer import SublayerConfig, TransformerLayerConfig

from .common import HuggingFaceLMConfig

__all__ = ["HFQwen4ExpConfig", "ngram_head_vocab_sizes", "ngram_multipliers"]

UINT64_MASK = (1 << 64) - 1
SPLITMIX_GAMMA = 0x9E3779B97F4A7C15


def splitmix64(value: int) -> int:
    value = (value + SPLITMIX_GAMMA) & UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & UINT64_MASK
    return (value ^ (value >> 31)) & UINT64_MASK


def ngram_multipliers(token_vocab_size: int, ngram_size: int, ple_layer_index: int, seed: int) -> tuple[int, ...]:
    half_bound = max(1, ((1 << 63) - 1) // token_vocab_size // 2)
    base_seed = seed + 10007 * ple_layer_index
    return tuple(
        2 * (splitmix64((base_seed + SPLITMIX_GAMMA * (index + 1)) & UINT64_MASK) % half_bound) + 1
        for index in range(ngram_size)
    )


def is_prime(value: int) -> bool:
    return value >= 2 and all(value % divisor for divisor in range(2, math.isqrt(value) + 1))


def ngram_head_vocab_sizes(ngram_vocab_size_base: int, num_heads: int, ple_layer_index: int) -> tuple[int, ...]:
    primes = (candidate for candidate in count(ngram_vocab_size_base) if is_prime(candidate))
    return tuple(islice(primes, ple_layer_index * num_heads, (ple_layer_index + 1) * num_heads))


@dataclass(frozen=True)
class Qwen4ExpRopeParameters:
    rope_theta: float
    partial_rotary_factor: float


@dataclass(frozen=True)
class HFQwen4ExpConfig(HuggingFaceLMConfig):
    model_type: Literal["qwen4_exp", "qwen4_exp_text"]
    hidden_act: Literal["silu"]
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    attention_bias: Literal[False]
    max_position_embeddings: int
    rms_norm_eps: float
    tie_word_embeddings: Literal[False]
    vocab_size: int
    eos_token_id: int | list[int]
    rope_parameters: Qwen4ExpRopeParameters

    layer_types: list[Literal["linear_attention", "full_attention", "indexed_attention"]]
    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    output_gate_type: Literal["silu", "sigmoid"]

    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int

    hc_count: int
    hc_lowrank: int

    indexer_n_heads: int
    indexer_kv_heads: Literal[1]
    indexer_head_dim: int
    indexer_budget: int
    indexer_compress_ratio: int

    ple_layer_ids: list[int]
    ple_embed_dim: int
    ple_conv_kernel_size: int
    ngram_size: int
    heads_per_ngram: int
    ngram_vocab_size_base: int
    seed: int = 1234

    def __post_init__(self) -> None:
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(f"Expected {self.num_hidden_layers} layer types, got {len(self.layer_types)}.")
        if self.shared_expert_intermediate_size != self.moe_intermediate_size:
            raise ValueError("The shared expert must have the same hidden dimension as the routed experts.")
        if any(self.layer_types[layer_id - 1] != "linear_attention" for layer_id in self.ple_layer_ids):
            raise ValueError(
                f"N-gram injection is only supported on linear attention layers, got {self.ple_layer_ids}."
            )

    @classmethod
    def from_json(cls, json_path: Path | str) -> Self:
        with open(json_path) as json_file:
            config = json.load(json_file)
        text_config = config.pop("text_config", {})
        return cls._converter.structure({**text_config, **config}, cls)

    def ngram_injection_config(self, ple_layer_index: int, norm_config: NormalizationConfig) -> NGramInjectionConfig:
        eos_token_id, *_ = (self.eos_token_id,) if isinstance(self.eos_token_id, int) else self.eos_token_id
        num_heads = (self.ngram_size - 1) * self.heads_per_ngram
        return NGramInjectionConfig(
            hash_config=NGramHashConfig(
                token_vocab_size=self.vocab_size,
                ngram_size=self.ngram_size,
                heads_per_ngram=self.heads_per_ngram,
                eos_token_id=eos_token_id,
                multipliers=ngram_multipliers(self.vocab_size, self.ngram_size, ple_layer_index, self.seed),
                head_vocab_sizes=ngram_head_vocab_sizes(self.ngram_vocab_size_base, num_heads, ple_layer_index),
            ),
            linear_config=LinearConfig(),
            norm_config=norm_config,
            conv_config=SeparableCausalConvConfig(has_biases=False),
            embedding_dim=self.ple_embed_dim,
            conv_kernel_size=(self.ple_conv_kernel_size - 1) * self.ngram_size + 1,
        )

    def to_decoder_config(
        self,
        metadata_dict: Mapping[str, str],  # noqa: ARG002
    ) -> DecoderConfig:
        linear_config = LinearConfig()
        rmsnorm_config = NormalizationConfig(
            epsilon=self.rms_norm_eps,
            scale_offset=1.0,
            upcast_mode=UpcastMode.FULL_LAYER,
            subtract_mean=False,
        )
        rope_config = UnscaledRoPEConfig(
            base=self.rope_parameters.rope_theta,
            max_sequence_length=self.max_position_embeddings,
            head_dim=int(self.head_dim * self.rope_parameters.partial_rotary_factor),
        )
        delta_net_config = DeltaNetConfig(
            in_proj_config=linear_config,
            conv_config=SeparableCausalConvConfig(has_biases=False),
            out_proj_config=linear_config,
            norm_config=NormalizationConfig(
                epsilon=self.rms_norm_eps,
                scale_offset=None,
                upcast_mode=UpcastMode.ONLY_NORMALIZATION,
                subtract_mean=False,
            ),
            num_heads=self.linear_num_value_heads,
            num_groups=self.linear_num_key_heads,
            head_dim=self.linear_key_head_dim,
            value_head_dim=self.linear_value_head_dim,
            kernel_size=self.linear_conv_kernel_dim,
            output_gate=Sigmoid() if self.output_gate_type == "sigmoid" else SiLU(),
        )
        sparse_attention_config = SparseAttentionConfig(
            attention_config=AttentionConfig(
                qkvg_projection_config=linear_config,
                out_projection_config=linear_config,
                query_norm_config=rmsnorm_config,
                key_norm_config=rmsnorm_config,
                num_heads=self.num_attention_heads,
                num_groups=self.num_key_value_heads,
                head_dim=self.head_dim,
                is_causal=True,
                scale=None,
                sliding_window_size=None,
                logit_soft_cap=None,
                has_sinks=False,
                has_qkvg_biases=False,
                has_out_biases=False,
                has_gate=True,
            ),
            indexer_config=SparseAttentionIndexerConfig(
                projection_config=linear_config,
                query_norm_config=rmsnorm_config,
                key_norm_config=rmsnorm_config,
                num_heads=self.indexer_n_heads,
                head_dim=self.indexer_head_dim,
                compression_ratio=self.indexer_compress_ratio,
                token_budget=self.indexer_budget,
            ),
        )
        mlp_config = MixtureOfExpertsConfig(
            expert_config=DenseMLPConfig(
                linear_config=linear_config,
                activation=SiLU(),
                has_up_biases=False,
                has_down_biases=False,
                up_clipping=None,
                gate_clipping=None,
            ),
            router_config=linear_config,
            routing_function=SoftmaxRouting(),
            num_routed_experts=self.num_experts,
            num_active_routed_experts=self.num_experts_per_tok,
            router_has_biases=False,
            num_shared_experts=1,
            expert_hidden_dim=self.moe_intermediate_size,
            gate_config=linear_config,
        )
        ple_layer_ids = sorted(set(self.ple_layer_ids))
        layer_configs = tuple(
            TransformerLayerConfig(
                prologue_configs=(
                    (self.ngram_injection_config(ple_layer_ids.index(layer_index + 1), rmsnorm_config),)
                    if layer_index + 1 in ple_layer_ids
                    else ()
                ),
                mixer=SublayerConfig(
                    pre_norm_config=None,
                    conv_config=None,
                    body_config=delta_net_config if layer_type == "linear_attention" else sparse_attention_config,
                    post_norm_config=None,
                ),
                mlp=SublayerConfig(
                    pre_norm_config=None,
                    conv_config=None,
                    body_config=mlp_config,
                    post_norm_config=None,
                ),
                rope_config=None if layer_type == "linear_attention" else rope_config,
            )
            for layer_index, layer_type in enumerate(self.layer_types)
        )
        return DecoderConfig(
            embedding_config=UntiedEmbeddingConfig(input_scale=None, logit_soft_cap=None),
            transformer_config=TransformerConfig(
                residual_config=GatedResidualConfig(
                    stream_count=self.hc_count,
                    low_rank_dim=self.hc_lowrank,
                    norm_config=rmsnorm_config,
                    linear_config=linear_config,
                ),
                layer_configs=layer_configs,
                output_norm_config=None,
                model_dim=self.hidden_size,
                hidden_dim=self.moe_intermediate_size,
            ),
            vocab_size=self.vocab_size,
        )
