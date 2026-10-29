from dataclasses import dataclass, replace
from functools import partial
from typing import Literal

from lalamo.model_import.loaders.binding import Binder
from lalamo.model_import.loaders.qwen3_8 import load_gated_residual, load_ngram, load_sparse_attention
from lalamo.modules.activations import Sigmoid, SiLU
from lalamo.modules.decoder import DecoderConfig
from lalamo.modules.embedding import UntiedEmbeddingConfig
from lalamo.modules.flow import EmitConfig, Expand, SequenceConfig, StepConfig, residual
from lalamo.modules.gated_residual import GatedResidualConfig
from lalamo.modules.linear import LinearConfig
from lalamo.modules.mlp import DenseMLPConfig, MixtureOfExpertsConfig, SoftmaxRouting
from lalamo.modules.ngram import NGramConfig
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.residual_connections import routed
from lalamo.modules.rope import UnscaledRoPEConfig
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.modules.token_mixers.convolutions import SeparableCausalConvConfig
from lalamo.modules.token_mixers.deltanet import DeltaNetConfig
from lalamo.modules.token_mixers.qsa import QSAConfig
from lalamo.utils.parameter_path import ParameterPath

from .common import HuggingFaceLMConfig


@dataclass(frozen=True)
class Qwen38RopeParameters:
    rope_type: Literal["default"]
    rope_theta: float
    partial_rotary_factor: float


@dataclass(frozen=True)
class Qwen38TextConfig:
    model_type: Literal["qwen4_exp_text"]
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float
    vocab_size: int
    eos_token_id: int
    rope_parameters: Qwen38RopeParameters
    # Transformers saves the checkpoint's "full_attention" layers as "indexed_attention".
    layer_types: list[Literal["linear_attention", "full_attention", "indexed_attention"]]
    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
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
    make_ngram_vocab_size_divisible_by: int
    split_ngram_parts: int
    hidden_act: Literal["silu"]
    output_gate_type: Literal["sigmoid"]
    tie_word_embeddings: Literal[False]
    attention_bias: Literal[False]
    seed: int = 1234
    norm_topk_prob: Literal[True] = True

    def __post_init__(self) -> None:
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must describe every Qwen3.8 layer.")
        if any(
            layer_id < 1 or layer_id > self.num_hidden_layers or self.layer_types[layer_id - 1] != "linear_attention"
            for layer_id in self.ple_layer_ids
        ):
            raise ValueError("PLE ids must reference one-based linear-attention layers.")


@dataclass(frozen=True)
class HFQwen38Config(HuggingFaceLMConfig):
    model_type: Literal["qwen4_exp"]
    text_config: Qwen38TextConfig

    @property
    def eos_token_ids(self) -> list[int]:
        return [self.text_config.eos_token_id]

    def build_decoder(self, at: Binder) -> DecoderConfig:
        config = self.text_config
        if config.shared_expert_intermediate_size != config.moe_intermediate_size:
            raise ValueError("Qwen3.8-Flash-Next uses the same hidden dimension for routed and shared experts.")
        linear = LinearConfig()
        norm = NormalizationConfig(config.rms_norm_eps, 1.0, UpcastMode.FULL_LAYER, subtract_mean=False)
        connection = GatedResidualConfig(config.hc_count, config.hc_lowrank, norm, linear, linear)
        rope = UnscaledRoPEConfig(
            base=config.rope_parameters.rope_theta,
            max_sequence_length=config.max_position_embeddings,
            head_dim=int(config.head_dim * config.rope_parameters.partial_rotary_factor),
        )
        mlp = MixtureOfExpertsConfig(
            num_routed_experts=config.num_experts,
            num_active_routed_experts=config.num_experts_per_tok,
            routing_function=SoftmaxRouting(),
            router_config=linear,
            router_has_biases=False,
            expert_config=DenseMLPConfig(
                linear_config=linear,
                activation=SiLU(),
                has_up_biases=False,
                has_down_biases=False,
                up_clipping=None,
                gate_clipping=None,
                hidden_dim=config.moe_intermediate_size,
            ),
            gate_config=linear,
            num_shared_experts=1,
        )
        ple_layer_ids = sorted(set(config.ple_layer_ids))
        load_ngram_parts = partial(load_ngram, parts=config.split_ngram_parts)
        steps: list[StepConfig] = [Expand(config.hc_count)]
        taps: list[EmitConfig] = []
        for index, kind in enumerate(config.layer_types):
            path = ParameterPath("model.layers") / index
            if kind == "linear_attention":
                mixer = DeltaNetConfig(
                    in_proj_config=linear,
                    conv_config=SeparableCausalConvConfig(has_biases=False),
                    out_proj_config=linear,
                    norm_config=NormalizationConfig(
                        config.rms_norm_eps, None, UpcastMode.ONLY_NORMALIZATION, subtract_mean=False
                    ),
                    num_heads=config.linear_num_value_heads,
                    num_groups=config.linear_num_key_heads,
                    head_dim=config.linear_key_head_dim,
                    value_head_dim=config.linear_value_head_dim,
                    kernel_size=config.linear_conv_kernel_dim,
                    output_gate=Sigmoid(),
                )
                mixer = at(path / "linear_attn", mixer)
            else:
                mixer = QSAConfig(
                    attention_config=AttentionConfig(
                        qkvg_projection_config=linear,
                        out_projection_config=linear,
                        query_norm_config=norm,
                        key_norm_config=norm,
                        num_heads=config.num_attention_heads,
                        num_groups=config.num_key_value_heads,
                        head_dim=config.head_dim,
                        is_causal=True,
                        scale=None,
                        sliding_window_size=None,
                        logit_soft_cap=None,
                        has_sinks=False,
                        has_qkvg_biases=False,
                        has_out_biases=False,
                        has_gate=True,
                    ),
                    indexer_projection_config=linear,
                    indexer_norm_config=norm,
                    indexer_num_heads=config.indexer_n_heads,
                    indexer_head_dim=config.indexer_head_dim,
                    token_budget=config.indexer_budget,
                    compress_ratio=config.indexer_compress_ratio,
                    rope_config=rope,
                )
                mixer = at(path / "self_attn", mixer, load=load_sparse_attention)
            if index + 1 in ple_layer_ids:
                ngram = NGramConfig(
                    vocab_size=config.vocab_size,
                    seed=config.seed,
                    eos_token_id=config.eos_token_id,
                    ngram_size=config.ngram_size,
                    heads_per_ngram=config.heads_per_ngram,
                    vocab_size_base=config.ngram_vocab_size_base,
                    vocab_size_divisor=config.make_ngram_vocab_size_divisible_by,
                    embedding_dim=config.ple_embed_dim,
                    kernel_size=config.ple_conv_kernel_size,
                    norm_config=norm,
                    layer_index=ple_layer_ids.index(index + 1),
                    linear_config=linear,
                    streams=config.hc_count,
                )
                steps.append(residual(at(path / "ple", ngram, load=load_ngram_parts)))
            tap = EmitConfig(f"hidden_states.{index}")
            steps += [
                routed(at(path / "attn_hyper_connection", connection, load=load_gated_residual), mixer),
                routed(at(path / "mlp_hyper_connection", connection, load=load_gated_residual), at(path / "mlp", mlp)),
                tap,
            ]
            taps.append(tap)
        readout = replace(connection, injection_config=None)
        steps.append(at("model.hyper_connection_mixer", readout, load=load_gated_residual))
        return DecoderConfig(
            at("model.embed_tokens", UntiedEmbeddingConfig(input_scale=None, logit_soft_cap=None)),
            SequenceConfig(tuple(steps)),
            config.hidden_size,
            config.vocab_size,
            tuple(taps),
        )
