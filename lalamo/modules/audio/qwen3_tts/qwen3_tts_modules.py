import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Self

import jax
import jax.numpy as jnp
from einops import rearrange
from jax import vmap
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree, require_array, require_tree
from lalamo.modules.audio.common_modules import (
    CausalConv1d,
    CausalConv1dConfig,
)
from lalamo.modules.common import LalamoModule
from lalamo.modules.linear import FullPrecisionLinear, FullPrecisionLinearConfig
from lalamo.modules.normalization import Normalization, NormalizationConfig

__all__ = [
    "Qwen3TTSCausalTransposeConv1d",
    "Qwen3TTSCausalTransposeConv1dConfig",
    "Qwen3TTSConvNeXtBlock",
    "Qwen3TTSConvNeXtBlockConfig",
    "Qwen3TTSDecoderBlock",
    "Qwen3TTSDecoderBlockConfig",
    "Qwen3TTSEuclideanCodebook",
    "Qwen3TTSEuclideanCodebookConfig",
    "Qwen3TTSResidualUnit",
    "Qwen3TTSResidualUnitConfig",
    "Qwen3TTSResidualVectorQuantization",
    "Qwen3TTSResidualVectorQuantizationConfig",
    "Qwen3TTSResidualVectorQuantizer",
    "Qwen3TTSResidualVectorQuantizerConfig",
    "Qwen3TTSSnakeBeta",
    "Qwen3TTSSnakeBetaConfig",
    "Qwen3TTSSplitResidualVectorQuantizer",
    "Qwen3TTSSplitResidualVectorQuantizerConfig",
    "Qwen3TTSVectorQuantization",
    "Qwen3TTSVectorQuantizationConfig",
    "apply_rotary_pos_emb",
    "rotate_half",
]


def _debug_tensor(name: str, value: Array, *, enabled: bool) -> None:
    if not enabled:
        return
    jax.debug.print(
        "[qwen3_tts] {name}: shape={shape} min={min_val:.5f} max={max_val:.5f}",
        name=name,
        shape=value.shape,
        min_val=jnp.min(value),
        max_val=jnp.max(value),
    )


def rotate_half(x: Float[Array, "*batch channels"]) -> Float[Array, "*batch channels"]:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(
    q: Float[Array, "batch heads tokens channels"],
    k: Float[Array, "batch heads tokens channels"],
    cos: Float[Array, "batch tokens channels"],
    sin: Float[Array, "batch tokens channels"],
    unsqueeze_dim: int = 1,
) -> tuple[Float[Array, "batch heads tokens channels"], Float[Array, "batch heads tokens channels"]]:
    cos = jnp.expand_dims(cos, axis=unsqueeze_dim)
    sin = jnp.expand_dims(sin, axis=unsqueeze_dim)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


@dataclass(frozen=True)
class Qwen3TTSCausalTransposeConv1dConfig:
    precision: DTypeLike
    has_biases: bool

    def empty(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
    ) -> "Qwen3TTSCausalTransposeConv1d":
        in_per_group = in_channels // groups
        weights = jnp.zeros((out_channels, in_per_group, kernel_size), dtype=self.precision)
        if self.has_biases:
            biases = jnp.zeros((out_channels,), dtype=self.precision)
        else:
            biases = None
        pad = kernel_size - stride
        left_pad = math.ceil(pad)
        right_pad = left_pad
        return Qwen3TTSCausalTransposeConv1d(
            config=self,
            weights=weights,
            biases=biases,
            in_channels=in_channels,
            stride=stride,
            groups=groups,
            left_pad=left_pad,
            right_pad=right_pad,
        )

    def random_init(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        *,
        key: PRNGKeyArray,
    ) -> "Qwen3TTSCausalTransposeConv1d":
        in_per_group = in_channels // groups
        weights = jax.random.normal(key, (out_channels, in_per_group, kernel_size), dtype=self.precision)
        if self.has_biases:
            biases = jnp.zeros((out_channels,), dtype=self.precision)
        else:
            biases = None
        pad = kernel_size - stride
        left_pad = math.ceil(pad)
        right_pad = left_pad
        return Qwen3TTSCausalTransposeConv1d(
            config=self,
            weights=weights,
            biases=biases,
            in_channels=in_channels,
            stride=stride,
            groups=groups,
            left_pad=left_pad,
            right_pad=right_pad,
        )


class Qwen3TTSCausalTransposeConv1d(LalamoModule[Qwen3TTSCausalTransposeConv1dConfig]):
    weights: Float[Array, "out_channels in_channels_per_group kernel_size"]
    biases: Float[Array, " out_channels"] | None

    in_channels: int
    stride: int
    groups: int
    left_pad: int
    right_pad: int

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def out_channels(self) -> int:
        out_channels, _, _ = self.weights.shape
        return int(out_channels)

    @property
    def kernel_size(self) -> int:
        return int(self.weights.shape[-1])

    def __call__(
        self,
        x: Float[Array, "batch tokens channels"],
    ) -> Float[Array, "batch tokens channels"]:
        padding = ((self.kernel_size - 1, self.kernel_size - 1),)
        output = jax.lax.conv_general_dilated(
            x,
            self.weights,
            window_strides=(1,),
            padding=padding,
            lhs_dilation=(self.stride,),
            rhs_dilation=(1,),
            dimension_numbers=("NHC", "OIH", "NHC"),
            feature_group_count=self.groups,
        )
        if self.biases is not None:
            output = output + self.biases[None, None, :]

        _, output_tokens, _ = output.shape
        end_index = output_tokens - self.right_pad if self.right_pad > 0 else output_tokens
        return output[:, self.left_pad : end_index, :]

    def export_weights(self) -> ParameterTree[Array]:
        result: dict[str, Array] = {"weights": self.weights}
        if self.biases is not None:
            result["biases"] = self.biases
        return result

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        if self.biases is not None:
            biases = require_array(weights["biases"])
        else:
            biases = None
        return replace(
            self,
            weights=require_array(weights["weights"]),
            biases=biases,
        )


@dataclass(frozen=True)
class Qwen3TTSSnakeBetaConfig:
    precision: DTypeLike
    alpha_init: float = 1.0
    no_div_by_zero: float = 1e-9
    enable_debug: bool = True

    def empty(self, channels: int) -> "Qwen3TTSSnakeBeta":
        alpha = jnp.zeros((channels,), dtype=self.precision) * self.alpha_init
        beta = jnp.zeros((channels,), dtype=self.precision) * self.alpha_init
        return Qwen3TTSSnakeBeta(config=self, alpha=alpha, beta=beta)

    def random_init(
        self,
        channels: int,
        *,
        key: PRNGKeyArray,  # noqa: ARG002
    ) -> "Qwen3TTSSnakeBeta":
        return self.empty(channels)


class Qwen3TTSSnakeBeta(LalamoModule[Qwen3TTSSnakeBetaConfig]):
    alpha: Float[Array, " channels"]
    beta: Float[Array, " channels"]

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def __call__(
        self,
        x: Float[Array, "batch tokens channels"],
    ) -> Float[Array, "batch tokens channels"]:
        _debug_tensor("snake_beta.input", x, enabled=self.config.enable_debug)
        alpha = jnp.exp(self.alpha)[None, None, :]
        beta = jnp.exp(self.beta)[None, None, :]
        result = x + (jnp.reciprocal(beta + self.config.no_div_by_zero) * jnp.square(jnp.sin(x * alpha)))
        _debug_tensor("snake_beta.output", result, enabled=self.config.enable_debug)
        return result

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            alpha=require_array(weights["alpha"]),
            beta=require_array(weights["beta"]),
        )


@dataclass(frozen=True)
class Qwen3TTSConvNeXtBlockConfig:
    precision: DTypeLike
    conv_config: CausalConv1dConfig
    norm_config: NormalizationConfig
    linear_config: FullPrecisionLinearConfig
    gamma_init: float = 1e-6
    enable_debug: bool = True

    def empty(self, dim: int) -> "Qwen3TTSConvNeXtBlock":
        depthwise_conv = self.conv_config.empty(
            in_channels=dim,
            out_channels=dim,
            kernel_size=7,
            dilation=1,
            stride=1,
            groups=dim,
        )
        norm = self.norm_config.empty(dim)
        pointwise_conv_step1 = self.linear_config.empty(dim, (4 * dim,), has_biases=True)
        pointwise_conv_step2 = self.linear_config.empty(4 * dim, (dim,), has_biases=True)
        gamma = jnp.ones((dim,), dtype=self.precision) * self.gamma_init
        return Qwen3TTSConvNeXtBlock(
            config=self,
            depthwise_conv=depthwise_conv,
            norm=norm,
            pointwise_conv_step1=pointwise_conv_step1,
            pointwise_conv_step2=pointwise_conv_step2,
            gamma=gamma,
        )

    def random_init(self, dim: int, *, key: PRNGKeyArray) -> "Qwen3TTSConvNeXtBlock":
        key_dw, key_pw1, key_pw2 = jax.random.split(key, 3)
        depthwise_conv = self.conv_config.random_init(
            in_channels=dim,
            out_channels=dim,
            kernel_size=7,
            dilation=1,
            stride=1,
            groups=dim,
            key=key_dw,
        )
        norm = self.norm_config.init(dim)
        pointwise_conv_step1 = self.linear_config.random_init(dim, (4 * dim,), has_biases=True, key=key_pw1)
        pointwise_conv_step2 = self.linear_config.random_init(4 * dim, (dim,), has_biases=True, key=key_pw2)
        gamma = jnp.ones((dim,), dtype=self.precision) * self.gamma_init
        return Qwen3TTSConvNeXtBlock(
            config=self,
            depthwise_conv=depthwise_conv,
            norm=norm,
            pointwise_conv_step1=pointwise_conv_step1,
            pointwise_conv_step2=pointwise_conv_step2,
            gamma=gamma,
        )


class Qwen3TTSConvNeXtBlock(LalamoModule[Qwen3TTSConvNeXtBlockConfig]):
    depthwise_conv: CausalConv1d
    norm: Normalization
    pointwise_conv_step1: FullPrecisionLinear
    pointwise_conv_step2: FullPrecisionLinear
    gamma: Float[Array, " channels"]

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def __call__(
        self,
        x: Float[Array, "batch tokens channels"],
    ) -> Float[Array, "batch tokens channels"]:
        _debug_tensor("convnext.input", x, enabled=self.config.enable_debug)
        residual = x
        x = self.depthwise_conv(x)
        x = vmap(vmap(self.norm))(x)
        (x,) = vmap(vmap(self.pointwise_conv_step1))(x)
        x = jax.nn.gelu(x, approximate=False)
        (x,) = vmap(vmap(self.pointwise_conv_step2))(x)
        x = x * self.gamma[None, None, :]
        result = residual + x
        _debug_tensor("convnext.output", result, enabled=self.config.enable_debug)
        return result

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "depthwise_conv": self.depthwise_conv.export_weights(),
            "norm": self.norm.export_weights(),
            "pointwise_conv_step1": self.pointwise_conv_step1.export_weights(),
            "pointwise_conv_step2": self.pointwise_conv_step2.export_weights(),
            "gamma": self.gamma,
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            depthwise_conv=self.depthwise_conv.import_weights(require_tree(weights["depthwise_conv"])),
            norm=self.norm.import_weights(require_tree(weights["norm"])),
            pointwise_conv_step1=self.pointwise_conv_step1.import_weights(
                require_tree(weights["pointwise_conv_step1"])
            ),
            pointwise_conv_step2=self.pointwise_conv_step2.import_weights(
                require_tree(weights["pointwise_conv_step2"])
            ),
            gamma=require_array(weights["gamma"]),
        )


@dataclass(frozen=True)
class Qwen3TTSResidualUnitConfig:
    precision: DTypeLike
    snake_config: Qwen3TTSSnakeBetaConfig
    conv_config: CausalConv1dConfig
    enable_debug: bool = True

    def empty(self, dim: int, dilation: int) -> "Qwen3TTSResidualUnit":
        return Qwen3TTSResidualUnit(
            config=self,
            act1=self.snake_config.empty(dim),
            conv1=self.conv_config.empty(
                in_channels=dim,
                out_channels=dim,
                kernel_size=7,
                dilation=dilation,
                stride=1,
                groups=1,
            ),
            act2=self.snake_config.empty(dim),
            conv2=self.conv_config.empty(
                in_channels=dim,
                out_channels=dim,
                kernel_size=1,
                dilation=1,
                stride=1,
                groups=1,
            ),
        )

    def random_init(self, dim: int, dilation: int, *, key: PRNGKeyArray) -> "Qwen3TTSResidualUnit":
        key_conv1, key_conv2 = jax.random.split(key, 2)
        return Qwen3TTSResidualUnit(
            config=self,
            act1=self.snake_config.random_init(dim, key=key),
            conv1=self.conv_config.random_init(
                in_channels=dim,
                out_channels=dim,
                kernel_size=7,
                dilation=dilation,
                stride=1,
                groups=1,
                key=key_conv1,
            ),
            act2=self.snake_config.random_init(dim, key=key),
            conv2=self.conv_config.random_init(
                in_channels=dim,
                out_channels=dim,
                kernel_size=1,
                dilation=1,
                stride=1,
                groups=1,
                key=key_conv2,
            ),
        )


class Qwen3TTSResidualUnit(LalamoModule[Qwen3TTSResidualUnitConfig]):
    act1: Qwen3TTSSnakeBeta
    conv1: CausalConv1d
    act2: Qwen3TTSSnakeBeta
    conv2: CausalConv1d

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def __call__(
        self,
        x: Float[Array, "batch tokens channels"],
    ) -> Float[Array, "batch tokens channels"]:
        _debug_tensor("residual_unit.input", x, enabled=self.config.enable_debug)
        residual = x
        x = self.act1(x)
        x = self.conv1(x)
        x = self.act2(x)
        x = self.conv2(x)

        _, residual_tokens, _ = residual.shape
        _, output_tokens, _ = x.shape
        if residual_tokens != output_tokens:
            aligned_tokens = min(residual_tokens, output_tokens)
            residual = residual[:, :aligned_tokens, :]
            x = x[:, :aligned_tokens, :]

        result = residual + x
        _debug_tensor("residual_unit.output", result, enabled=self.config.enable_debug)
        return result

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "act1": self.act1.export_weights(),
            "conv1": self.conv1.export_weights(),
            "act2": self.act2.export_weights(),
            "conv2": self.conv2.export_weights(),
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            act1=self.act1.import_weights(require_tree(weights["act1"])),
            conv1=self.conv1.import_weights(require_tree(weights["conv1"])),
            act2=self.act2.import_weights(require_tree(weights["act2"])),
            conv2=self.conv2.import_weights(require_tree(weights["conv2"])),
        )


@dataclass(frozen=True)
class Qwen3TTSDecoderBlockConfig:
    precision: DTypeLike
    snake_config: Qwen3TTSSnakeBetaConfig
    transposed_conv_config: Qwen3TTSCausalTransposeConv1dConfig
    residual_unit_config: Qwen3TTSResidualUnitConfig
    enable_debug: bool = True

    def empty(self, in_dim: int, out_dim: int, upsample_rate: int) -> "Qwen3TTSDecoderBlock":
        return Qwen3TTSDecoderBlock(
            config=self,
            snake=self.snake_config.empty(in_dim),
            transposed_conv=self.transposed_conv_config.empty(
                in_channels=in_dim,
                out_channels=out_dim,
                kernel_size=2 * upsample_rate,
                stride=upsample_rate,
                groups=1,
            ),
            residual_units=(
                self.residual_unit_config.empty(out_dim, dilation=1),
                self.residual_unit_config.empty(out_dim, dilation=3),
                self.residual_unit_config.empty(out_dim, dilation=9),
            ),
        )

    def random_init(
        self, in_dim: int, out_dim: int, upsample_rate: int, *, key: PRNGKeyArray
    ) -> "Qwen3TTSDecoderBlock":
        key_transposed, key_r1, key_r2, key_r3 = jax.random.split(key, 4)
        return Qwen3TTSDecoderBlock(
            config=self,
            snake=self.snake_config.random_init(in_dim, key=key),
            transposed_conv=self.transposed_conv_config.random_init(
                in_channels=in_dim,
                out_channels=out_dim,
                kernel_size=2 * upsample_rate,
                stride=upsample_rate,
                groups=1,
                key=key_transposed,
            ),
            residual_units=(
                self.residual_unit_config.random_init(out_dim, dilation=1, key=key_r1),
                self.residual_unit_config.random_init(out_dim, dilation=3, key=key_r2),
                self.residual_unit_config.random_init(out_dim, dilation=9, key=key_r3),
            ),
        )


class Qwen3TTSDecoderBlock(LalamoModule[Qwen3TTSDecoderBlockConfig]):
    snake: Qwen3TTSSnakeBeta
    transposed_conv: Qwen3TTSCausalTransposeConv1d
    residual_units: tuple[Qwen3TTSResidualUnit, Qwen3TTSResidualUnit, Qwen3TTSResidualUnit]

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def __call__(
        self,
        x: Float[Array, "batch tokens channels"],
    ) -> Float[Array, "batch tokens channels"]:
        _debug_tensor("decoder_block.input", x, enabled=self.config.enable_debug)
        x = self.snake(x)
        x = self.transposed_conv(x)
        for residual_unit in self.residual_units:
            x = residual_unit(x)
        _debug_tensor("decoder_block.output", x, enabled=self.config.enable_debug)
        return x

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "snake": self.snake.export_weights(),
            "transposed_conv": self.transposed_conv.export_weights(),
            "residual_units": [residual_unit.export_weights() for residual_unit in self.residual_units],
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        residual_unit_weights = weights["residual_units"]
        assert isinstance(residual_unit_weights, Sequence)
        return replace(
            self,
            snake=self.snake.import_weights(require_tree(weights["snake"])),
            transposed_conv=self.transposed_conv.import_weights(require_tree(weights["transposed_conv"])),
            residual_units=tuple(
                residual_unit.import_weights(require_tree(residual_weights))
                for residual_unit, residual_weights in zip(self.residual_units, residual_unit_weights, strict=True)
            ),
        )


@dataclass(frozen=True)
class Qwen3TTSEuclideanCodebookConfig:
    precision: DTypeLike
    epsilon: float = 1e-5

    def empty(self, dim: int, codebook_size: int) -> "Qwen3TTSEuclideanCodebook":
        return Qwen3TTSEuclideanCodebook(
            config=self,
            cluster_usage=jnp.ones((codebook_size,), dtype=self.precision),
            embedding_sum=jnp.zeros((codebook_size, dim), dtype=self.precision),
        )

    def random_init(self, dim: int, codebook_size: int, *, key: PRNGKeyArray) -> "Qwen3TTSEuclideanCodebook":
        return self.empty(dim=dim, codebook_size=codebook_size)


class Qwen3TTSEuclideanCodebook(LalamoModule[Qwen3TTSEuclideanCodebookConfig]):
    cluster_usage: Float[Array, " codebook_size"]
    embedding_sum: Float[Array, "codebook_size dim"]

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def codebook_size(self) -> int:
        (codebook_size,) = self.cluster_usage.shape
        return codebook_size

    def decode(
        self,
        codes: Int[Array, "batch tokens"],
    ) -> Float[Array, "batch tokens dim"]:
        embedding = self.embedding_sum / jnp.clip(self.cluster_usage, min=self.config.epsilon)[:, None]
        return jnp.take(embedding, codes, axis=0)

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "cluster_usage": self.cluster_usage,
            "embedding_sum": self.embedding_sum,
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            cluster_usage=require_array(weights["cluster_usage"]),
            embedding_sum=require_array(weights["embedding_sum"]),
        )


@dataclass(frozen=True)
class Qwen3TTSVectorQuantizationConfig:
    precision: DTypeLike
    codebook_config: Qwen3TTSEuclideanCodebookConfig
    project_out_config: FullPrecisionLinearConfig
    enable_debug: bool = True

    def empty(self, dim: int, codebook_size: int, codebook_dim: int | None = None) -> "Qwen3TTSVectorQuantization":
        codebook_dim = dim if codebook_dim is None else codebook_dim
        codebook = self.codebook_config.empty(dim=codebook_dim, codebook_size=codebook_size)
        if codebook_dim == dim:
            project_out = None
        else:
            project_out = self.project_out_config.empty(
                input_dim=codebook_dim,
                output_dims=(dim,),
                has_biases=True,
            )
        return Qwen3TTSVectorQuantization(
            config=self,
            codebook=codebook,
            project_out=project_out,
            output_dim=dim,
        )

    def random_init(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: int | None,
        *,
        key: PRNGKeyArray,
    ) -> "Qwen3TTSVectorQuantization":
        key_project = key
        codebook_dim = dim if codebook_dim is None else codebook_dim
        codebook = self.codebook_config.random_init(dim=codebook_dim, codebook_size=codebook_size, key=key)
        if codebook_dim == dim:
            project_out = None
        else:
            project_out = self.project_out_config.random_init(
                input_dim=codebook_dim,
                output_dims=(dim,),
                has_biases=True,
                key=key_project,
            )
        return Qwen3TTSVectorQuantization(
            config=self,
            codebook=codebook,
            project_out=project_out,
            output_dim=dim,
        )


class Qwen3TTSVectorQuantization(LalamoModule[Qwen3TTSVectorQuantizationConfig]):
    codebook: Qwen3TTSEuclideanCodebook
    project_out: FullPrecisionLinear | None
    output_dim: int

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def decode(
        self,
        codes: Int[Array, "batch tokens"],
    ) -> Float[Array, "batch channels tokens"]:
        quantized = self.codebook.decode(codes)
        if self.project_out is not None:
            (quantized,) = vmap(vmap(self.project_out))(quantized)
        quantized = rearrange(quantized, "batch tokens channels -> batch channels tokens")
        _debug_tensor("vector_quantization.output", quantized, enabled=self.config.enable_debug)
        return quantized

    def export_weights(self) -> ParameterTree[Array]:
        project_out_weights: ParameterTree[Array]
        if self.project_out is None:
            project_out_weights = {}
        else:
            project_out_weights = self.project_out.export_weights()
        return {
            "codebook": self.codebook.export_weights(),
            "project_out": project_out_weights,
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        if self.project_out is None:
            project_out = None
        else:
            project_out = self.project_out.import_weights(require_tree(weights["project_out"]))
        return replace(
            self,
            codebook=self.codebook.import_weights(require_tree(weights["codebook"])),
            project_out=project_out,
        )


@dataclass(frozen=True)
class Qwen3TTSResidualVectorQuantizationConfig:
    precision: DTypeLike
    vector_quantization_config: Qwen3TTSVectorQuantizationConfig
    enable_debug: bool = True

    def empty(
        self,
        num_quantizers: int,
        dim: int,
        codebook_size: int,
        codebook_dim: int | None = None,
    ) -> "Qwen3TTSResidualVectorQuantization":
        layers = tuple(
            self.vector_quantization_config.empty(
                dim=dim,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
            )
            for _ in range(num_quantizers)
        )
        return Qwen3TTSResidualVectorQuantization(config=self, layers=layers)

    def random_init(
        self,
        num_quantizers: int,
        dim: int,
        codebook_size: int,
        codebook_dim: int | None,
        *,
        key: PRNGKeyArray,
    ) -> "Qwen3TTSResidualVectorQuantization":
        keys = jax.random.split(key, num_quantizers)
        layers = tuple(
            self.vector_quantization_config.random_init(
                dim=dim,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                key=layer_key,
            )
            for layer_key in keys
        )
        return Qwen3TTSResidualVectorQuantization(config=self, layers=layers)


class Qwen3TTSResidualVectorQuantization(LalamoModule[Qwen3TTSResidualVectorQuantizationConfig]):
    layers: tuple[Qwen3TTSVectorQuantization, ...]

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def decode(
        self,
        codes: Int[Array, "num_quantizers batch tokens"],
    ) -> Float[Array, "batch channels tokens"]:
        layer_pairs = zip(self.layers, codes, strict=True)
        first_layer, first_codes = next(layer_pairs)
        quantized = first_layer.decode(first_codes)
        for layer, layer_codes in layer_pairs:
            quantized = quantized + layer.decode(layer_codes)
        _debug_tensor("residual_vector_quantization.output", quantized, enabled=self.config.enable_debug)
        return quantized

    def __call__(
        self,
        codes: Int[Array, "batch num_quantizers tokens"],
    ) -> Float[Array, "batch channels tokens"]:
        return self.decode(rearrange(codes, "batch num_quantizers tokens -> num_quantizers batch tokens"))

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "layers": [layer.export_weights() for layer in self.layers],
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        layers_weights = weights["layers"]
        assert isinstance(layers_weights, Sequence)
        return replace(
            self,
            layers=tuple(
                layer.import_weights(require_tree(layer_weights))
                for layer, layer_weights in zip(self.layers, layers_weights, strict=True)
            ),
        )


@dataclass(frozen=True)
class Qwen3TTSResidualVectorQuantizerConfig:
    precision: DTypeLike
    rvq_config: Qwen3TTSResidualVectorQuantizationConfig
    output_projection_config: FullPrecisionLinearConfig
    enable_debug: bool = True

    def empty(
        self,
        dimension: int,
        input_dimension: int,
        output_dimension: int,
        n_q: int,
        bins: int,
        force_projection: bool = False,
    ) -> "Qwen3TTSResidualVectorQuantizer":
        if input_dimension != dimension and not force_projection:
            raise ValueError("Decode path requires force_projection when input_dimension differs from dimension.")
        if output_dimension == dimension and not force_projection:
            output_projection = None
        else:
            output_projection = self.output_projection_config.empty(
                input_dim=dimension,
                output_dims=(output_dimension,),
                has_biases=False,
            )

        return Qwen3TTSResidualVectorQuantizer(
            config=self,
            dimension=dimension,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            n_q=n_q,
            bins=bins,
            rvq=self.rvq_config.empty(
                num_quantizers=n_q,
                dim=dimension,
                codebook_size=bins,
                codebook_dim=dimension,
            ),
            output_projection=output_projection,
        )

    def random_init(
        self,
        dimension: int,
        input_dimension: int,
        output_dimension: int,
        n_q: int,
        bins: int,
        *,
        key: PRNGKeyArray,
        force_projection: bool = False,
    ) -> "Qwen3TTSResidualVectorQuantizer":
        key_rvq, key_out = jax.random.split(key)
        if input_dimension != dimension and not force_projection:
            raise ValueError("Decode path requires force_projection when input_dimension differs from dimension.")
        if output_dimension == dimension and not force_projection:
            output_projection = None
        else:
            output_projection = self.output_projection_config.random_init(
                input_dim=dimension,
                output_dims=(output_dimension,),
                has_biases=False,
                key=key_out,
            )

        return Qwen3TTSResidualVectorQuantizer(
            config=self,
            dimension=dimension,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            n_q=n_q,
            bins=bins,
            rvq=self.rvq_config.random_init(
                num_quantizers=n_q,
                dim=dimension,
                codebook_size=bins,
                codebook_dim=dimension,
                key=key_rvq,
            ),
            output_projection=output_projection,
        )


class Qwen3TTSResidualVectorQuantizer(LalamoModule[Qwen3TTSResidualVectorQuantizerConfig]):
    dimension: int
    input_dimension: int
    output_dimension: int
    n_q: int
    bins: int

    rvq: Qwen3TTSResidualVectorQuantization
    output_projection: FullPrecisionLinear | None

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def decode(
        self,
        codes: Int[Array, "batch num_quantizers tokens"],
    ) -> Float[Array, "batch channels tokens"]:
        quantized = self.rvq.decode(rearrange(codes, "batch num_quantizers tokens -> num_quantizers batch tokens"))
        if self.output_projection is not None:
            quantized_nsc = rearrange(quantized, "batch channels tokens -> batch tokens channels")
            (quantized_nsc,) = vmap(vmap(self.output_projection))(quantized_nsc)
            quantized = rearrange(quantized_nsc, "batch tokens channels -> batch channels tokens")
        _debug_tensor("residual_vector_quantizer.output", quantized, enabled=self.config.enable_debug)
        return quantized

    def export_weights(self) -> ParameterTree[Array]:
        if self.output_projection is None:
            output_projection = {}
        else:
            output_projection = self.output_projection.export_weights()
        return {
            "rvq": self.rvq.export_weights(),
            "output_projection": output_projection,
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        if self.output_projection is None:
            output_projection = None
        else:
            output_projection = self.output_projection.import_weights(require_tree(weights["output_projection"]))
        return replace(
            self,
            rvq=self.rvq.import_weights(require_tree(weights["rvq"])),
            output_projection=output_projection,
        )


@dataclass(frozen=True)
class Qwen3TTSSplitResidualVectorQuantizerConfig:
    precision: DTypeLike
    residual_vector_quantizer_config: Qwen3TTSResidualVectorQuantizerConfig
    n_q_semantic: int = 1
    enable_debug: bool = True

    def empty(
        self,
        dimension: int,
        n_q: int,
        bins: int,
        input_dimension: int,
        output_dimension: int,
    ) -> "Qwen3TTSSplitResidualVectorQuantizer":
        if n_q <= self.n_q_semantic:
            raise ValueError(f"n_q must be > n_q_semantic ({self.n_q_semantic}), got {n_q}")

        rvq_first = self.residual_vector_quantizer_config.empty(
            dimension=dimension,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            n_q=self.n_q_semantic,
            bins=bins,
            force_projection=True,
        )
        rvq_rest = self.residual_vector_quantizer_config.empty(
            dimension=dimension,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            n_q=n_q - self.n_q_semantic,
            bins=bins,
            force_projection=True,
        )

        return Qwen3TTSSplitResidualVectorQuantizer(
            config=self,
            n_q=n_q,
            n_q_semantic=self.n_q_semantic,
            rvq_first=rvq_first,
            rvq_rest=rvq_rest,
        )

    def random_init(
        self,
        dimension: int,
        n_q: int,
        bins: int,
        input_dimension: int,
        output_dimension: int,
        *,
        key: PRNGKeyArray,
    ) -> "Qwen3TTSSplitResidualVectorQuantizer":
        key_first, key_rest = jax.random.split(key)
        if n_q <= self.n_q_semantic:
            raise ValueError(f"n_q must be > n_q_semantic ({self.n_q_semantic}), got {n_q}")

        rvq_first = self.residual_vector_quantizer_config.random_init(
            dimension=dimension,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            n_q=self.n_q_semantic,
            bins=bins,
            key=key_first,
            force_projection=True,
        )
        rvq_rest = self.residual_vector_quantizer_config.random_init(
            dimension=dimension,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            n_q=n_q - self.n_q_semantic,
            bins=bins,
            key=key_rest,
            force_projection=True,
        )

        return Qwen3TTSSplitResidualVectorQuantizer(
            config=self,
            n_q=n_q,
            n_q_semantic=self.n_q_semantic,
            rvq_first=rvq_first,
            rvq_rest=rvq_rest,
        )


class Qwen3TTSSplitResidualVectorQuantizer(LalamoModule[Qwen3TTSSplitResidualVectorQuantizerConfig]):
    n_q: int
    n_q_semantic: int
    rvq_first: Qwen3TTSResidualVectorQuantizer
    rvq_rest: Qwen3TTSResidualVectorQuantizer

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def decode(
        self,
        codes: Int[Array, "batch num_quantizers tokens"],
    ) -> Float[Array, "batch channels tokens"]:
        quantized = self.rvq_first.decode(codes[:, : self.n_q_semantic, :])
        _, num_quantizers, _ = codes.shape
        if num_quantizers > self.n_q_semantic:
            quantized = quantized + self.rvq_rest.decode(codes[:, self.n_q_semantic :, :])
        _debug_tensor("split_residual_vector_quantizer.output", quantized, enabled=self.config.enable_debug)
        return quantized

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "rvq_first": self.rvq_first.export_weights(),
            "rvq_rest": self.rvq_rest.export_weights(),
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            rvq_first=self.rvq_first.import_weights(require_tree(weights["rvq_first"])),
            rvq_rest=self.rvq_rest.import_weights(require_tree(weights["rvq_rest"])),
        )
