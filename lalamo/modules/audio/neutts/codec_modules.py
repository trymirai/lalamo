from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from math import prod
from typing import Self, cast

import jax.numpy as jnp
from jax import lax, nn, vmap
from jaxtyping import Array, Complex, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree, require_array, require_tree
from lalamo.modules.common import LalamoModule

__all__ = [
    "NeuCodecAttention",
    "NeuCodecAttentionConfig",
    "NeuCodecConv1d",
    "NeuCodecConv1dConfig",
    "NeuCodecFSQ",
    "NeuCodecFSQConfig",
    "NeuCodecGroupNorm",
    "NeuCodecGroupNormConfig",
    "NeuCodecISTFTHead",
    "NeuCodecISTFTHeadConfig",
    "NeuCodecLayerNorm",
    "NeuCodecLayerNormConfig",
    "NeuCodecLinear",
    "NeuCodecLinearConfig",
    "NeuCodecMLP",
    "NeuCodecMLPConfig",
    "NeuCodecRMSNorm",
    "NeuCodecRMSNormConfig",
    "NeuCodecResidualFSQ",
    "NeuCodecResidualFSQConfig",
    "NeuCodecResnetBlock",
    "NeuCodecResnetBlockConfig",
    "NeuCodecTransformerBlock",
    "NeuCodecTransformerBlockConfig",
    "NeuCodecVocosBackbone",
    "NeuCodecVocosBackboneConfig",
    "NeuCodecVocosDecoder",
    "NeuCodecVocosDecoderConfig",
]


@dataclass(frozen=True)
class NeuCodecFSQConfig:
    levels: tuple[int, ...]
    precision: DTypeLike

    def __post_init__(self) -> None:
        if not self.levels:
            raise ValueError("NeuCodec FSQ levels must be non-empty.")
        if any(level <= 1 for level in self.levels):
            raise ValueError("NeuCodec FSQ levels must all be greater than 1.")

    @property
    def codebook_dim(self) -> int:
        return len(self.levels)

    @property
    def codebook_size(self) -> int:
        return prod(self.levels)

    def empty(self) -> "NeuCodecFSQ":
        levels = jnp.asarray(self.levels, dtype=jnp.int32)
        basis = jnp.cumprod(jnp.concatenate([jnp.asarray([1], dtype=jnp.int32), levels[:-1]]))
        return NeuCodecFSQ(
            config=self,
            levels=levels,
            basis=basis,
        )

    def random_init(
        self,
        *,
        key: PRNGKeyArray,  # noqa: ARG002
    ) -> "NeuCodecFSQ":
        return self.empty()


class NeuCodecFSQ(LalamoModule[NeuCodecFSQConfig]):
    levels: Int[Array, " codebook_dim"]
    basis: Int[Array, " codebook_dim"]

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def indices_to_level_indices(
        self,
        indices: Int[Array, "batch tokens"],
    ) -> Int[Array, "batch tokens codebook_dim"]:
        return (indices[..., None] // self.basis) % self.levels

    def indices_to_codes(
        self,
        indices: Int[Array, "batch tokens"],
    ) -> Float[Array, "batch tokens codebook_dim"]:
        level_indices = self.indices_to_level_indices(indices)
        half_width = self.levels // 2
        return (level_indices - half_width).astype(self.config.precision) / half_width.astype(self.config.precision)

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "levels": self.levels,
            "basis": self.basis,
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            levels=require_array(weights["levels"]),
            basis=require_array(weights["basis"]),
        )


@dataclass(frozen=True)
class NeuCodecLinearConfig:
    input_dim: int
    output_dim: int
    precision: DTypeLike
    has_bias: bool = True

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("NeuCodec linear input_dim must be positive.")
        if self.output_dim <= 0:
            raise ValueError("NeuCodec linear output_dim must be positive.")

    def empty(self) -> "NeuCodecLinear":
        weights = jnp.zeros((self.output_dim, self.input_dim), dtype=self.precision)
        biases = jnp.zeros((self.output_dim,), dtype=self.precision) if self.has_bias else None
        return NeuCodecLinear(config=self, weights=weights, biases=biases)

    def random_init(
        self,
        *,
        key: PRNGKeyArray,  # noqa: ARG002
    ) -> "NeuCodecLinear":
        return self.empty()


class NeuCodecLinear(LalamoModule[NeuCodecLinearConfig]):
    weights: Float[Array, "output_dim input_dim"]
    biases: Float[Array, " output_dim"] | None

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def __call__(
        self,
        inputs: Float[Array, "*batch input_dim"],
    ) -> Float[Array, "*batch output_dim"]:
        output = jnp.einsum("...i,oi->...o", inputs, self.weights)
        if self.biases is not None:
            output = output + self.biases
        return output

    def export_weights(self) -> ParameterTree[Array]:
        result: dict[str, Array] = {"weights": self.weights}
        if self.biases is not None:
            result["biases"] = self.biases
        return result

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            weights=require_array(weights["weights"]),
            biases=require_array(weights["biases"]) if self.biases is not None else None,
        )


@dataclass(frozen=True)
class NeuCodecISTFTHeadConfig:
    dim: int
    n_fft: int
    hop_length: int
    precision: DTypeLike

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("NeuCodec ISTFTHead dim must be positive.")
        if self.n_fft <= 0:
            raise ValueError("NeuCodec ISTFTHead n_fft must be positive.")
        if self.n_fft % 2 != 0:
            raise ValueError("NeuCodec ISTFTHead n_fft must be even.")
        if self.hop_length <= 0:
            raise ValueError("NeuCodec ISTFTHead hop_length must be positive.")
        if self.hop_length >= self.n_fft:
            raise ValueError("NeuCodec ISTFTHead hop_length must be smaller than n_fft.")
        if (self.n_fft - self.hop_length) % 2 != 0:
            raise ValueError("NeuCodec ISTFTHead same padding requires an even n_fft - hop_length.")

    @property
    def frequency_bins(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def output_dim(self) -> int:
        return 2 * self.frequency_bins

    def empty(self) -> "NeuCodecISTFTHead":
        return NeuCodecISTFTHead(
            config=self,
            out=NeuCodecLinearConfig(
                input_dim=self.dim,
                output_dim=self.output_dim,
                precision=self.precision,
            ).empty(),
        )

    def random_init(
        self,
        *,
        key: PRNGKeyArray,  # noqa: ARG002
    ) -> "NeuCodecISTFTHead":
        return self.empty()


class NeuCodecISTFTHead(LalamoModule[NeuCodecISTFTHeadConfig]):
    out: NeuCodecLinear

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def __call__(
        self,
        inputs: Float[Array, "batch tokens dim"],
    ) -> Float[Array, "batch channels samples"]:
        projected = jnp.transpose(self.out(inputs), (0, 2, 1))
        magnitudes, phases = jnp.split(projected, 2, axis=1)
        magnitudes = jnp.minimum(jnp.exp(magnitudes), jnp.asarray(1e2, dtype=magnitudes.dtype))
        spectrogram = magnitudes * (jnp.cos(phases) + 1j * jnp.sin(phases))
        audio = _neucodec_istft_same(
            spectrogram,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
        )
        return audio[:, None, :]

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "out": self.out.export_weights(),
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            out=self.out.import_weights(require_tree(weights["out"])),
        )


@dataclass(frozen=True)
class NeuCodecLayerNormConfig:
    dim: int
    precision: DTypeLike
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("NeuCodec LayerNorm dim must be positive.")
        if self.eps <= 0:
            raise ValueError("NeuCodec LayerNorm eps must be positive.")

    def empty(self) -> "NeuCodecLayerNorm":
        return NeuCodecLayerNorm(
            config=self,
            weights=jnp.ones((self.dim,), dtype=self.precision),
            biases=jnp.zeros((self.dim,), dtype=self.precision),
        )

    def random_init(
        self,
        *,
        key: PRNGKeyArray,  # noqa: ARG002
    ) -> "NeuCodecLayerNorm":
        return self.empty()


class NeuCodecLayerNorm(LalamoModule[NeuCodecLayerNormConfig]):
    weights: Float[Array, " dim"]
    biases: Float[Array, " dim"]

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def __call__(
        self,
        inputs: Float[Array, "*batch dim"],
    ) -> Float[Array, "*batch dim"]:
        mean = jnp.mean(inputs, axis=-1, keepdims=True)
        variance = jnp.mean(jnp.square(inputs - mean), axis=-1, keepdims=True)
        normalized = (inputs - mean) * lax.rsqrt(variance + self.config.eps)
        return normalized * self.weights + self.biases

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "weights": self.weights,
            "biases": self.biases,
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            weights=require_array(weights["weights"]),
            biases=require_array(weights["biases"]),
        )


def _neucodec_istft_same(
    spectrogram: Complex[Array, "batch frequency_bins frames"],
    *,
    n_fft: int,
    hop_length: int,
) -> Float[Array, "batch samples"]:
    _, _, num_frames = spectrogram.shape
    window = _periodic_hann_window(n_fft, dtype=jnp.float32)
    inverse_fft = jnp.fft.irfft(spectrogram, n=n_fft, axis=1, norm="backward") * window[None, :, None]
    frames = jnp.transpose(inverse_fft, (0, 2, 1))
    output_size = (num_frames - 1) * hop_length + n_fft
    sample_indices = _istft_frame_sample_indices(
        num_frames=num_frames,
        win_length=n_fft,
        hop_length=hop_length,
    )
    audio = vmap(lambda batch_frames: _overlap_add(batch_frames, sample_indices, output_size))(frames)
    window_envelope = _overlap_add(
        jnp.broadcast_to(jnp.square(window)[None, :], (num_frames, n_fft)),
        sample_indices,
        output_size,
    )
    pad = (n_fft - hop_length) // 2
    return audio[:, pad:-pad] / window_envelope[pad:-pad]


def _periodic_hann_window(
    win_length: int,
    *,
    dtype: DTypeLike,
) -> Float[Array, " win_length"]:
    positions = jnp.arange(win_length, dtype=jnp.float32)
    window = 0.5 - 0.5 * jnp.cos(2.0 * jnp.pi * positions / win_length)
    return window.astype(dtype)


def _istft_frame_sample_indices(
    *,
    num_frames: int,
    win_length: int,
    hop_length: int,
) -> Int[Array, "frames win_length"]:
    frame_starts = hop_length * jnp.arange(num_frames, dtype=jnp.int32)
    sample_offsets = jnp.arange(win_length, dtype=jnp.int32)
    return frame_starts[:, None] + sample_offsets[None, :]


def _overlap_add(
    frames: Float[Array, "frames win_length"],
    sample_indices: Int[Array, "frames win_length"],
    output_size: int,
) -> Float[Array, " samples"]:
    return jnp.zeros((output_size,), dtype=frames.dtype).at[sample_indices].add(frames)


@dataclass(frozen=True)
class NeuCodecConv1dConfig:
    in_channels: int
    out_channels: int
    kernel_size: int
    padding: int
    precision: DTypeLike
    has_bias: bool = True

    def __post_init__(self) -> None:
        if self.in_channels <= 0:
            raise ValueError("NeuCodec Conv1d in_channels must be positive.")
        if self.out_channels <= 0:
            raise ValueError("NeuCodec Conv1d out_channels must be positive.")
        if self.kernel_size <= 0:
            raise ValueError("NeuCodec Conv1d kernel_size must be positive.")
        if self.padding < 0:
            raise ValueError("NeuCodec Conv1d padding must be non-negative.")

    def empty(self) -> "NeuCodecConv1d":
        weights = jnp.zeros((self.out_channels, self.in_channels, self.kernel_size), dtype=self.precision)
        biases = jnp.zeros((self.out_channels,), dtype=self.precision) if self.has_bias else None
        return NeuCodecConv1d(config=self, weights=weights, biases=biases)

    def random_init(
        self,
        *,
        key: PRNGKeyArray,  # noqa: ARG002
    ) -> "NeuCodecConv1d":
        return self.empty()


class NeuCodecConv1d(LalamoModule[NeuCodecConv1dConfig]):
    weights: Float[Array, "out_channels in_channels kernel_size"]
    biases: Float[Array, " out_channels"] | None

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def __call__(
        self,
        inputs: Float[Array, "batch sequence in_channels"],
    ) -> Float[Array, "batch sequence_out out_channels"]:
        output = lax.conv_general_dilated(
            inputs,
            self.weights,
            window_strides=(1,),
            padding=((self.config.padding, self.config.padding),),
            dimension_numbers=("NHC", "OIH", "NHC"),
        )
        if self.biases is not None:
            output = output + self.biases[None, None, :]
        return output

    def export_weights(self) -> ParameterTree[Array]:
        result: dict[str, Array] = {"weights": self.weights}
        if self.biases is not None:
            result["biases"] = self.biases
        return result

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            weights=require_array(weights["weights"]),
            biases=require_array(weights["biases"]) if self.biases is not None else None,
        )


@dataclass(frozen=True)
class NeuCodecGroupNormConfig:
    channels: int
    precision: DTypeLike
    num_groups: int = 32
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.channels <= 0:
            raise ValueError("NeuCodec GroupNorm channels must be positive.")
        if self.num_groups <= 0:
            raise ValueError("NeuCodec GroupNorm num_groups must be positive.")
        if self.channels % self.num_groups != 0:
            raise ValueError("NeuCodec GroupNorm channels must be divisible by num_groups.")
        if self.eps <= 0:
            raise ValueError("NeuCodec GroupNorm eps must be positive.")

    def empty(self) -> "NeuCodecGroupNorm":
        return NeuCodecGroupNorm(
            config=self,
            weights=jnp.ones((self.channels,), dtype=self.precision),
            biases=jnp.zeros((self.channels,), dtype=self.precision),
        )

    def random_init(
        self,
        *,
        key: PRNGKeyArray,  # noqa: ARG002
    ) -> "NeuCodecGroupNorm":
        return self.empty()


class NeuCodecGroupNorm(LalamoModule[NeuCodecGroupNormConfig]):
    weights: Float[Array, " channels"]
    biases: Float[Array, " channels"]

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def __call__(
        self,
        inputs: Float[Array, "batch sequence channels"],
    ) -> Float[Array, "batch sequence channels"]:
        batch_size, sequence_length, channels = inputs.shape
        channels_per_group = channels // self.config.num_groups
        grouped_inputs = inputs.reshape(batch_size, sequence_length, self.config.num_groups, channels_per_group)
        mean = jnp.mean(grouped_inputs, axis=(1, 3), keepdims=True)
        variance = jnp.mean(jnp.square(grouped_inputs - mean), axis=(1, 3), keepdims=True)
        normalized = (grouped_inputs - mean) * lax.rsqrt(variance + self.config.eps)
        return normalized.reshape(inputs.shape) * self.weights[None, None, :] + self.biases[None, None, :]

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "weights": self.weights,
            "biases": self.biases,
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            weights=require_array(weights["weights"]),
            biases=require_array(weights["biases"]),
        )


@dataclass(frozen=True)
class NeuCodecResnetBlockConfig:
    channels: int
    precision: DTypeLike

    def __post_init__(self) -> None:
        if self.channels <= 0:
            raise ValueError("NeuCodec ResnetBlock channels must be positive.")
        NeuCodecGroupNormConfig(channels=self.channels, precision=self.precision)

    def empty(self) -> "NeuCodecResnetBlock":
        norm_config = NeuCodecGroupNormConfig(channels=self.channels, precision=self.precision)
        conv_config = NeuCodecConv1dConfig(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=3,
            padding=1,
            precision=self.precision,
        )
        return NeuCodecResnetBlock(
            config=self,
            norm1=norm_config.empty(),
            conv1=conv_config.empty(),
            norm2=norm_config.empty(),
            conv2=conv_config.empty(),
        )

    def random_init(
        self,
        *,
        key: PRNGKeyArray,  # noqa: ARG002
    ) -> "NeuCodecResnetBlock":
        return self.empty()


class NeuCodecResnetBlock(LalamoModule[NeuCodecResnetBlockConfig]):
    norm1: NeuCodecGroupNorm
    conv1: NeuCodecConv1d
    norm2: NeuCodecGroupNorm
    conv2: NeuCodecConv1d

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def __call__(
        self,
        inputs: Float[Array, "batch sequence channels"],
    ) -> Float[Array, "batch sequence channels"]:
        hidden_states = self.conv1(_swish(self.norm1(inputs)))
        hidden_states = self.conv2(_swish(self.norm2(hidden_states)))
        return inputs + hidden_states

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "norm1": self.norm1.export_weights(),
            "conv1": self.conv1.export_weights(),
            "norm2": self.norm2.export_weights(),
            "conv2": self.conv2.export_weights(),
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            norm1=self.norm1.import_weights(require_tree(weights["norm1"])),
            conv1=self.conv1.import_weights(require_tree(weights["conv1"])),
            norm2=self.norm2.import_weights(require_tree(weights["norm2"])),
            conv2=self.conv2.import_weights(require_tree(weights["conv2"])),
        )


def _swish(inputs: Float[Array, "*batch"]) -> Float[Array, "*batch"]:
    return inputs * nn.sigmoid(inputs)


@dataclass(frozen=True)
class NeuCodecRMSNormConfig:
    dim: int
    precision: DTypeLike
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("NeuCodec RMSNorm dim must be positive.")
        if self.eps <= 0:
            raise ValueError("NeuCodec RMSNorm eps must be positive.")

    def empty(self) -> "NeuCodecRMSNorm":
        return NeuCodecRMSNorm(
            config=self,
            weights=jnp.ones((self.dim,), dtype=self.precision),
        )

    def random_init(
        self,
        *,
        key: PRNGKeyArray,  # noqa: ARG002
    ) -> "NeuCodecRMSNorm":
        return self.empty()


class NeuCodecRMSNorm(LalamoModule[NeuCodecRMSNormConfig]):
    weights: Float[Array, " dim"]

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def __call__(
        self,
        inputs: Float[Array, "*batch dim"],
    ) -> Float[Array, "*batch dim"]:
        norm_x = jnp.mean(jnp.square(inputs), axis=-1, keepdims=True)
        return inputs * lax.rsqrt(norm_x + self.config.eps) * self.weights

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "weights": self.weights,
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            weights=require_array(weights["weights"]),
        )


@dataclass(frozen=True)
class NeuCodecAttentionConfig:
    dim: int
    num_heads: int
    rotary_dim: int
    precision: DTypeLike
    rope_base: float = 10_000.0

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("NeuCodec attention dim must be positive.")
        if self.num_heads <= 0:
            raise ValueError("NeuCodec attention num_heads must be positive.")
        if self.dim % self.num_heads != 0:
            raise ValueError("NeuCodec attention dim must be divisible by num_heads.")
        if self.rotary_dim <= 0:
            raise ValueError("NeuCodec attention rotary_dim must be positive.")
        if self.rotary_dim % 2 != 0:
            raise ValueError("NeuCodec attention rotary_dim must be even.")
        if self.rotary_dim > self.head_dim:
            raise ValueError("NeuCodec attention rotary_dim must not exceed head_dim.")
        if self.rope_base <= 0:
            raise ValueError("NeuCodec attention rope_base must be positive.")

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads

    def empty(self) -> "NeuCodecAttention":
        return NeuCodecAttention(
            config=self,
            c_attn=NeuCodecLinearConfig(
                input_dim=self.dim,
                output_dim=3 * self.dim,
                precision=self.precision,
                has_bias=False,
            ).empty(),
            c_proj=NeuCodecLinearConfig(
                input_dim=self.dim,
                output_dim=self.dim,
                precision=self.precision,
                has_bias=False,
            ).empty(),
        )

    def random_init(
        self,
        *,
        key: PRNGKeyArray,  # noqa: ARG002
    ) -> "NeuCodecAttention":
        return self.empty()


class NeuCodecAttention(LalamoModule[NeuCodecAttentionConfig]):
    c_attn: NeuCodecLinear
    c_proj: NeuCodecLinear

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def __call__(
        self,
        inputs: Float[Array, "batch tokens dim"],
    ) -> Float[Array, "batch tokens dim"]:
        batch_size, num_tokens, _ = inputs.shape
        head_dim = self.config.head_dim
        qkv = self.c_attn(inputs).reshape(batch_size, num_tokens, 3, self.config.num_heads, head_dim)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        queries = _apply_neucodec_rotary_embeddings(
            queries,
            rotary_dim=self.config.rotary_dim,
            base=self.config.rope_base,
        )
        keys = _apply_neucodec_rotary_embeddings(
            keys,
            rotary_dim=self.config.rotary_dim,
            base=self.config.rope_base,
        )
        attention_logits = jnp.einsum("bhtd,bhsd->bhts", queries, keys) * (head_dim**-0.5)
        attention_weights = nn.softmax(attention_logits, axis=-1)
        attended = jnp.einsum("bhts,bhsd->bhtd", attention_weights, values)
        output = jnp.transpose(attended, (0, 2, 1, 3)).reshape(batch_size, num_tokens, self.config.dim)
        return self.c_proj(output)

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "c_attn": self.c_attn.export_weights(),
            "c_proj": self.c_proj.export_weights(),
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            c_attn=self.c_attn.import_weights(require_tree(weights["c_attn"])),
            c_proj=self.c_proj.import_weights(require_tree(weights["c_proj"])),
        )


def _apply_neucodec_rotary_embeddings(
    inputs: Float[Array, "batch heads tokens head_dim"],
    *,
    rotary_dim: int,
    base: float,
) -> Float[Array, "batch heads tokens head_dim"]:
    _, num_heads, _, _ = inputs.shape
    positions = jnp.arange(num_heads, dtype=jnp.float32)
    channel_indices = jnp.arange(0, rotary_dim, 2, dtype=jnp.float32)
    theta = 1.0 / (base ** (channel_indices / rotary_dim))
    idx_theta = jnp.einsum("i,j->ij", positions, theta)
    cosines = jnp.cos(idx_theta)[None, :, None, :, None]
    sines = jnp.sin(idx_theta)[None, :, None, :, None]

    rotated = inputs[..., :rotary_dim].astype(jnp.float32).reshape(*inputs.shape[:-1], rotary_dim // 2, 2)
    rotated_output = jnp.stack(
        [
            rotated[..., 0] * cosines[..., 0] - rotated[..., 1] * sines[..., 0],
            rotated[..., 1] * cosines[..., 0] + rotated[..., 0] * sines[..., 0],
        ],
        axis=-1,
    ).reshape(*inputs.shape[:-1], rotary_dim)
    rotated_output = rotated_output.astype(inputs.dtype)
    if rotary_dim == inputs.shape[-1]:
        return rotated_output
    return jnp.concatenate([rotated_output, inputs[..., rotary_dim:]], axis=-1)


@dataclass(frozen=True)
class NeuCodecMLPConfig:
    dim: int
    precision: DTypeLike

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("NeuCodec MLP dim must be positive.")

    def empty(self) -> "NeuCodecMLP":
        return NeuCodecMLP(
            config=self,
            fc1=NeuCodecLinearConfig(
                input_dim=self.dim,
                output_dim=4 * self.dim,
                precision=self.precision,
                has_bias=False,
            ).empty(),
            fc2=NeuCodecLinearConfig(
                input_dim=4 * self.dim,
                output_dim=self.dim,
                precision=self.precision,
                has_bias=False,
            ).empty(),
        )

    def random_init(
        self,
        *,
        key: PRNGKeyArray,  # noqa: ARG002
    ) -> "NeuCodecMLP":
        return self.empty()


class NeuCodecMLP(LalamoModule[NeuCodecMLPConfig]):
    fc1: NeuCodecLinear
    fc2: NeuCodecLinear

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def __call__(
        self,
        inputs: Float[Array, "batch tokens dim"],
    ) -> Float[Array, "batch tokens dim"]:
        return self.fc2(_swish(self.fc1(inputs)))

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "fc1": self.fc1.export_weights(),
            "fc2": self.fc2.export_weights(),
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            fc1=self.fc1.import_weights(require_tree(weights["fc1"])),
            fc2=self.fc2.import_weights(require_tree(weights["fc2"])),
        )


@dataclass(frozen=True)
class NeuCodecTransformerBlockConfig:
    dim: int
    num_heads: int
    rotary_dim: int
    precision: DTypeLike
    eps: float = 1e-6
    rope_base: float = 10_000.0

    def __post_init__(self) -> None:
        NeuCodecRMSNormConfig(dim=self.dim, precision=self.precision, eps=self.eps)
        NeuCodecAttentionConfig(
            dim=self.dim,
            num_heads=self.num_heads,
            rotary_dim=self.rotary_dim,
            precision=self.precision,
            rope_base=self.rope_base,
        )
        NeuCodecMLPConfig(dim=self.dim, precision=self.precision)

    def empty(self) -> "NeuCodecTransformerBlock":
        return NeuCodecTransformerBlock(
            config=self,
            att_norm=NeuCodecRMSNormConfig(dim=self.dim, precision=self.precision, eps=self.eps).empty(),
            ffn_norm=NeuCodecRMSNormConfig(dim=self.dim, precision=self.precision, eps=self.eps).empty(),
            att=NeuCodecAttentionConfig(
                dim=self.dim,
                num_heads=self.num_heads,
                rotary_dim=self.rotary_dim,
                precision=self.precision,
                rope_base=self.rope_base,
            ).empty(),
            mlp=NeuCodecMLPConfig(dim=self.dim, precision=self.precision).empty(),
        )

    def random_init(
        self,
        *,
        key: PRNGKeyArray,  # noqa: ARG002
    ) -> "NeuCodecTransformerBlock":
        return self.empty()


class NeuCodecTransformerBlock(LalamoModule[NeuCodecTransformerBlockConfig]):
    att_norm: NeuCodecRMSNorm
    ffn_norm: NeuCodecRMSNorm
    att: NeuCodecAttention
    mlp: NeuCodecMLP

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def __call__(
        self,
        inputs: Float[Array, "batch tokens dim"],
    ) -> Float[Array, "batch tokens dim"]:
        hidden_states = inputs + self.att(self.att_norm(inputs))
        return hidden_states + self.mlp(self.ffn_norm(hidden_states))

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "att_norm": self.att_norm.export_weights(),
            "ffn_norm": self.ffn_norm.export_weights(),
            "att": self.att.export_weights(),
            "mlp": self.mlp.export_weights(),
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            att_norm=self.att_norm.import_weights(require_tree(weights["att_norm"])),
            ffn_norm=self.ffn_norm.import_weights(require_tree(weights["ffn_norm"])),
            att=self.att.import_weights(require_tree(weights["att"])),
            mlp=self.mlp.import_weights(require_tree(weights["mlp"])),
        )


@dataclass(frozen=True)
class NeuCodecVocosBackboneConfig:
    hidden_dim: int
    depth: int
    heads: int
    rotary_dim: int
    precision: DTypeLike
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError("NeuCodec VocosBackbone hidden_dim must be positive.")
        if self.depth < 0:
            raise ValueError("NeuCodec VocosBackbone depth must be non-negative.")
        if self.heads <= 0:
            raise ValueError("NeuCodec VocosBackbone heads must be positive.")
        NeuCodecConv1dConfig(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=7,
            padding=3,
            precision=self.precision,
        )
        NeuCodecResnetBlockConfig(channels=self.hidden_dim, precision=self.precision)
        NeuCodecTransformerBlockConfig(
            dim=self.hidden_dim,
            num_heads=self.heads,
            rotary_dim=self.rotary_dim,
            precision=self.precision,
            eps=self.eps,
        )
        NeuCodecLayerNormConfig(dim=self.hidden_dim, precision=self.precision, eps=self.eps)

    def empty(self) -> "NeuCodecVocosBackbone":
        resnet_config = NeuCodecResnetBlockConfig(channels=self.hidden_dim, precision=self.precision)
        transformer_config = NeuCodecTransformerBlockConfig(
            dim=self.hidden_dim,
            num_heads=self.heads,
            rotary_dim=self.rotary_dim,
            precision=self.precision,
            eps=self.eps,
        )
        return NeuCodecVocosBackbone(
            config=self,
            embed=NeuCodecConv1dConfig(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                kernel_size=7,
                padding=3,
                precision=self.precision,
            ).empty(),
            prior_net=tuple(resnet_config.empty() for _ in range(2)),
            transformers=tuple(transformer_config.empty() for _ in range(self.depth)),
            post_net=tuple(resnet_config.empty() for _ in range(2)),
            final_layer_norm=NeuCodecLayerNormConfig(
                dim=self.hidden_dim,
                precision=self.precision,
                eps=self.eps,
            ).empty(),
        )

    def random_init(
        self,
        *,
        key: PRNGKeyArray,  # noqa: ARG002
    ) -> "NeuCodecVocosBackbone":
        return self.empty()


class NeuCodecVocosBackbone(LalamoModule[NeuCodecVocosBackboneConfig]):
    embed: NeuCodecConv1d
    prior_net: tuple[NeuCodecResnetBlock, ...]
    transformers: tuple[NeuCodecTransformerBlock, ...]
    post_net: tuple[NeuCodecResnetBlock, ...]
    final_layer_norm: NeuCodecLayerNorm

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def __call__(
        self,
        inputs: Float[Array, "batch tokens hidden_dim"],
    ) -> Float[Array, "batch tokens hidden_dim"]:
        hidden_states = self.embed(inputs)
        for block in self.prior_net:
            hidden_states = block(hidden_states)
        for block in self.transformers:
            hidden_states = block(hidden_states)
        for block in self.post_net:
            hidden_states = block(hidden_states)
        return self.final_layer_norm(hidden_states)

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "embed": self.embed.export_weights(),
            "prior_net": [block.export_weights() for block in self.prior_net],
            "transformers": [block.export_weights() for block in self.transformers],
            "post_net": [block.export_weights() for block in self.post_net],
            "final_layer_norm": self.final_layer_norm.export_weights(),
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        prior_net_weights = _require_sequence_weights(require_tree(weights["prior_net"]))
        transformer_weights = _require_sequence_weights(require_tree(weights["transformers"]))
        post_net_weights = _require_sequence_weights(require_tree(weights["post_net"]))
        return replace(
            self,
            embed=self.embed.import_weights(require_tree(weights["embed"])),
            prior_net=tuple(
                block.import_weights(require_tree(block_weights))
                for block, block_weights in zip(self.prior_net, prior_net_weights, strict=True)
            ),
            transformers=tuple(
                block.import_weights(require_tree(block_weights))
                for block, block_weights in zip(self.transformers, transformer_weights, strict=True)
            ),
            post_net=tuple(
                block.import_weights(require_tree(block_weights))
                for block, block_weights in zip(self.post_net, post_net_weights, strict=True)
            ),
            final_layer_norm=self.final_layer_norm.import_weights(require_tree(weights["final_layer_norm"])),
        )


@dataclass(frozen=True)
class NeuCodecVocosDecoderConfig:
    hidden_dim: int
    depth: int
    heads: int
    rotary_dim: int
    hop_length: int
    precision: DTypeLike

    def __post_init__(self) -> None:
        if self.hop_length <= 0:
            raise ValueError("NeuCodec VocosDecoder hop_length must be positive.")
        NeuCodecVocosBackboneConfig(
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            heads=self.heads,
            rotary_dim=self.rotary_dim,
            precision=self.precision,
        )
        NeuCodecISTFTHeadConfig(
            dim=self.hidden_dim,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            precision=self.precision,
        )

    @property
    def n_fft(self) -> int:
        return self.hop_length * 4

    def empty(self) -> "NeuCodecVocosDecoder":
        return NeuCodecVocosDecoder(
            config=self,
            backbone=NeuCodecVocosBackboneConfig(
                hidden_dim=self.hidden_dim,
                depth=self.depth,
                heads=self.heads,
                rotary_dim=self.rotary_dim,
                precision=self.precision,
            ).empty(),
            head=NeuCodecISTFTHeadConfig(
                dim=self.hidden_dim,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                precision=self.precision,
            ).empty(),
        )

    def random_init(
        self,
        *,
        key: PRNGKeyArray,  # noqa: ARG002
    ) -> "NeuCodecVocosDecoder":
        return self.empty()


class NeuCodecVocosDecoder(LalamoModule[NeuCodecVocosDecoderConfig]):
    backbone: NeuCodecVocosBackbone
    head: NeuCodecISTFTHead

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def __call__(
        self,
        inputs: Float[Array, "batch tokens hidden_dim"],
    ) -> Float[Array, "batch channels samples"]:
        return self.head(self.backbone(inputs))

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "backbone": self.backbone.export_weights(),
            "head": self.head.export_weights(),
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            backbone=self.backbone.import_weights(require_tree(weights["backbone"])),
            head=self.head.import_weights(require_tree(weights["head"])),
        )


def _require_sequence_weights(weights: ParameterTree[Array]) -> Sequence[Array | ParameterTree[Array]]:
    if not isinstance(weights, Sequence):
        raise TypeError("NeuCodec weight tree entry must be a sequence.")
    return cast("Sequence[Array | ParameterTree[Array]]", weights)


@dataclass(frozen=True)
class NeuCodecResidualFSQConfig:
    levels: tuple[int, ...]
    num_quantizers: int
    output_dim: int
    precision: DTypeLike

    def __post_init__(self) -> None:
        if self.num_quantizers <= 0:
            raise ValueError("NeuCodec residual FSQ num_quantizers must be positive.")
        if self.num_quantizers != 1:
            raise ValueError("NeuCodec residual FSQ currently supports num_quantizers == 1 only.")
        if self.output_dim <= 0:
            raise ValueError("NeuCodec residual FSQ output_dim must be positive.")
        NeuCodecFSQConfig(levels=self.levels, precision=self.precision)

    @property
    def codebook_dim(self) -> int:
        return len(self.levels)

    def empty(self) -> "NeuCodecResidualFSQ":
        return NeuCodecResidualFSQ(
            config=self,
            fsq=NeuCodecFSQConfig(levels=self.levels, precision=self.precision).empty(),
            project_out=NeuCodecLinearConfig(
                input_dim=self.codebook_dim,
                output_dim=self.output_dim,
                precision=self.precision,
            ).empty(),
        )

    def random_init(
        self,
        *,
        key: PRNGKeyArray,  # noqa: ARG002
    ) -> "NeuCodecResidualFSQ":
        return self.empty()


class NeuCodecResidualFSQ(LalamoModule[NeuCodecResidualFSQConfig]):
    fsq: NeuCodecFSQ
    project_out: NeuCodecLinear

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def get_output_from_indices(
        self,
        indices: Int[Array, "batch tokens num_quantizers"],
    ) -> Float[Array, "batch tokens output_dim"]:
        if indices.ndim != 3:
            raise ValueError(f"NeuCodec residual FSQ indices must have 3 dimensions; got {indices.shape}.")
        if indices.shape[-1] != self.config.num_quantizers:
            raise ValueError(
                "NeuCodec residual FSQ indices last dimension must match num_quantizers"
                f" ({self.config.num_quantizers}); got {indices.shape[-1]}.",
            )
        codes = self.fsq.indices_to_codes(indices[..., 0])
        return self.project_out(codes)

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "fsq": self.fsq.export_weights(),
            "project_out": self.project_out.export_weights(),
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            fsq=self.fsq.import_weights(require_tree(weights["fsq"])),
            project_out=self.project_out.import_weights(require_tree(weights["project_out"])),
        )
