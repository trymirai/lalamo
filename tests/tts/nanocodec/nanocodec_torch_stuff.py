import io
import logging
import math
import tarfile
from abc import abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Optional

import einops
import huggingface_hub
import torch
import torch.nn.functional as F
import yaml
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch._tensor import Tensor
from torch.nn.modules.module import Module
from torch.types import Number

_HAS_HYDRA = True
NEMO_MODEL_ID = "nemo-nano-codec-22khz-1.78kbps-12.5fps"


def try_locate_fish_audio_model_path() -> Path | None:
    fish_audiod_repo_id = f"nvidia/{NEMO_MODEL_ID}"

    repos = huggingface_hub.scan_cache_dir().repos
    try:
        fish_audio_model_info = next(filter(lambda repo: repo.repo_id == fish_audiod_repo_id, repos))

        api = huggingface_hub.HfApi()
        cache_info = api.model_info(fish_audiod_repo_id)
        commit_hash = cache_info.sha
        return fish_audio_model_info.repo_path / "snapshots" / str(commit_hash) / f"{NEMO_MODEL_ID}.nemo"
    except StopIteration:
        return None


def load_nemo_data(path: str | Path) -> tuple[dict, dict]:
    """Load NeMo model weights and config from .nemo file.

    Returns:
        (state_dict, config)
    """
    state_dict = None
    config = None

    with tarfile.open(path, "r") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".ckpt"):
                f = tar.extractfile(member)
                if f is None:
                    raise ValueError("Failed to load model checkpoint")
                state_dict = torch.load(io.BytesIO(f.read()), map_location="cpu")
            elif member.name.endswith(".yaml"):
                f = tar.extractfile(member)
                if f is None:
                    raise ValueError("Failed to load model checkpoint")
                config = yaml.safe_load(f)

    if state_dict is None:
        raise ValueError(f"No .ckpt file found in {path}")
    if config is None:
        raise ValueError(f"No .yaml config found in {path}")

    return state_dict, config


def mask_sequence_tensor(tensor: torch.Tensor, lengths: torch.Tensor):
    """
    For tensors containing sequences, zero out out-of-bound elements given lengths of every element in the batch.

    tensor: tensor of shape (B, L), (B, D, L) or (B, D1, D2, L),
    lengths: LongTensor of shape (B,)
    """
    batch_size, *_, max_lengths = tensor.shape

    if len(tensor.shape) == 2:
        mask = torch.ones(batch_size, max_lengths).cumsum(dim=-1).type_as(lengths)
        mask = mask <= einops.rearrange(lengths, "B -> B 1")
    elif len(tensor.shape) == 3:
        mask = torch.ones(batch_size, 1, max_lengths).cumsum(dim=-1).type_as(lengths)
        mask = mask <= einops.rearrange(lengths, "B -> B 1 1")
    elif len(tensor.shape) == 4:
        mask = torch.ones(batch_size, 1, 1, max_lengths).cumsum(dim=-1).type_as(lengths)
        mask = mask <= einops.rearrange(lengths, "B -> B 1 1 1")
    else:
        raise ValueError("Can only mask tensors of shape B x L, B x D x L and B x D1 x D2 x L")

    return tensor * mask


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


def get_down_sample_padding(kernel_size: int, stride: int) -> int:
    return (kernel_size - stride + 1) // 2


class VectorQuantizerBase(torch.nn.Module):
    @abstractmethod
    def forward(self, inputs: torch.Tensor, input_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def encode(self, inputs: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def decode(self, indices: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        pass


class FiniteScalarQuantizer(VectorQuantizerBase):
    """This quantizer is based on the Finite Scalar Quantization (FSQ) method.
    It quantizes each element of the input vector independently into a number of levels.

    Args:
        num_levels: number of levels for each dimension/element of the input vector
        eps: small regularization constant for scaling

    References:
        Mentzer et al., Finite Scalar Quantization: VQ-VAE Made Simple (https://arxiv.org/abs/2309.15505v1)
    """

    def __init__(self, num_levels: list[int], eps: float = 1e-3):
        super().__init__()

        # index base per dimension of the input vector
        # this is used to convert between per-dimension indices and a codebook token index
        dim_base_index = torch.cumprod(torch.tensor([1] + num_levels[:-1]), dim=0, dtype=torch.int32)
        dim_base_index = rearrange(dim_base_index, "D -> 1 D 1")
        self.register_buffer("dim_base_index", dim_base_index)

        # Register the number of levels for each dimension
        num_levels = torch.tensor(num_levels, dtype=torch.int32)
        num_levels = rearrange(num_levels, "D -> 1 D 1")
        self.register_buffer("num_levels", num_levels)

        # Regularization
        self.eps = eps

    @property
    def codebook_size(self) -> Number:
        """Returns the size of the corresponding codebook."""
        return self.get_buffer("num_levels").prod().item()

    @property
    def dim(self):
        """Returns the dimension of the input vector."""
        return self.get_buffer("num_levels").numel()

    @property
    def codebook_dim(self):
        """Returns the dimension of the input vector.
        Keeping for compatiblitiy with the original RVQ implementation.
        """
        return self.dim

    @property
    def codes(self):
        """Returns the codebooks entries.

        Note that the codebook entries are implicitly defined by the number of levels.
        """
        indices = torch.arange(self.codebook_size)
        # [D, B, T]
        indices = rearrange(indices, "B -> 1 B 1")
        # [B, D, T]
        codes = self.decode(indices=indices, input_len=None)
        # Remove the time dimension
        codes = codes.squeeze(-1)
        return codes

    @property
    def codebook(self):
        """Returns the codebooks entries.
        See self.codes for more details.
        """
        return self.codes

    @staticmethod
    def round(inputs: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        """Round the input tensor to nearest integer
        and use a straight-through estimator for the gradient.
        """
        inputs_rounded = torch.round(inputs)
        return inputs + (inputs_rounded - inputs).detach()

    def compress(self, inputs: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        """Apply compression to the input, to limit to values."""
        output_scale = (self.get_buffer("num_levels") - 1) / 2
        # scale down a bit to avoid rounding issues
        output_scale = output_scale * (1 - self.eps)
        # offset for even number of levels
        output_offset = torch.where(self.get_buffer("num_levels") % 2 == 0, 0.5, 0)
        # shift for even number of levels
        input_shift = (output_offset / output_scale).tan()
        # compressed output
        output = output_scale * (inputs + input_shift).tanh() - output_offset
        return output

    def inputs_to_codes(self, inputs: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        # apply compression
        compressed = self.compress(inputs=inputs, input_len=input_len)
        # apply rounding to nearest integer
        codes = self.round(inputs=compressed, input_len=input_len)
        # normalize to [-1, 1]
        scale = self.get_buffer("num_levels") // 2
        codes = codes / scale
        return codes

    def codes_to_nonnegative(self, codes: torch.Tensor) -> torch.Tensor:
        """Convert values centered arouund zero to nonnegative values."""
        scale = offset = self.get_buffer("num_levels") // 2
        return scale * codes + offset

    def nonnegative_to_codes(self, codes_nonnegative: torch.Tensor) -> torch.Tensor:
        """Convert nonnegative values to values centered arouund zero."""
        scale = offset = self.get_buffer("num_levels") // 2
        return (codes_nonnegative - offset) / scale

    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        """Converts a code vector to a single index."""
        if codes.size(1) != self.dim:
            raise RuntimeError(
                f"Input code dimension {codes.size(1)} not matching the expected dimension {self.dim}, input codes shape {codes.shape}"
            )
        # convert code vectors to nonnegative values
        indices = self.codes_to_nonnegative(codes)
        # convert one nonnegative index per dimension to a single index per code vector
        indices = torch.sum(indices * self.get_buffer("dim_base_index"), dim=1)
        return indices.to(torch.int32)

    def forward(
        self, inputs: torch.Tensor, input_len: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.size(1) != self.dim:
            raise RuntimeError(
                f"Input dimension {inputs.size(1)} not matching the expected dimension {self.dim}, inputs shape {inputs.shape}"
            )

        dequantized = self.inputs_to_codes(inputs=inputs, input_len=input_len)
        indices = self.codes_to_indices(codes=dequantized)

        if input_len is not None:
            # apply masking
            dequantized = mask_sequence_tensor(dequantized, input_len)
            indices = mask_sequence_tensor(indices, input_len)

        # only 1 codebook, but return in [D, B, T] format to match RVQ API
        indices = indices.unsqueeze(0)
        return dequantized, indices

    def encode(self, inputs: torch.Tensor, input_len: torch.Tensor | None = None) -> torch.Tensor:
        """Convert a continuous code vector to a single index."""
        _, indices = self(inputs=inputs, input_len=input_len)
        return indices

    def decode(self, indices: torch.Tensor, input_len: torch.Tensor | None = None) -> torch.Tensor:
        """Convert a single index to a continuous code vector."""
        if indices.size(0) > 1:
            # codebook dimension used for compatibility with RVQ
            raise ValueError(
                f"Expected a single codebook, got {indices.size(0)} codebooks for indices with shape {indices.shape}."
            )

        indices = rearrange(indices, "D B T -> B D T")
        # convert a single index to nonnegative index per-dimension
        codes_nonnegative = (indices // self.get_buffer("dim_base_index")) % self.get_buffer("num_levels")
        # convert nonnegative codes to codes (centered around zero)
        dequantized = self.nonnegative_to_codes(codes_nonnegative)

        if input_len is not None:
            # apply masking
            dequantized = mask_sequence_tensor(dequantized, input_len)
        return dequantized


class GroupFiniteScalarQuantizer(VectorQuantizerBase):
    """Split the input vector into groups and apply FSQ on each group separately.
    This class is for convenience. Since FSQ is applied on each group separately,
    groups can be defined arbitrarily by splitting the input vector. However, this
    class makes it easy to construct several groups with the same quantization num_levels.

    Args:
        num_groups: number of groups to split the input into, each group will be quantized separately using num_codebooks//num_groups codebooks
        codebook_dim: embedding dimension, will be split into num_groups
        **kwargs: parameters of FiniteScalarQuantizer

    References:
        Yang et al, HiFi-Codec: Group-residual Vector quantization for High Fidelity Audio Codec, 2023 (http://arxiv.org/abs/2305.02765).
    """

    def __init__(self, num_groups: int, num_levels_per_group: list[int], **kwargs):
        super().__init__()

        self.num_groups = num_groups
        self.codebook_dim_per_group = len(num_levels_per_group)

        # Initialize FSQ for each group
        self.fsqs = torch.nn.ModuleList(
            [FiniteScalarQuantizer(num_levels=num_levels_per_group, **kwargs) for _ in range(self.num_groups)]
        )

    @property
    def codebook_dim(self):
        """Input vector dimension."""
        return self.codebook_dim_per_group * self.num_groups

    @property
    def codebook_size_per_group(self) -> Tensor | Module:
        """Returns the size of the implicit codebook for each group."""
        return self.fsqs[0].codebook_size

    @property
    def codebook_size(self):
        """Returns the size of the implicit codebook."""
        return self.codebook_size_per_group**self.num_groups

    # @typecheck()
    def forward(self, inputs, input_len):
        """Quantize each group separately, then concatenate the results."""
        inputs_grouped = inputs.chunk(self.num_groups, dim=1)

        dequantized, indices = [], []

        for in_group, fsq_group in zip(inputs_grouped, self.fsqs):
            dequantized_group, indices_group = fsq_group(inputs=in_group, input_len=input_len)
            dequantized.append(dequantized_group)
            indices.append(indices_group)

        # concatenate along the feature dimension
        dequantized = torch.cat(dequantized, dim=1)

        # concatente along the codebook dimension
        indices = torch.cat(indices, dim=0)

        return dequantized, indices

    def encode(self, inputs: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        """Input is split into groups, each group is encoded separately, then the results are concatenated."""
        inputs_grouped = inputs.chunk(self.num_groups, dim=1)
        indices = []

        for in_group, fsq_group in zip(inputs_grouped, self.fsqs):
            indices_group = fsq_group.encode(inputs=in_group, input_len=input_len)
            indices.append(indices_group)

        # concatenate along the codebook dimension
        indices = torch.cat(indices, dim=0)

        return indices

    def decode(self, indices: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        """Input indices are split into groups, each group is decoded separately, then the results are concatenated."""
        indices_grouped = indices.chunk(self.num_groups, dim=0)
        dequantized = []

        for indices_group, fsq_group in zip(indices_grouped, self.fsqs):
            dequantized_group = fsq_group.decode(indices=indices_group, input_len=input_len)
            dequantized.append(dequantized_group)

        # concatenate along the feature dimension
        dequantized = torch.cat(dequantized, dim=1)

        return dequantized

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ) -> None:
        print(f"Loading with prefix: '{prefix}'")
        # print(f"Keys: {[k for k in state_dict.keys() if k.startswith(prefix)]}")

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs=error_msgs
        )


class CausalConv1dNorm(torch.nn.Module):
    """Conv1d with causal padding and normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        pad_mode: str = "zeros",
        extra_pad_mode: str = "constant",
        bias: bool = True,
    ):
        super().__init__()
        self.extra_pad_mode = extra_pad_mode

        # warn user on unusual setup between dilation and stride
        if stride > 1 and dilation > 1:
            print(
                "CausalConv1dNorm has been initialized with stride > 1 and dilation > 1"
                f" (kernel_size={kernel_size} stride={stride}, dilation={dilation})."
            )

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=pad_mode,
        )

        kernel_size = self.conv.kernel_size[0]
        stride = torch.tensor(self.conv.stride[0], dtype=torch.int64)
        dilation = self.conv.dilation[0]

        # Effective kernel size with dilations.
        kernel_size = torch.tensor((kernel_size - 1) * dilation + 1, dtype=torch.int64)

        self.register_buffer("stride", stride, persistent=False)
        self.register_buffer("kernel_size", kernel_size, persistent=False)
        self.register_buffer("padding_total", torch.tensor(kernel_size - stride, dtype=torch.int64), persistent=False)

        # add weight norm
        self.conv = nn.utils.parametrizations.weight_norm(self.conv)

    def remove_weight_norm(self):
        torch.nn.utils.parametrize.remove_parametrizations(self.conv, "weight")

    # Copied from transformers.models.encodec.modeling_encodec.EncodecConv1d._get_extra_padding_for_conv1d
    def _get_extra_padding_for_conv1d(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """See `pad_for_conv1d`."""
        length = hidden_states.shape[-1]
        n_frames = (length - self.kernel_size + self.padding_total) / self.stride + 1
        n_frames = torch.ceil(n_frames).to(torch.int64) - 1
        ideal_length = n_frames * self.stride + self.kernel_size - self.padding_total

        return ideal_length - length

    @staticmethod
    # Copied from transformers.models.encodec.modeling_encodec.EncodecConv1d._pad1d
    def _pad1d(hidden_states: torch.Tensor, paddings: tuple[int, int], mode: str = "zero", value: float = 0.0):
        """Tiny wrapper around torch.nn.functional.pad, just to allow for reflect padding on small input.
        If this is the case, we insert extra 0 padding to the right before the reflection happens.
        """
        length = hidden_states.shape[-1]
        padding_left, padding_right = paddings
        if not mode == "reflect":
            return nn.functional.pad(hidden_states, paddings, mode, value)

        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            hidden_states = nn.functional.pad(hidden_states, (0, extra_pad))
        padded = nn.functional.pad(hidden_states, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]

    def forward(self, inputs, input_len):
        extra_padding = self._get_extra_padding_for_conv1d(inputs)

        # Left padding for causal
        hidden_states = self._pad1d(inputs, (self.padding_total, extra_padding), mode=self.extra_pad_mode)
        hidden_states = self.conv(hidden_states)

        # mask output
        hidden_states = mask_sequence_tensor(hidden_states, input_len)

        return hidden_states


@torch.jit.script
def snake(x: torch.Tensor, alpha: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    equation for snake activation function: x + (alpha + eps)^-1 * sin(alpha * x)^2
    """
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + eps).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake(nn.Module):
    """
    Snake activation function introduced in 'https://arxiv.org/abs/2006.08195'
    """

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return snake(x, self.alpha)


class HalfSnake(nn.Module):
    """
    Activation which applies snake to the first half of input elements and leaky relu to the second half.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.snake_channels = channels // 2
        self.snake_act = Snake(self.snake_channels)
        self.lrelu = torch.nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        snake_out = self.snake_act(x[:, : self.snake_channels, :])
        lrelu_out = self.lrelu(x[:, self.snake_channels :, :])
        out = torch.cat([snake_out, lrelu_out], dim=1)
        return out


class CodecActivation(nn.Module):
    """
    Choose between activation based on the input parameter.

    Args:
        activation: Name of activation to use. Valid options are "elu" (default), "lrelu", and "snake".
        channels: Input dimension.
    """

    def __init__(self, activation: str = "elu", channels: int = 1):
        super().__init__()
        activation = activation.lower()
        if activation == "elu":
            self.activation = nn.ELU()
        elif activation == "lrelu":
            self.activation = torch.nn.LeakyReLU()
        elif activation == "snake":
            self.activation = Snake(channels)
        elif activation == "half_snake":
            self.activation = HalfSnake(channels)
        else:
            raise ValueError(f"Unknown activation {activation}")

    def forward(self, x):
        return self.activation(x)


class ClampActivation(nn.Module):
    def __init__(self, min_value: float = -1.0, max_value: float = 1.0):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.clamp(input, min=self.min_value, max=self.max_value)


class CausalConvTranspose1dNorm(torch.nn.Module):
    """ConvTranspose1d causal padding and normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = None,
        trim_right_ratio: int = 1,
        bias=True,
    ):
        super().__init__()

        self.trim_right_ratio = trim_right_ratio

        # if groups are None, create a group for each out channel as done in Mini Codec
        groups = out_channels if groups is None else groups

        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, groups=groups, bias=bias)

        kernel_size = self.conv.kernel_size[0]
        stride = self.conv.stride[0]
        padding_total = kernel_size - stride

        # Trim the padding on the right according to the specified ratio
        # if trim_right_ratio = 1.0, trim everything from right
        self.padding_right = math.ceil(padding_total * self.trim_right_ratio)
        self.padding_left = padding_total - self.padding_right

        # add weight norm
        self.conv = nn.utils.parametrizations.weight_norm(self.conv)

    def apply_weight_norm(self):
        weight_norm = nn.utils.parametrizations.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        weight_norm(self.conv)

    def remove_weight_norm(self):
        torch.nn.utils.parametrize.remove_parametrizations(self.conv, "weight")

    def forward(self, inputs, input_len):
        hidden_states = self.conv(inputs)

        # unpad
        end = hidden_states.shape[-1] - self.padding_right
        hidden_states = hidden_states[..., self.padding_left : end]
        # mask
        hidden_states = mask_sequence_tensor(hidden_states, input_len)
        return hidden_states


class Conv1dNorm(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        padding: Optional[int] = None,
        pad_mode: str = "reflect",
    ):
        super().__init__()
        if not padding:
            padding = get_padding(kernel_size=kernel_size, dilation=dilation)
        conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            padding_mode=pad_mode,
        )
        self.conv = nn.utils.parametrizations.weight_norm(conv)

    def remove_weight_norm(self):
        torch.nn.utils.parametrize.remove_parametrizations(self.conv, "weight")

    # @typecheck()
    def forward(self, inputs, input_len):
        out = self.conv(inputs)
        out = mask_sequence_tensor(out, input_len)
        return out


class ResidualBlock(torch.nn.Module):
    """
    The residual block structure defined by the HiFi-GAN V1 and V2 configurations.

    Args:
        channels: Input dimension.
        filters: Number of channels in the residual convolutions.
        kernel_size: Kernel size of the residual convolutions.
        dilation: Dilation of the residual convolutions.
        dropout_rate: Dropout to apply to residuals.
        activation: Activation to apply in between residual convolutions.
    """

    def __init__(
        self,
        channels: int,
        filters: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout_rate: float = 0.0,
        activation: str = "lrelu",
        is_causal: bool = False,
        pad_mode: str = "reflect",
    ):
        super(ResidualBlock, self).__init__()

        self.input_activation = CodecActivation(activation=activation, channels=channels)
        self.skip_activation = CodecActivation(activation=activation, channels=filters)
        self.dropout = torch.nn.Dropout(dropout_rate)
        if not is_causal:
            self.input_conv = Conv1dNorm(
                in_channels=channels,
                out_channels=filters,
                kernel_size=kernel_size,
                dilation=dilation,
                pad_mode=pad_mode,
            )
            self.skip_conv = Conv1dNorm(
                in_channels=filters, out_channels=channels, kernel_size=kernel_size, pad_mode=pad_mode
            )
        else:
            self.input_conv = CausalConv1dNorm(
                in_channels=channels,
                out_channels=filters,
                kernel_size=kernel_size,
                dilation=dilation,
                pad_mode=pad_mode,
            )
            self.skip_conv = CausalConv1dNorm(
                in_channels=filters, out_channels=channels, kernel_size=kernel_size, pad_mode=pad_mode
            )

    def remove_weight_norm(self):
        self.input_conv.remove_weight_norm()
        self.skip_conv.remove_weight_norm()

    def forward(self, inputs, input_len):
        conv_input = self.input_activation(inputs)
        skip_input = self.input_conv(inputs=conv_input, input_len=input_len)
        skip_input = self.skip_activation(skip_input)
        res = self.skip_conv(inputs=skip_input, input_len=input_len)
        res = self.dropout(res)
        out = inputs + res
        return out


class HiFiGANResBlock(torch.nn.Module):
    """
    Residual block wrapper for HiFi-GAN which creates a block for multiple dilations.

    Args:
        channels: Input dimension.
        kernel_size: Kernel size of the residual blocks.
        dilations: List of dilations. One residual block will be created for each dilation in the list.
        activation: Activation for the residual blocks.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilations: Iterable[int],
        activation: str,
        is_causal: bool = False,
        pad_mode: str = "reflect",
    ):
        super().__init__()

        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    channels=channels,
                    filters=channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    activation=activation,
                    is_causal=is_causal,
                    pad_mode=pad_mode,
                )
                for dilation in dilations
            ]
        )

    def remove_weight_norm(self):
        for res_block in self.res_blocks:
            res_block.remove_weight_norm()

    def forward(self, inputs, input_len):
        out = inputs
        for res_block in self.res_blocks:
            out = res_block(inputs=out, input_len=input_len)
        return out


class HiFiGANResLayer(torch.nn.Module):
    """
    Residual block wrapper for HiFi-GAN which creates a block for multiple kernel sizes and dilations.
    One residual block is created for each combination of kernel size and dilation.

    Args:
        channels: Input dimension.
        kernel_sizes: List of kernel sizes.
        dilations: List of dilations.
        activation: Activation for the residual layers.

    """

    def __init__(
        self,
        channels: int,
        kernel_sizes: Iterable[int],
        dilations: Iterable[int],
        activation: str,
        is_causal: bool = False,
        pad_mode: str = "reflect",
    ):
        super().__init__()

        self.res_blocks = nn.ModuleList(
            [
                HiFiGANResBlock(
                    channels=channels,
                    kernel_size=kernel_size,
                    dilations=dilations,
                    activation=activation,
                    is_causal=is_causal,
                    pad_mode=pad_mode,
                )
                for kernel_size in kernel_sizes
            ]
        )

    def remove_weight_norm(self):
        for res_block in self.res_blocks:
            res_block.remove_weight_norm()

    def forward(self, inputs, input_len):
        residuals = [res_block(inputs=inputs, input_len=input_len) for res_block in self.res_blocks]
        out = sum(residuals) / len(residuals)
        return out

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ) -> None:
        print(f"Loading with prefix: '{prefix}'")
        # print(f"Keys: {[k for k in state_dict.keys() if k.startswith(prefix)]}")

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs=error_msgs
        )


class CausalHiFiGANDecoder(torch.nn.Module):
    """
    Codec decoder using the HiFi-GAN generator architecture with Causal Convolutions.

    Args:
        input_dim: Input dimension.
        up_sample_rates: Rate to upsample for each decoder block. The product of the upsample rates should be the same
            as the overall downsample rate for your encoder. For example, a symmetric encoder/decoder can be created
            with encoder downsample rates [2, 2, 8, 8] and decoder upsample rates [8, 8, 2, 2].
        base_channels: Number of filters in the first convolution. The number of channels will be cut in
            half after each upsample layer.
        in_kernel_size: Kernel size of the input convolution.
        out_kernel_size: Kernel size of the output convolution.
        resblock_kernel_sizes: List of kernel sizes to use in each residual block.
        resblock_dilation_sizes: List of dilations to use in each residual block.
        activation: Activation to use in residual and upsample layers, defaults to leaky relu.
        output_activation: Activation to apply to output. To produce a valid audio signal, it should output values in
         the range [-1.0, 1.0]. Supports "tanh" and "clamp".
    """

    def __init__(
        self,
        input_dim: int,
        up_sample_rates: Iterable[int] = (8, 8, 2, 2),
        base_channels: int = 512,
        in_kernel_size: int = 7,
        out_kernel_size: int = 3,
        resblock_kernel_sizes: Iterable[int] = (3, 7, 11),
        resblock_dilation_sizes: Iterable[int] = (1, 3, 5),
        activation: str = "lrelu",
        output_activation: str = "tanh",
        pad_mode: str = "zeros",
        n_groups_equal_to_out_channels: bool = True,
        **args,
    ):
        assert in_kernel_size > 0
        assert out_kernel_size > 0

        super().__init__()

        self.up_sample_rates = up_sample_rates

        self.pre_conv = CausalConv1dNorm(
            in_channels=input_dim, out_channels=base_channels, kernel_size=in_kernel_size, pad_mode=pad_mode
        )

        in_channels = base_channels
        self.activations = nn.ModuleList([])
        self.up_sample_conv_layers = nn.ModuleList([])
        self.res_layers = nn.ModuleList([])
        for i, up_sample_rate in enumerate(self.up_sample_rates):
            out_channels = in_channels // 2
            kernel_size = 2 * up_sample_rate

            act = CodecActivation(activation, channels=in_channels)
            self.activations.append(act)

            up_sample_conv = CausalConvTranspose1dNorm(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=up_sample_rate,
                groups=out_channels if n_groups_equal_to_out_channels else 1,
            )
            in_channels = out_channels
            self.up_sample_conv_layers.append(up_sample_conv)

            res_layer = HiFiGANResLayer(
                channels=in_channels,
                kernel_sizes=resblock_kernel_sizes,
                dilations=resblock_dilation_sizes,
                activation=activation,
                is_causal=True,
                pad_mode=pad_mode,
            )
            self.res_layers.append(res_layer)

        self.post_activation = CodecActivation(activation, channels=in_channels)
        self.post_conv = CausalConv1dNorm(
            in_channels=in_channels, out_channels=1, kernel_size=out_kernel_size, pad_mode=pad_mode
        )
        if output_activation == "tanh":
            self.out_activation = nn.Tanh()
        elif output_activation == "clamp":
            self.out_activation = ClampActivation()
        else:
            raise ValueError(f"Invalid audio output activation {output_activation}")

    def remove_weight_norm(self):
        self.pre_conv.remove_weight_norm()
        for up_sample_conv in self.up_sample_conv_layers:
            up_sample_conv.remove_weight_norm()
        for res_layer in self.res_layers:
            res_layer.remove_weight_norm()

    def forward(self, inputs, input_len):
        audio_len = input_len
        # [B, C, T_encoded]
        out = self.pre_conv(inputs=inputs, input_len=audio_len)
        for act, res_layer, up_sample_conv, up_sample_rate in zip(
            self.activations, self.res_layers, self.up_sample_conv_layers, self.up_sample_rates
        ):
            audio_len = audio_len * up_sample_rate
            out = act(out)
            # [B, C / 2, T * up_sample_rate]
            out = up_sample_conv(inputs=out, input_len=audio_len)
            out = res_layer(inputs=out, input_len=audio_len)

        out = self.post_activation(out)
        # [B, 1, T_audio]
        out = self.post_conv(inputs=out, input_len=audio_len)
        audio = self.out_activation(out)
        audio = rearrange(audio, "B 1 T -> B T")
        return audio, audio_len

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ) -> None:
        print(f"Loading with prefix: '{prefix}'")
        # print(f"Keys: {[k for k in state_dict.keys() if k.startswith(prefix)]}")

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs=error_msgs
        )


def convert_model_config_to_dict_config(cfg: Mapping[Any, Any]) -> "DictConfig":
    """
    Converts its input into a standard DictConfig.
    Possible input values are:
    -   DictConfig
    -   A dataclass which is a subclass of NemoConfig

    Args:
        cfg: A dict-like object.

    Returns:
        The equivalent DictConfig
    """
    if not _HAS_HYDRA:
        logging.error("This function requires Hydra/Omegaconf and it was not installed.")
        exit(1)
    if not isinstance(cfg, (OmegaConf, DictConfig)) and is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    if not isinstance(cfg, DictConfig):
        raise ValueError(f"cfg constructor argument must be of type DictConfig/dict but got {type(cfg)} instead.")

    config = OmegaConf.to_container(cfg, resolve=True)
    config = OmegaConf.create(config)
    assert isinstance(config, DictConfig)
    return config


class HiFiGANEncoder(torch.nn.Module):
    """
    Audio encoder created by inverting the HiFi-GAN decoder.

    Args:
        encoded_dim: Dimension of encoder output.
        down_sample_rates: Rate to upsample for each decoder block. The product of the downsample rates will
            determine the output token rate. For example 2 * 2 * 8 * 8 = 256 samples per token.
        base_channels: Number of filters in the first convolution. The number of channels will be doubled after each
            downsample layer.
        in_kernel_size: Kernel size of the input convolution.
        out_kernel_size: Kernel size of the output convolution.
        resblock_kernel_sizes: List of kernel sizes to use in each residual block.
        resblock_dilation_sizes: List of dilations to use in each residual block.
        activation: Activation to use in residual and downsample layers, defaults to leaky relu.
    """

    def __init__(
        self,
        encoded_dim: int,
        down_sample_rates: Iterable[int] = (2, 2, 8, 8),
        base_channels: int = 32,
        in_kernel_size: int = 7,
        out_kernel_size: int = 7,
        resblock_kernel_sizes: Iterable[int] = (3, 7, 11),
        resblock_dilation_sizes: Iterable[int] = (1, 3, 5),
        activation: str = "lrelu",
        pad_mode: str = "reflect",
        **args,
    ):
        assert in_kernel_size > 0
        assert out_kernel_size > 0

        super().__init__()

        self.down_sample_rates = down_sample_rates
        self.pre_conv = Conv1dNorm(
            in_channels=1, out_channels=base_channels, kernel_size=in_kernel_size, pad_mode=pad_mode
        )

        in_channels = base_channels
        self.activations = nn.ModuleList([])
        self.down_sample_conv_layers = nn.ModuleList([])
        self.res_layers = nn.ModuleList([])
        for i, down_sample_rate in enumerate(self.down_sample_rates):
            res_layer = HiFiGANResLayer(
                channels=in_channels,
                kernel_sizes=resblock_kernel_sizes,
                dilations=resblock_dilation_sizes,
                activation=activation,
                pad_mode=pad_mode,
            )
            self.res_layers.append(res_layer)

            act = CodecActivation(activation, channels=in_channels)
            self.activations.append(act)

            out_channels = 2 * in_channels
            kernel_size = 2 * down_sample_rate

            padding = get_down_sample_padding(kernel_size=kernel_size, stride=down_sample_rate)
            down_sample_conv = Conv1dNorm(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=down_sample_rate,
                padding=padding,
                pad_mode=pad_mode,
            )
            in_channels = out_channels
            self.down_sample_conv_layers.append(down_sample_conv)

        self.post_activation = CodecActivation(activation, channels=in_channels)
        self.post_conv = Conv1dNorm(
            in_channels=in_channels, out_channels=encoded_dim, kernel_size=out_kernel_size, pad_mode=pad_mode
        )

    def remove_weight_norm(self):
        self.pre_conv.remove_weight_norm()
        self.post_conv.remove_weight_norm()
        for res_layer in self.res_layers:
            res_layer.remove_weight_norm()
        for down_sample_conv in self.down_sample_conv_layers:
            down_sample_conv.remove_weight_norm()

    def forward(self, audio, audio_len):
        encoded_len = audio_len
        audio = rearrange(audio, "B T -> B 1 T")
        # [B, C, T_audio]
        out = self.pre_conv(inputs=audio, input_len=encoded_len)
        for act, res_layer, down_sample_conv, down_sample_rate in zip(
            self.activations, self.res_layers, self.down_sample_conv_layers, self.down_sample_rates
        ):
            # [B, C, T]
            out = res_layer(inputs=out, input_len=encoded_len)
            out = act(out)

            encoded_len = encoded_len // down_sample_rate
            # [B, 2 * C, T / down_sample_rate]
            out = down_sample_conv(inputs=out, input_len=encoded_len)

        out = self.post_activation(out)
        # [B, encoded_dim, T_encoded]
        encoded = self.post_conv(inputs=out, input_len=encoded_len)
        return encoded, encoded_len


class AudioCodecModel(torch.nn.Module):
    def __init__(self, cfg: DictConfig, trainer: Any = None):
        super().__init__()
        # Convert to Hydra 1.0 compatible DictConfig
        cfg = convert_model_config_to_dict_config(cfg)
        # cfg = model_utils.maybe_update_config_version(cfg)
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_devices

        # super().__init__(cfg=cfg, trainer=trainer)
        self._cfg = cfg

        # Expected sample rate for the input audio
        self.sample_rate = cfg.sample_rate

        # Number of samples in each audio frame that is encoded
        self.samples_per_frame = cfg.samples_per_frame

        # Discriminator updates
        self.disc_updates_per_period = cfg.get("disc_updates_per_period", 1)
        self.disc_update_period = cfg.get("disc_update_period", 1)
        if self.disc_updates_per_period > self.disc_update_period:
            raise ValueError(
                f"Number of discriminator updates ({self.disc_updates_per_period}) per period must be less or equal to the configured period ({self.disc_update_period})"
            )

        if "vector_quantizer" in cfg:
            self.vector_quantizer = GroupFiniteScalarQuantizer(
                cfg.vector_quantizer.num_groups, cfg.vector_quantizer.num_levels_per_group
            )
        else:
            logging.warning("Vector quantizer will not be used.")
            self.vector_quantizer = None

        self.audio_decoder = CausalHiFiGANDecoder(**cfg.audio_decoder)
        self.audio_encoder = HiFiGANEncoder(**cfg.audio_encoder)

    def state_dict(self, destination=None, prefix="", keep_vars=False) -> dict[Any, Any]:
        if hasattr(self, "_no_state_dict") and self._no_state_dict:
            return {}
        # Don't save the speaker verification and codec model in the state dict
        state_dict = super().state_dict()
        for key in list(state_dict.keys()):
            if "speaker_encoder." in key:
                del state_dict[key]
            if ".slm_model.ssl_model." in key:
                del state_dict[key]
        return state_dict

    def load_state_dict(self, state_dict, strict=False) -> None:
        # Override to load all the keys except .speaker_encoder. and WavLM model
        for key in list(state_dict.keys()):
            if "speaker_encoder." in key:
                del state_dict[key]
            if ".slm_model.ssl_model." in key:
                del state_dict[key]

        super().load_state_dict(state_dict, strict=strict)

    def decode_audio(self, inputs: torch.Tensor, input_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply decoder on the input. Note that the input is a non-quantized encoder output or a dequantized representation.

        Args:
            inputs: encoded signal
            input_len: valid length for each example in the batch

        Returns:
            Decoded output `audio` in the time domain and its length in number of samples `audio_len`.
            Note that `audio_len` will be a multiple of `self.samples_per_frame`.
        """
        audio, audio_len = self.audio_decoder(inputs=inputs, input_len=input_len)
        return audio, audio_len

    def quantize(self, encoded: torch.Tensor, encoded_len: torch.Tensor) -> torch.Tensor:
        """Quantize the continuous encoded representation into a discrete
        representation for each frame.

        Args:
            encoded: encoded signal representation
            encoded_len: valid length of the encoded representation in frames

        Returns:
            A tensor of tokens for each codebook for each frame.
        """
        if not self.vector_quantizer:
            raise ValueError("Cannot quantize without quantizer")

        # vector quantizer is returning [C, B, T], where C is the number of codebooks
        tokens = self.vector_quantizer.encode(inputs=encoded, input_len=encoded_len)
        # use batch first for the output
        tokens = rearrange(tokens, "C B T -> B C T")
        return tokens

    def dequantize(self, tokens: torch.Tensor, tokens_len: torch.Tensor) -> torch.Tensor:
        """Convert the discrete tokens into a continuous encoded representation.

        Args:
            tokens: discrete tokens for each codebook for each time frame
            tokens_len: valid length of each example in the batch

        Returns:
            Continuous encoded representation of the discrete input representation.
        """
        if not self.vector_quantizer:
            raise ValueError("Cannot dequantize without quantizer")

        # vector quantizer is using [C, B, T], where C is the number of codebooks
        tokens = rearrange(tokens, "B C T -> C B T")
        dequantized = self.vector_quantizer.decode(indices=tokens, input_len=tokens_len)
        return dequantized

    def decode(self, tokens: torch.Tensor, tokens_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert discrete tokens into a continuous time-domain signal.

        Args:
            tokens: discrete tokens for each codebook for each time frame, shape `(batch, number of codebooks, number of frames)`
            tokens_len: valid lengths, shape `(batch,)`

        Returns:
            Decoded output `audio` in the time domain and its length in number of samples `audio_len`.
            Note that `audio_len` will be a multiple of `self.samples_per_frame`.
        """
        # Convert a discrete representation to a dequantized vector for each frame
        dequantized = self.dequantize(tokens=tokens, tokens_len=tokens_len)
        # Apply decoder to obtain time-domain audio for each frame
        audio, audio_len = self.decode_audio(inputs=dequantized, input_len=tokens_len)

        return audio, audio_len

    def encode_audio(self, audio: torch.Tensor, audio_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply encoder on the input audio signal. Input will be padded with zeros so
        the last frame has full `self.samples_per_frame` samples.

        Args:
            audio: input time-domain signal
            audio_len: valid length for each example in the batch

        Returns:
            Encoder output `encoded` and its length in number of frames `encoded_len`
        """
        audio, audio_len = self.pad_audio(audio, audio_len)
        encoded, encoded_len = self.audio_encoder(audio=audio, audio_len=audio_len)
        return encoded, encoded_len

    def encode(self, audio: torch.Tensor, audio_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert input time-domain audio signal into a discrete representation (tokens).

        Args:
            audio: input time-domain signal, shape `(batch, number of samples)`
            audio_len: valid length for each example in the batch, shape `(batch size,)`

        Returns:
            Tokens for each codebook for each frame, shape `(batch, number of codebooks, number of frames)`,
            and the corresponding valid lengths, shape `(batch,)`
        """
        # Apply encoder to obtain a continuous vector for each frame
        encoded, encoded_len = self.encode_audio(audio=audio, audio_len=audio_len)
        # Apply quantizer to obtain discrete representation per frame
        tokens = self.quantize(encoded=encoded, encoded_len=encoded_len)
        return tokens, encoded_len

    def pad_audio(self, audio, audio_len):
        """Zero pad the end of the audio so that we do not have a partial end frame.
        The output will be zero-padded to have an integer number of frames of
        length `self.samples_per_frame`.

        Args:
            audio: input time-domain signal
            audio_len: valid length for each example in the batch

        Returns:
            Padded time-domain signal `padded_audio` and its length `padded_len`.
        """
        padded_len = self.samples_per_frame * torch.ceil(audio_len / self.samples_per_frame).int()
        max_len = padded_len.max().item()
        num_padding = max_len - audio.shape[1]
        padded_audio = F.pad(audio, (0, num_padding))
        return padded_audio, padded_len


def validate_model_weights(
    model: AudioCodecModel,
    state_dict: dict,
    verbose: bool = True,
) -> dict:
    """Validate that model weights are loaded correctly from NeMo state_dict.

    This utility helps diagnose weight loading issues by:
    1. Checking key format compatibility
    2. Verifying weights were actually loaded (not random)
    3. Reporting any missing or unexpected keys

    Args:
        model: AudioCodecModel instance (after load_state_dict)
        state_dict: Original state_dict from load_nemo_data()
        verbose: If True, print detailed diagnostics

    Returns:
        Dict with validation results:
        - "keys_match": bool - True if all relevant keys match
        - "weights_loaded": int - Number of weights that changed from init
        - "missing_keys": list - Keys expected but not in state_dict
        - "extra_keys": list - Keys in state_dict but not in model
        - "encoder_output_range": tuple - (min, max) of encoder output
    """
    results: dict = {
        "keys_match": True,
        "weights_loaded": 0,
        "missing_keys": [],
        "extra_keys": [],
        "encoder_output_range": None,
    }

    # Filter to relevant prefixes
    relevant_prefixes = ("audio_encoder.", "audio_decoder.", "vector_quantizer.")

    nemo_keys = {k for k in state_dict if any(k.startswith(p) for p in relevant_prefixes)}
    model_keys = {k for k in model.state_dict() if any(k.startswith(p) for p in relevant_prefixes)}

    results["missing_keys"] = sorted(model_keys - nemo_keys)
    results["extra_keys"] = sorted(nemo_keys - model_keys)
    results["keys_match"] = len(results["missing_keys"]) == 0 and len(results["extra_keys"]) == 0

    if verbose:
        print(f"Key validation: {'PASS' if results['keys_match'] else 'FAIL'}")
        print(f"  Model expects {len(model_keys)} keys, NeMo has {len(nemo_keys)} keys")
        if results["missing_keys"]:
            print(f"  Missing from NeMo: {results['missing_keys'][:5]}...")
        if results["extra_keys"]:
            print(f"  Extra in NeMo: {results['extra_keys'][:5]}...")

    # Check if weights were loaded by comparing to fresh model
    fresh_model = AudioCodecModel(model._cfg)
    fresh_state = fresh_model.state_dict()
    loaded_state = model.state_dict()

    for key in model_keys:
        if key in fresh_state and key in loaded_state and not torch.equal(fresh_state[key], loaded_state[key]):
            results["weights_loaded"] += 1

    if verbose:
        print(f"Weights loaded: {results['weights_loaded']} / {len(model_keys)} changed from init")

    # Test encoder output range with a simple signal
    model.eval()
    sample_rate = model.sample_rate
    t = torch.linspace(0, 0.5, int(sample_rate * 0.5))
    audio = torch.sin(2 * 3.14159 * 440 * t).unsqueeze(0)
    audio_len = torch.tensor([audio.shape[1]])

    with torch.no_grad():
        encoded, _ = model.encode_audio(audio=audio, audio_len=audio_len)
        results["encoder_output_range"] = (encoded.min().item(), encoded.max().item())

    if verbose:
        enc_min, enc_max = results["encoder_output_range"]
        print(f"Encoder output range: [{enc_min:.2f}, {enc_max:.2f}]")
        if abs(enc_min) > 2 or abs(enc_max) > 2:
            print("  Note: Large encoder output is EXPECTED - FSQ uses tanh compression")

    return results
