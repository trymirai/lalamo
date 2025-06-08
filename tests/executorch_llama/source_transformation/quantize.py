# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import nn


class WeightOnlyInt8Linear(nn.Module):
    __constants__: ClassVar = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        device: torch.device,  # noqa: ARG002
        in_features: int,
        out_features: int,
        bias: bool = True,  # noqa: ARG002
        dtype: torch.dtype | None = None,  # noqa: ARG002
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight", torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer("scales", torch.ones(out_features, dtype=torch.bfloat16))

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        return F.linear(input, self.weight.to(dtype=input.dtype)) * self.scales  # type: ignore


def linear_forward_8da8w(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    out_features: int,  # noqa: ARG001
    precision: torch.dtype,
) -> torch.Tensor:
    from torchao.quantization.utils import per_token_dynamic_quant

    x = per_token_dynamic_quant(x)
    n_bit = 8
    quant_min = -(2 ** (n_bit - 1))
    quant_max = 2 ** (n_bit - 1) - 1
    w_dq = torch.ops.quantized_decomposed.dequantize_per_channel(  # type: ignore
        weight_int8,
        scales,
        zeros,
        0,
        quant_min,
        quant_max,
        torch.int8,
        out_dtype=precision,
    )
    c = torch.nn.functional.linear(x, w_dq)

    return c


class Int8DynActInt8WeightLinear(nn.Module):
    __constants__: ClassVar = ["in_features", "out_features"]

    in_features: int
    out_features: int
    weight: torch.Tensor

    """
    This module implements a dynamic quantized linear layer with int8 weight.
    Weights are per channel quantized. Parameters of importance
    precision: precision of input and output. e.g. torch.float32 means input
    activation is float32 and output is float32.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | None = None,  # noqa: ARG002
        dtype: torch.dtype | None = None,
        precision: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        assert not bias, "require bias=False"
        self.precision = precision

        if dtype is not None:
            raise ValueError("Please specify 'precision' instead of 'dtype'")

        # currently storing unpacked int8 weights
        self.register_buffer(
            "weight",
            torch.zeros((out_features, in_features), dtype=torch.int8),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (out_features),
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "zeros",
            torch.zeros(
                (out_features),
                dtype=torch.float32,
            ),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        input = input.to(self.precision)  # noqa: A001
        return linear_forward_8da8w(
            input,
            self.weight,
            self.scales,  # type: ignore
            self.zeros,  # type: ignore
            self.out_features,
            self.precision,
        )


def embedding_byte(
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    weight_zero_points: torch.Tensor | None,
    weight_quant_min: int,
    weight_quant_max: int,
    indices: torch.Tensor,
) -> torch.Tensor:
    group_size = weight.size(1) // (weight_scales.size(1) if weight_scales.dim() == 2 else 1)
    weight = torch.ops.quantized_decomposed.dequantize_per_channel_group.default(  # type: ignore
        weight,
        weight_scales,
        weight_zero_points,
        weight_quant_min,
        weight_quant_max,
        weight.dtype,
        group_size,
        weight_scales.dtype,
    )
    return torch.ops.aten.embedding.default(weight, indices)  # type: ignore


class QuantizedGroupEmbedding(nn.Module):
    def __init__(
        self,
        device: torch.device,
        vocab_size: int,
        embedding_dim: int,
        group_size: int | None = None,
        dtype: torch.dtype = torch.half,
        packed: bool = False,
        bitwidth: int = 8,
    ) -> None:
        super().__init__()
        if group_size is None or group_size == 0:
            group_size = embedding_dim
        self.group_size = group_size
        self.dtype = dtype
        self.packed = packed
        self.bitwidth = bitwidth
        if not packed:
            self.register_buffer(
                "weight",
                torch.zeros((vocab_size, embedding_dim), dtype=torch.int8, device=device),
            )
        elif bitwidth == 2:
            self.register_buffer(
                "weight",
                torch.zeros(
                    (vocab_size, embedding_dim // 4),
                    dtype=torch.uint8,
                    device=device,
                ),
            )
        elif bitwidth == 4:
            self.register_buffer(
                "weight",
                torch.zeros(
                    (vocab_size, embedding_dim // 2),
                    dtype=torch.uint8,
                    device=device,
                ),
            )

        groups_per_row = (embedding_dim + group_size - 1) // group_size
        if groups_per_row > 1:
            self.register_buffer(
                "scales",
                torch.ones((vocab_size, groups_per_row), dtype=torch.float16, device=device),
            )
        else:
            self.register_buffer("scales", torch.ones((vocab_size,), dtype=torch.float16, device=device))

    @torch.no_grad()
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if not self.packed:  # 8bit
            return embedding_byte(  # type: ignore
                self.weight,  # type: ignore
                self.scales,  # type: ignore
                None,
                0,
                0,
                indices,
            ).to(self.dtype)
        # packed
        if self.bitwidth == 2:
            return torch.ops.quantized_decomposed.embedding_2bit.dtype(  # type: ignore
                self.weight,
                self.scales,
                None,
                0,
                0,
                indices,
                dtype=self.dtype,
            )

        # Remaining case (always return to make pyre happy)
        assert self.bitwidth == 4
        return torch.ops.quantized_decomposed.embedding_4bit.dtype(  # type: ignore
            self.weight,
            self.scales,
            None,
            0,
            0,
            indices,
            dtype=self.dtype,
        )
