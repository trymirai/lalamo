# @lint-ignore-every LICENSELINT
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Llama 2 is licensed under the LLAMA 2 Community License,
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

# Please refer to README.md in the same folder for more information.

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .rope import (
    RotaryEmbedding,
    precompute_freqs_cis,
)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int | None = None
    vocab_size: int = -1  # defined later by tokenizer
    hidden_dim: int | None = None
    head_dim: int | None = None  # Optional customized head_dim
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: float | None = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048
    moe: bool = False  # True to enable the MoE (Mixture of Experts)
    num_experts: int = 8  # Number of experts
    num_activated_experts: int = 2  # Number of experts to activate
    use_kv_cache: bool = False  # Use key/value cache
    use_sdpa_with_kv_cache_op: bool = False  # Use custom sdpa op that updates kv cache in-place
    # Generate logits for all inputs. When it's True, it would take big memory usage
    # at runtime. Enable it only necessary (e.g., use perplexity tools that requires
    # logits for all input tokens.)
    generate_full_logits: bool = False
    enable_dynamic_shape: bool = False  # export model with dynamic shape support
    # A dictionary mapping from pruned token-id to original token-id
    input_prune_map: dict[int, int] | None = None
    # A dictionary mapping from pruned token-id to original token-id
    output_prune_map: dict[int, int] | None = None
    use_hf_rope: bool = False  # Use HuggingFace's RoPE implementation
    rope_theta: float | None = None  # The official name to override self.rope_freq_base.
    rope_freq_base: float = 10000.0  # The base frequency for RoPE. Keep it for BC.
    use_scaled_rope: bool = False  # Use scaled RoPE, introduced in llama3.1.
    rope_scale_factor: int = 8
    # Additional Model Metadata needed at runtime
    bos_idx: int = 1
    eos_idx: int = 3
    bos_count: int = -1  # i.e., a single EOS is used as BOS
    eos_count: int = 2

    quantization_args: dict | None = None
    lora_args: dict | None = None

    def __post_init__(self) -> None:
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads

        # rope_theta overrides rope_freq_base since it's the official name.
        if self.rope_theta is not None:
            self.rope_freq_base = self.rope_theta

        if self.use_sdpa_with_kv_cache_op:
            assert self.use_kv_cache, "use_sdpa_with_kv_cache_op requires use_kv_cache"

        if self.hidden_dim is None:
            # If hidden_dim is not explicitly set in the ModelArgs,
            # then calculate implicitly based on dim and also multiple of `args.multiple_of`
            multiple_of = self.multiple_of
            hidden_dim = 4 * self.dim
            hidden_dim = int(2 * hidden_dim / 3)
            if self.ffn_dim_multiplier is not None:
                hidden_dim = int(self.ffn_dim_multiplier * hidden_dim)
            self.hidden_dim = find_multiple(hidden_dim, multiple_of)

        if self.head_dim is None:
            self.head_dim = self.dim // self.n_heads


class Rope(torch.nn.Module):
    def __init__(self, params: ModelArgs) -> None:
        super().__init__()
        self.params = params

        freqs_cos, freqs_sin = precompute_freqs_cis(
            self.params.head_dim,  # type: ignore
            self.params.max_seq_len * 2,
            self.params.rope_freq_base,
            use_scaled=self.params.use_scaled_rope,
            scale_factor=self.params.rope_scale_factor,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        self.apply_rotary_emb = RotaryEmbedding()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.apply_rotary_emb(q, k, freqs_cos, freqs_sin)

    def get_freqs(self, input_pos: torch.Tensor | None, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the precomputed frequencies for the given input position and sequence length.

        Args:
            input_pos (torch.Tensor): The input position tensor.
            seq_len (int): The sequence length.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                The precomputed frequencies for the given input position and sequence length.
        """

        assert input_pos is None, "input_pos is unused when use_kv_cache is False"
        freqs_cos = self.freqs_cos[:seq_len]  # type: ignore
        freqs_sin = self.freqs_sin[:seq_len]  # type: ignore
        return freqs_cos, freqs_sin


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int, rope: Rope) -> None:
        super().__init__()
        self.use_kv_cache = args.use_kv_cache
        self.n_heads = args.n_heads
        self.n_kv_heads = self.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert self.n_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = self.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.head_dim
        self.max_batch_size = args.max_batch_size
        self.max_seq_len = args.max_seq_len
        self.dim = args.dim
        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)  # type: ignore
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)  # type: ignore
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)  # type: ignore
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)  # type: ignore

        self.layer_id = layer_id

        self.rope = rope

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        input_pos: torch.Tensor | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        # QKV
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        # We need view_copy elimination
        q = q.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # RoPE relative positional embeddings
        q, k = self.rope.forward(q, k, freqs_cos, freqs_sin)

        q = q.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        output = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        output = self.wo(output)

        return output


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        assert args.hidden_dim is not None
        hidden_dim: int = args.hidden_dim
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs, rope: Rope) -> None:
        super().__init__()
        self.use_kv_cache = args.use_kv_cache
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.head_dim
        self.attention = Attention(args, layer_id, rope)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        input_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin, input_pos)
        return h + self.feed_forward(self.ffn_norm(h))


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs) -> None:
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.rope = Rope(params)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params, self.rope))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        self.use_kv_cache = params.use_kv_cache
        self.generate_full_logits = params.generate_full_logits
        self.max_seq_len = params.max_seq_len
        self.input_prune_map = params.input_prune_map
        self.output_prune_map = params.output_prune_map

    def forward(
        self,
        tokens: torch.LongTensor | None = None,
        input_pos: torch.LongTensor | None = None,  # Scalar tensor indicating size of window of the caches
        h: torch.FloatTensor | None = None,  # embeddings
    ) -> torch.Tensor:
        seqlen = tokens.shape[1]  # type: ignore
        freqs_cos, freqs_sin = self.rope.get_freqs(input_pos, seqlen)

        h = self.tok_embeddings(tokens)

        for layer in self.layers:
            h = layer(
                h,
                freqs_cos,
                freqs_sin,
                input_pos,
            )

        h = self.norm(h)

        logits = self.output(h)

        return logits
