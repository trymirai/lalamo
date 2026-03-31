from __future__ import annotations

import re
from enum import IntEnum

from lalamo.model_import.model_specs.common import ModelSpec
from tests.helpers import unsi

PARAM_UNITS = ("", "K", "M", "B")


class ModelTier(IntEnum):
    CORE = 0
    STANDARD = 1
    EXTRA = 2


class ModelSize(IntEnum):
    SMALL = 0  # < 10B params
    LARGE = 1  # >= 10B params


def num_params(size: str) -> int | None:
    normalized = re.sub(r"([0-9.]+)\s*([A-Z])", r"\1 \2", size.strip())
    try:
        return unsi(normalized, base=1000, units=PARAM_UNITS)
    except (ValueError, IndexError):
        return None


def model_size(spec: ModelSpec) -> ModelSize:
    params = num_params(spec.size)
    if params is None:
        return ModelSize.LARGE
    if params < 10_000_000_000:
        return ModelSize.SMALL
    return ModelSize.LARGE


MODEL_TIERS: tuple[tuple[str, ModelTier], ...] = (
    # Liquid
    ("mlx-community/LFM2-350M-8bit", ModelTier.CORE),
    ("LiquidAI/LFM2-700M", ModelTier.CORE),
    ("LiquidAI/LFM2.5-350M", ModelTier.CORE),
    ("mlx-community/LFM2-700M-4bit", ModelTier.STANDARD),
    ("LiquidAI/LFM2-2.6B", ModelTier.STANDARD),
    ("LiquidAI/LFM2-2.6B-Exp", ModelTier.STANDARD),
    ("mlx-community/LFM2-2.6B-Exp-4bit", ModelTier.STANDARD),
    ("LiquidAI/LFM2.5-1.2B-Instruct", ModelTier.STANDARD),
    ("LiquidAI/LFM2-350M", ModelTier.EXTRA),
    ("mlx-community/LFM2-350M-4bit", ModelTier.EXTRA),
    ("mlx-community/LFM2-700M-8bit", ModelTier.EXTRA),
    ("LiquidAI/LFM2-1.2B", ModelTier.EXTRA),
    ("mlx-community/LFM2-1.2B-4bit", ModelTier.EXTRA),
    ("mlx-community/LFM2-1.2B-8bit", ModelTier.EXTRA),
    ("mlx-community/LFM2-2.6B-Exp-8bit", ModelTier.EXTRA),
    # Llama
    ("meta-llama/Llama-3.2-1B-Instruct", ModelTier.CORE),
    ("mlx-community/Llama-3.2-1B-Instruct-4bit", ModelTier.STANDARD),
    ("meta-llama/Llama-3.2-3B-Instruct", ModelTier.STANDARD),
    ("mlx-community/Llama-3.2-3B-Instruct-8bit", ModelTier.STANDARD),
    ("mlx-community/Llama-3.2-1B-Instruct-8bit", ModelTier.EXTRA),
    ("mlx-community/Llama-3.2-3B-Instruct-4bit", ModelTier.EXTRA),
    ("meta-llama/Llama-3.1-8B-Instruct", ModelTier.EXTRA),
    ("mlx-community/Llama-3.1-8B-Instruct-4bit", ModelTier.EXTRA),
    # LLamba
    ("cartesia-ai/Llamba-1B", ModelTier.CORE),
    ("cartesia-ai/Llamba-1B-4bit-mlx", ModelTier.EXTRA),
    # DeepSeek
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", ModelTier.STANDARD),
    # Gemma
    ("google/gemma-3-1b-it", ModelTier.CORE),
    ("google/gemma-2-2b-it", ModelTier.STANDARD),
    ("mlx-community/gemma-3-1b-it-4bit", ModelTier.STANDARD),
    ("google/gemma-3-4b-it", ModelTier.STANDARD),
    ("mlx-community/gemma-3-27b-it-4bit", ModelTier.STANDARD),
    ("mlx-community/gemma-3-1b-it-8bit", ModelTier.EXTRA),
    ("mlx-community/gemma-3-4b-it-4bit", ModelTier.EXTRA),
    ("mlx-community/gemma-3-4b-it-8bit", ModelTier.EXTRA),
    ("google/gemma-3-27b-it", ModelTier.EXTRA),
    ("mlx-community/gemma-3-27b-it-8bit", ModelTier.EXTRA),
    ("google/functiongemma-270m-it", ModelTier.EXTRA),
    # SmolLM
    ("HuggingFaceTB/SmolLM2-1.7B-Instruct", ModelTier.STANDARD),
    ("HuggingFaceTB/SmolLM3-3B", ModelTier.STANDARD),
    ("mlx-community/SmolLM3-3B-4bit", ModelTier.EXTRA),
    ("mlx-community/SmolLM3-3B-8bit", ModelTier.EXTRA),
    # Mistral
    ("mistral-community/Codestral-22B-v0.1", ModelTier.STANDARD),
    ("mistralai/Devstral-Small-2505", ModelTier.STANDARD),
    # GPT-OSS
    ("openai/gpt-oss-20b", ModelTier.STANDARD),
    # Polaris
    ("POLARIS-Project/Polaris-4B-Preview", ModelTier.EXTRA),
    # Reka
    ("RekaAI/reka-flash-3.1", ModelTier.EXTRA),
    # Nanbeige
    ("Nanbeige/Nanbeige4.1-3B", ModelTier.STANDARD),
    # Essential AI
    ("EssentialAI/rnj-1-instruct", ModelTier.EXTRA),
    # Qwen2.5
    ("Qwen/Qwen2.5-0.5B-Instruct", ModelTier.CORE),
    ("Qwen/Qwen2.5-7B-Instruct", ModelTier.STANDARD),
    ("Qwen/Qwen2.5-1.5B-Instruct", ModelTier.EXTRA),
    ("Qwen/Qwen2.5-3B-Instruct", ModelTier.EXTRA),
    ("Qwen/Qwen2.5-14B-Instruct", ModelTier.EXTRA),
    ("Qwen/Qwen2.5-32B-Instruct", ModelTier.EXTRA),
    # Qwen2.5-Coder
    ("Qwen/Qwen2.5-Coder-0.5B-Instruct", ModelTier.STANDARD),
    ("Qwen/Qwen2.5-Coder-1.5B-Instruct", ModelTier.EXTRA),
    ("Qwen/Qwen2.5-Coder-3B-Instruct", ModelTier.EXTRA),
    ("Qwen/Qwen2.5-Coder-7B-Instruct", ModelTier.EXTRA),
    ("Qwen/Qwen2.5-Coder-14B-Instruct", ModelTier.EXTRA),
    ("Qwen/Qwen2.5-Coder-32B-Instruct", ModelTier.EXTRA),
    # Qwen3
    ("Qwen/Qwen3-0.6B", ModelTier.STANDARD),
    ("Qwen/Qwen3-0.6B-MLX-4bit", ModelTier.STANDARD),
    ("Qwen/Qwen3-1.7B", ModelTier.STANDARD),
    ("Qwen/Qwen3-8B", ModelTier.STANDARD),
    ("Qwen/Qwen3-8B-MLX-4bit", ModelTier.STANDARD),
    ("Qwen/Qwen3-14B", ModelTier.STANDARD),
    ("Qwen/Qwen3-4B", ModelTier.EXTRA),
    ("Qwen/Qwen3-4B-AWQ", ModelTier.EXTRA),
    ("Qwen/Qwen3-4B-MLX-4bit", ModelTier.EXTRA),
    ("Qwen/Qwen3-8B-AWQ", ModelTier.EXTRA),
    ("Qwen/Qwen3-14B-AWQ", ModelTier.EXTRA),
    ("Qwen/Qwen3-32B-AWQ", ModelTier.EXTRA),
    ("Qwen/Qwen3-32B", ModelTier.EXTRA),
    # Qwen3.5
    ("mlx-community/Qwen3.5-0.8B-MLX-4bit", ModelTier.CORE),
    ("Qwen/Qwen3.5-2B", ModelTier.STANDARD),
    ("Qwen/Qwen3.5-0.8B", ModelTier.STANDARD),
    ("mlx-community/Qwen3.5-9B-MLX-4bit", ModelTier.STANDARD),
    ("Qwen/Qwen3.5-27B", ModelTier.STANDARD),
    ("mlx-community/Qwen3.5-27B-8bit", ModelTier.STANDARD),
    ("mlx-community/Qwen3.5-0.8B-MLX-8bit", ModelTier.EXTRA),
    ("mlx-community/Qwen3.5-2B-MLX-4bit", ModelTier.EXTRA),
    ("mlx-community/Qwen3.5-2B-MLX-8bit", ModelTier.EXTRA),
    ("Qwen/Qwen3.5-4B", ModelTier.EXTRA),
    ("mlx-community/Qwen3.5-4B-MLX-4bit", ModelTier.EXTRA),
    ("mlx-community/Qwen3.5-4B-MLX-8bit", ModelTier.EXTRA),
    ("Qwen/Qwen3.5-9B", ModelTier.EXTRA),
    ("mlx-community/Qwen3.5-9B-MLX-8bit", ModelTier.EXTRA),
    ("mlx-community/Qwen3.5-27B-4bit", ModelTier.EXTRA),
    # AWQ
    ("trymirai/Qwen2.5-3B-Instruct-AWQ", ModelTier.EXTRA),
    ("trymirai/Qwen2.5-7B-Instruct-AWQ", ModelTier.STANDARD),
    ("trymirai/Qwen2.5-Coder-3B-Instruct-AWQ", ModelTier.EXTRA),
    ("trymirai/Qwen2.5-Coder-7B-Instruct-AWQ", ModelTier.EXTRA),
    ("trymirai/DeepSeek-R1-Distill-Qwen-1.5B-AWQ", ModelTier.STANDARD),
    ("trymirai/SmolLM2-1.7B-Instruct-AWQ", ModelTier.EXTRA),
    ("trymirai/Llama-3.2-3B-Instruct-AWQ", ModelTier.EXTRA),
    # Qwen3-Next
    ("Qwen/Qwen3-Next-80B-A3B-Instruct", ModelTier.EXTRA),
    # Audio
    ("fishaudio/s1-mini", ModelTier.EXTRA),
    ("nvidia/nemo-nano-codec-22khz-1.78kbps-12.5fps", ModelTier.EXTRA),
    # Other
    ("trymirai/chat-moderation-router", ModelTier.EXTRA),
    ("amd/PARD-Qwen3-0.6B", ModelTier.EXTRA),
)

TIER_BY_REPO: dict[str, ModelTier] = dict(MODEL_TIERS)

COHERENCE_TTS_REPOS: tuple[str, ...] = ("fishaudio/s1-mini",)
