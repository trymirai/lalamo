# Fartsovka

To convert model run

```bash
uv run convert_model.py
```

Usage:

```shell
╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    model_repo      TEXT  [default: None] [required]                                                                                                                       │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --precision             TEXT     [default: None]                                                                                                                            │
│ --output-dir            PATH     [default: None]                                                                                                                            │
│ --context-length        INTEGER  [default: 8192]                                                                                                                            │
│ --help                           Show this message and exit.                                                                                                                │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## Features

- Convert various model formats (HuggingFace, ExecutorTorch) to a unified format
- Support for HuggingFace tokenizers
- Export language models with their tokenizers and message formatting specs
- Various model architectures (Llama, Gemma, Qwen)
- Text generation with different sampling strategies (greedy, temperature, top-p)
- Vectorized batch decoding for parallel text generation

## Currently supported models:

```shell
meta-llama/Llama-3.2-1B-Instruct
meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8
google/gemma-2-2b-it
Qwen/Qwen2.5-1.5B-Instruct
deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

## Output Format

When you convert a model, the output directory will contain:
- `config.json` - Complete model configuration including decoder and tokenizer
- `model.safetensors` - Model weights in safetensors format
- `tokenizer.json` - Tokenizer configuration for the HuggingFace tokenizers library

## Adding Support for New Models

To add support for a new model, write the corresponding `ModelSpec` in `fartsovka.model_import.model_import.py`

Example:

```python
ModelSpec(
    name="Gemma-2-2B-Instruct",
    repo="google/gemma-2-2b-it",
    config_type=HFGemma2Config,
    config_file_name="config.json",
    weights_file_names=(
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ),
    weights_type=WeightsType.SAFETENSORS,
    # Message format type for this model family
    message_format_type=MessageFormatType.GEMMA,
    # Optional: custom format specification if needed
    custom_format_spec=None,
)
```

## Message Formatting

The library supports different message format types for various model families:
- `PLAIN` - Simple text without formatting
- `LLAMA` - Llama-style chat format with system, user, and assistant messages
- `GEMMA` - Gemma-style chat format with turn indicators
- `QWEN` - Qwen-style chat format with message markers
- `CUSTOM` - Custom formats defined by the user

## Text Generation

The library provides text generation capabilities with different decoding strategies:

```python
from fartsovka.language_model import (
    DecodingStrategy, 
    DecodingConfig, 
    decode_text,
    decode_batch
)
import jax

# Create decoding configuration
config = DecodingConfig(
    strategy=DecodingStrategy.TOP_P,  # Can be GREEDY, SAMPLE, or TOP_P
    max_tokens=100,                   # Maximum tokens to generate
    temperature=0.8,                  # Temperature for sampling (higher = more random)
    top_p=0.9,                        # Nucleus sampling parameter (0.0-1.0)
    stop_tokens=[2],                  # Optional: tokens that trigger early stopping
    stop_strings=["</s>"],            # Optional: strings that trigger early stopping
)

# Generate text from a single prompt
output = decode_text(
    model=language_model,
    text="Hello, how are you?",
    config=config,
    key=jax.random.PRNGKey(42)  # Only needed for sampling strategies
)

# Or generate text from multiple prompts in parallel
outputs = decode_batch(
    model=language_model,
    texts=["Hello!", "What's the weather like?", "Tell me a joke."],
    config=config,
    key=jax.random.PRNGKey(42)
)
```