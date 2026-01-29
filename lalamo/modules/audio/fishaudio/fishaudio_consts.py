"""
Consts and configuration elements that were stored in Fishaudio
codebase as either global variables or magic consts
"""

DEFAULT_FISHAUDIO_RANDOM_SEED: int = 123
DEFAULT_FISH_AUDIO_SAMPLING_TEMPERATURE = 0.8008
DEFAULT_FISH_AUDIO_SAMPLING_TOP_P = 0.8008
DEFAULT_FISH_AUDIO_REPETITION_PENALTY: float = 1.1016
SHORT_LOGITS_SIZE: int = 1024
REPEAT_WINDOW_SIZE: int = 16
FISH_TIKTOKEN_PATTERN = "|".join(  # noqa: FLY002
    [
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
        r"\p{P}",
        r"[^\r\n\p{L}\p{N}]?\p{L}+",
        r"\p{N}",
        r" ?[^\s\p{L}\p{N}]+[\r\n]*",
        r"\s*[\r\n]+",
        r"\s+(\?!\S)",
        r"\s+",
    ],
)
IM_END_TOKEN = "<|im_end|>"


# NOTE: in fish-speech repo this was a YAML config stored right in the code
_default_audio_codec_config = {
    "_target_": "fish_speech.models.dac.modded_dac.DAC",
    "sample_rate": 44100,
    "encoder_dim": 64,
    "encoder_rates": [2, 4, 8, 8],
    "decoder_dim": 1536,
    "decoder_rates": [8, 8, 4, 2],
    "encoder_transformer_layers": [0, 0, 0, 4],
    "decoder_transformer_layers": [4, 0, 0, 0],
    "transformer_general_config": {
        "_target_": "fish_speech.models.dac.modded_dac.ModelArgs",
        "_partial_": True,
        "block_size": 16384,
        "n_local_heads": -1,
        "head_dim": 64,
        "rope_base": 10000,
        "norm_eps": 1e-5,
        "dropout_rate": 0.1,
        "attn_dropout_rate": 0.1,
        "channels_first": True,
    },
    "quantizer": {
        "_target_": "fish_speech.models.dac.rvq.DownsampleResidualVectorQuantize",
        "input_dim": 1024,
        "n_codebooks": 9,
        "codebook_size": 1024,
        "codebook_dim": 8,
        "quantizer_dropout": 0.5,
        "downsample_factor": [2, 2],
        "post_module": {
            "_target_": "fish_speech.models.dac.modded_dac.WindowLimitedTransformer",
            "causal": True,
            "window_size": 128,
            "input_dim": 1024,
            "config": {
                "_target_": "fish_speech.models.dac.modded_dac.ModelArgs",
                "block_size": 4096,
                "n_layer": 8,
                "n_head": 16,
                "dim": 1024,
                "intermediate_size": 3072,
                "n_local_heads": -1,
                "head_dim": 64,
                "rope_base": 10000,
                "norm_eps": 1e-5,
                "dropout_rate": 0.1,
                "attn_dropout_rate": 0.1,
                "channels_first": True,
            },
        },
        "pre_module": {
            "_target_": "fish_speech.models.dac.modded_dac.WindowLimitedTransformer",
            "causal": True,
            "window_size": 128,
            "input_dim": 1024,
            "config": {
                "_target_": "fish_speech.models.dac.modded_dac.ModelArgs",
                "block_size": 4096,
                "n_layer": 8,
                "n_head": 16,
                "dim": 1024,
                "intermediate_size": 3072,
                "n_local_heads": -1,
                "head_dim": 64,
                "rope_base": 10000,
                "norm_eps": 1e-5,
                "dropout_rate": 0.1,
                "attn_dropout_rate": 0.1,
                "channels_first": True,
            },
        },
        "semantic_codebook_size": 4096,
    },
}
