from dataclasses import dataclass
from typing import ClassVar, Literal, Optional, List, Tuple, Union
from pathlib import Path

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Bool

from fartsovka.common import DType
from fartsovka.model_import.loaders import load_huggingface
from fartsovka.modules import (
    Activation,
    AttentionConfig,
    Decoder,
    DecoderConfig,
    DecoderLayerConfig,
    FullPrecisionLinearConfig,
    LlamaRoPEConfig,
    MLPConfig,
    RMSNormConfig,
    TiedEmbeddingConfig,
    UnscaledRoPEConfig,
    VisionConfig,
    VisionTransformer,
    PatchEmbeddingConfig,
    VisionLayerConfig,
    PatchMergerConfig,
    VisionRoPEConfig,
    VisionSdpaAttentionConfig
)

from .common import ForeignConfig

__all__ = ["HFGemma2Config", "HFLlamaConfig", "HFQwen2Config", "HFQwen25VLConfig"]


@dataclass
class HuggingFaceConfig(ForeignConfig):
    _add_one_to_rms_norm_weights: ClassVar[bool] = False

    torch_dtype: Literal["bfloat16", "float16", "float32"]

    @property
    def default_precision(self) -> DType:
        return jnp.dtype(self.torch_dtype)

    @classmethod
    def _load_weights(
        cls,
        model: Decoder,
        weights_dict: dict[str, Array],
    ) -> Decoder:
        return load_huggingface(model, weights_dict, cls._add_one_to_rms_norm_weights)


@dataclass
class HFRopeScalingConfig:
    factor: float
    high_freq_factor: float
    low_freq_factor: float
    original_max_position_embeddings: int
    rope_type: Literal["llama3"]


@dataclass
class HFLlamaConfig(HuggingFaceConfig):
    architectures: list[Literal["LlamaForCausalLM"]]
    attention_bias: bool
    attention_dropout: float
    bos_token_id: int | list[int]
    eos_token_id: int | list[int]
    hidden_act: Literal["silu"]
    hidden_size: int
    initializer_range: float
    intermediate_size: int
    max_position_embeddings: int
    mlp_bias: bool
    model_type: Literal["llama"]
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    pretraining_tp: int
    rms_norm_eps: float
    rope_scaling: HFRopeScalingConfig | None
    rope_theta: float
    tie_word_embeddings: bool
    transformers_version: str
    use_cache: bool
    vocab_size: int
    head_dim: int | None = None

    def to_decoder_config(
        self,
        context_length: int,
        activation_precision: DType,
        accumulation_precision: DType,
    ) -> DecoderConfig:
        embedding_config = TiedEmbeddingConfig(
            input_scale=None,
            logits_soft_cap=None,
            precision=activation_precision,
        )
        if self.rope_scaling is None:
            rope_config = UnscaledRoPEConfig(
                precision=activation_precision,
                base=self.rope_theta,
                max_sequence_length=self.max_position_embeddings,
            )
        else:
            rope_config = LlamaRoPEConfig(
                precision=activation_precision,
                base=self.rope_theta,
                max_sequence_length=self.max_position_embeddings,
                scaling_factor=self.rope_scaling.factor,
                original_context_length=self.rope_scaling.original_max_position_embeddings,
                low_frequency_factor=self.rope_scaling.low_freq_factor,
                high_frequency_factor=self.rope_scaling.high_freq_factor,
            )
        rmsnorm_config = RMSNormConfig(
            scale_precision=activation_precision,
            accumulation_precision=accumulation_precision,
            epsilon=self.rms_norm_eps,
        )
        linear_config = FullPrecisionLinearConfig(
            precision=activation_precision,
        )
        attention_config = AttentionConfig(
            qkv_projection_config=linear_config,
            out_projection_config=linear_config,
            logit_soft_cap=None,
            has_qkv_biases=self.attention_bias,
            has_out_biases=False,
        )
        mlp_config = MLPConfig(
            linear_config=linear_config,
            activation=Activation.SILU,
        )
        decoder_layer_config = DecoderLayerConfig(
            pre_attention_norm_config=rmsnorm_config,
            attention_config=attention_config,
            post_attention_norm_config=None,
            pre_mlp_norm_config=rmsnorm_config,
            mlp_config=mlp_config,
            post_mlp_norm_config=None,
        )
        return DecoderConfig(
            embedding_config=embedding_config,
            rope_config=rope_config,
            layer_config=decoder_layer_config,
            output_norm_config=rmsnorm_config,
            vocab_size=self.vocab_size,
            model_dim=self.hidden_size,
            hidden_dim=self.intermediate_size,
            num_heads=self.num_attention_heads,
            num_groups=self.num_key_value_heads,
            head_dim=self.head_dim if self.head_dim is not None else self.hidden_size // self.num_attention_heads,
            attention_scale=None,
            num_layers=self.num_hidden_layers,
            sliding_window_sizes=None,
            context_length=context_length,
        )


@dataclass
class HFQwen2Config(HuggingFaceConfig):
    architectures: list[Literal["Qwen2ForCausalLM"]]
    attention_dropout: float
    bos_token_id: int | list[int]
    eos_token_id: int | list[int]
    hidden_act: Literal["silu"]
    hidden_size: int
    initializer_range: float
    intermediate_size: int
    max_position_embeddings: int
    max_window_layers: int
    model_type: Literal["qwen2"]
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    sliding_window: int
    tie_word_embeddings: bool
    transformers_version: str
    use_cache: bool
    use_sliding_window: bool
    vocab_size: int

    def _get_sliding_window_sizes(self) -> list[int | None]:
        sliding_window_sizes = []
        for i in range(self.num_hidden_layers):
            if i < self.max_window_layers:
                sliding_window_sizes.append(self.sliding_window)
            else:
                sliding_window_sizes.append(None)
        return sliding_window_sizes

    def to_decoder_config(
        self,
        context_length: int,
        activation_precision: DType,
        accumulation_precision: DType,
    ) -> DecoderConfig:
        embedding_config = TiedEmbeddingConfig(
            input_scale=None,
            logits_soft_cap=None,
            precision=activation_precision,
        )
        rope_config = UnscaledRoPEConfig(
            precision=activation_precision,
            base=self.rope_theta,
            max_sequence_length=self.max_position_embeddings,
        )
        rmsnorm_config = RMSNormConfig(
            scale_precision=activation_precision,
            accumulation_precision=accumulation_precision,
            epsilon=self.rms_norm_eps,
        )
        linear_config = FullPrecisionLinearConfig(
            precision=activation_precision,
        )
        attention_config = AttentionConfig(
            qkv_projection_config=linear_config,
            out_projection_config=linear_config,
            logit_soft_cap=None,
            has_qkv_biases=True,
            has_out_biases=False,
        )
        mlp_config = MLPConfig(
            linear_config=linear_config,
            activation=Activation.SILU,
        )
        decoder_layer_config = DecoderLayerConfig(
            pre_attention_norm_config=rmsnorm_config,
            attention_config=attention_config,
            post_attention_norm_config=None,
            pre_mlp_norm_config=rmsnorm_config,
            mlp_config=mlp_config,
            post_mlp_norm_config=None,
        )
        return DecoderConfig(
            embedding_config=embedding_config,
            rope_config=rope_config,
            layer_config=decoder_layer_config,
            output_norm_config=rmsnorm_config,
            vocab_size=self.vocab_size,
            model_dim=self.hidden_size,
            hidden_dim=self.intermediate_size,
            num_heads=self.num_attention_heads,
            num_groups=self.num_key_value_heads,
            head_dim=self.hidden_size // self.num_attention_heads,
            attention_scale=None,
            num_layers=self.num_hidden_layers,
            sliding_window_sizes=tuple(self._get_sliding_window_sizes()),
            context_length=context_length,
        )


@dataclass
class HFGemma2Config(HuggingFaceConfig):
    _add_one_to_rms_norm_weights: ClassVar[bool] = True

    architectures: list[Literal["Gemma2ForCausalLM"]]
    attention_bias: bool
    attention_dropout: float
    attn_logit_softcapping: float
    bos_token_id: int | list[int]
    cache_implementation: Literal["hybrid"]
    eos_token_id: int | list[int]
    final_logit_softcapping: float
    head_dim: int
    hidden_act: Literal["gelu_pytorch_tanh"]
    hidden_activation: Literal["gelu_pytorch_tanh"]
    hidden_size: int
    initializer_range: float
    intermediate_size: int
    max_position_embeddings: int
    model_type: Literal["gemma2"]
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    pad_token_id: int
    query_pre_attn_scalar: float
    rms_norm_eps: float
    rope_theta: float
    sliding_window: int
    transformers_version: str
    use_cache: bool
    vocab_size: int

    def to_decoder_config(
        self,
        context_length: int,
        activation_precision: DType,
        accumulation_precision: DType,
    ) -> DecoderConfig:
        sliding_window_sizes = tuple(
            self.sliding_window if not bool(i % 2) else None for i in range(self.num_hidden_layers)
        )
        embedding_input_scale = self.hidden_size**0.5
        attention_scale = self.query_pre_attn_scalar**-0.5
        embedding_config = TiedEmbeddingConfig(
            input_scale=embedding_input_scale,
            logits_soft_cap=self.final_logit_softcapping,
            precision=activation_precision,
        )
        rope_config = UnscaledRoPEConfig(
            precision=activation_precision,
            base=self.rope_theta,
            max_sequence_length=self.max_position_embeddings,
        )
        rmsnorm_config = RMSNormConfig(
            scale_precision=accumulation_precision,
            accumulation_precision=accumulation_precision,
            epsilon=self.rms_norm_eps,
        )
        linear_config = FullPrecisionLinearConfig(
            precision=activation_precision,
        )
        attention_config = AttentionConfig(
            qkv_projection_config=linear_config,
            out_projection_config=linear_config,
            logit_soft_cap=self.attn_logit_softcapping,
            has_qkv_biases=self.attention_bias,
            has_out_biases=False,
        )
        mlp_config = MLPConfig(
            linear_config=linear_config,
            activation=Activation.GELU,
        )
        decoder_layer_config = DecoderLayerConfig(
            pre_attention_norm_config=rmsnorm_config,
            attention_config=attention_config,
            post_attention_norm_config=rmsnorm_config,
            pre_mlp_norm_config=rmsnorm_config,
            mlp_config=mlp_config,
            post_mlp_norm_config=rmsnorm_config,
        )
        return DecoderConfig(
            embedding_config=embedding_config,
            rope_config=rope_config,
            layer_config=decoder_layer_config,
            output_norm_config=rmsnorm_config,
            vocab_size=self.vocab_size,
            model_dim=self.hidden_size,
            hidden_dim=self.intermediate_size,
            num_heads=self.num_attention_heads,
            num_groups=self.num_key_value_heads,
            head_dim=self.head_dim,
            attention_scale=attention_scale,
            num_layers=self.num_hidden_layers,
            sliding_window_sizes=sliding_window_sizes,
            context_length=context_length,
        )


@dataclass
class HFQwen25VLConfig(HuggingFaceConfig):
    """Configuration for Qwen 2.5 VL models."""
    architectures: list[Literal["Qwen2ForCausalLM", "Qwen2_5_VLForConditionalGeneration"]]
    attention_dropout: float
    bos_token_id: int
    eos_token_id: int
    hidden_act: Literal["silu"]
    hidden_size: int
    initializer_range: float
    intermediate_size: int
    max_position_embeddings: int
    max_window_layers: int
    model_type: Literal["qwen2", "qwen2_5_vl"]
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    sliding_window: int
    tie_word_embeddings: bool
    transformers_version: str
    use_cache: bool
    use_sliding_window: bool
    vocab_size: int
    vision_config: dict
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    
    def _get_sliding_window_sizes(self) -> list[int | None]:
        sliding_window_sizes = []
        for i in range(self.num_hidden_layers):
            if i < self.max_window_layers:
                sliding_window_sizes.append(self.sliding_window)
            else:
                sliding_window_sizes.append(None)
        return sliding_window_sizes
    
    def to_decoder_config(
        self,
        context_length: int,
        activation_precision: DType,
        accumulation_precision: DType,
    ) -> DecoderConfig:
        embedding_config = TiedEmbeddingConfig(
            input_scale=None,
            logits_soft_cap=None,
            precision=activation_precision,
        )
        rope_config = UnscaledRoPEConfig(
            precision=activation_precision,
            base=self.rope_theta,
            max_sequence_length=self.max_position_embeddings,
        )
        rmsnorm_config = RMSNormConfig(
            scale_precision=activation_precision,
            accumulation_precision=accumulation_precision,
            epsilon=self.rms_norm_eps,
        )
        linear_config = FullPrecisionLinearConfig(
            precision=activation_precision,
        )
        attention_config = AttentionConfig(
            qkv_projection_config=linear_config,
            out_projection_config=linear_config,
            logit_soft_cap=None,
            has_qkv_biases=True,
            has_out_biases=False,
        )
        mlp_config = MLPConfig(
            linear_config=linear_config,
            activation=Activation.SILU,
        )
        decoder_layer_config = DecoderLayerConfig(
            pre_attention_norm_config=rmsnorm_config,
            attention_config=attention_config,
            post_attention_norm_config=None,
            pre_mlp_norm_config=rmsnorm_config,
            mlp_config=mlp_config,
            post_mlp_norm_config=None,
        )
        return DecoderConfig(
            embedding_config=embedding_config,
            rope_config=rope_config,
            layer_config=decoder_layer_config,
            output_norm_config=rmsnorm_config,
            vocab_size=self.vocab_size,
            model_dim=self.hidden_size,
            hidden_dim=self.intermediate_size,
            num_heads=self.num_attention_heads,
            num_groups=self.num_key_value_heads,
            head_dim=self.hidden_size // self.num_attention_heads,
            attention_scale=None,
            num_layers=self.num_hidden_layers,
            sliding_window_sizes=tuple(self._get_sliding_window_sizes()),
            context_length=context_length,
        )
    
    def to_vision_config(
        self,
        precision: DType,
        accumulation_precision: DType,
    ) -> VisionConfig:
        """Convert to fartsovka VisionConfig."""
        vc = self.vision_config # This is the HF vision_config dict
        
        print(f"DEBUG: HF vision_config: {vc}")

        hf_main_hidden_size = vc["hidden_size"] # e.g., 1280
        hf_total_depth = vc["depth"] # e.g., 32
        hf_main_num_heads = vc["num_heads"] # e.g., 16 for the 1280-dim stage
        hf_main_intermediate_size = vc.get("intermediate_size", hf_main_hidden_size * 4)

        patch_embedding_config = PatchEmbeddingConfig(
            precision=precision,
            patch_size=vc["patch_size"],
            temporal_patch_size=vc["temporal_patch_size"],
            in_channels=vc.get("in_chans", vc.get("in_channels", 3)), 
        )
        
        if hf_main_hidden_size % hf_main_num_heads != 0:
            raise ValueError(f"HF vision_config hidden_size {hf_main_hidden_size} not divisible by num_heads {hf_main_num_heads}")
        actual_head_dim = hf_main_hidden_size // hf_main_num_heads

        if actual_head_dim % 4 != 0: 
            print(f"WARNING: actual_head_dim ({actual_head_dim}) is not divisible by 4. RoPE might be incorrect or fail.")
            # Fallback or error, for now, let it pass to see if VisionRoPEConfig catches it or if model runs.

        max_spatial_patches_for_rope = vc.get("image_size", vc.get("window_size", 224)) // vc["patch_size"]
        rope_config = VisionRoPEConfig(
            precision=precision,
            base=10000.0, 
            max_sequence_length=max_spatial_patches_for_rope,
        )
        
        norm_config = RMSNormConfig(
            scale_precision=precision,
            accumulation_precision=accumulation_precision,
            epsilon=vc.get("norm_eps", vc.get("layer_norm_eps", 1e-6)),
        )
        
        linear_config = FullPrecisionLinearConfig(precision=precision)
        
        hf_activation = vc.get("hidden_act", "silu")
        activation = Activation.SILU if hf_activation == "silu" else Activation.GELU
    
        attention_config = VisionSdpaAttentionConfig(
            qkv_projection_config=linear_config,
            out_projection_config=linear_config,
            logit_soft_cap=None,
            has_qkv_biases=vc.get("attention_bias", vc.get("qkv_bias", True)),
            has_out_biases=vc.get("attention_bias", vc.get("proj_bias", True)),
        )
        mlp_config = MLPConfig(
            linear_config=linear_config,
            activation=activation,
            has_biases=vc.get("mlp_bias", True),
        )
        
        layer_config = VisionLayerConfig(
            norm_config=norm_config,
            attention_config=attention_config,
            mlp_config=mlp_config,
        )

        patch_merger_config = PatchMergerConfig(
            precision=precision,
            spatial_merge_size=vc["spatial_merge_size"],
            has_biases=True, 
        )
        
        fartsovka_vision_config = VisionConfig(
            patch_embedding_config=patch_embedding_config,
            rope_config=rope_config,
            layer_config=layer_config, # Base config for all layers in the single stage
            patch_merger_config=patch_merger_config, # For the final merger
            output_norm_config=norm_config, # Norm before the final merger
            
            image_size=vc.get("image_size", vc.get("window_size", 224)),
            patch_size=vc["patch_size"],
            
            # Single stage definition
            stage_hidden_dims=(hf_main_hidden_size,),
            stage_depths=(hf_total_depth,),
            stage_num_heads=(hf_main_num_heads,),
            stage_mlp_intermediate_dims=(hf_main_intermediate_size,),
            
            attention_scale=None, 
            
            in_channels=vc.get("in_chans", vc.get("in_channels", 3)),
            temporal_patch_size=vc.get("temporal_patch_size", 2),
            temporal_pos_scale_factor=vc.get("temporal_pos_scale_factor", 1),
            spatial_merge_size=vc["spatial_merge_size"], # Used by the final merger
            out_hidden_size=vc["out_hidden_size"],
            fullatt_block_indexes=tuple(vc.get("fullatt_block_indexes", [])),
        )
        
        print(f"DEBUG: fartsovka VisionConfig (single stage): stage_hidden_dims: {fartsovka_vision_config.stage_hidden_dims}")
        print(f"DEBUG: fartsovka VisionConfig (single stage): stage_depths: {fartsovka_vision_config.stage_depths}")
        print(f"DEBUG: fartsovka VisionConfig (single stage): stage_num_heads: {fartsovka_vision_config.stage_num_heads}")
        print(f"DEBUG: fartsovka VisionConfig (single stage): out_hidden_size: {fartsovka_vision_config.out_hidden_size}")
        
        return fartsovka_vision_config

    def load_vision_model(
        self,
        precision: DType,
        accumulation_precision: DType,
        weights_dict: dict[str, Array],
    ) -> VisionTransformer:
        """Load Qwen 2.5 VL vision model from weights."""
        # Convert to our config format
        vision_config = self.to_vision_config(precision, accumulation_precision)
        
        # Initialize random model
        model = vision_config.random_init(key=jax.random.PRNGKey(0))
        print(f"DEBUG: Created vision model: {model}")
        
        # Import here to avoid circular imports
        from fartsovka.model_import.loaders.huggingface import load_vision_huggingface
        return load_vision_huggingface(model, weights_dict)

    def load_complete_model(
        self,
        context_length: int,
        precision: DType,
        accumulation_precision: DType,
        weights_dict: dict[str, Array],
    ) -> tuple[Decoder, VisionTransformer]:
        """Load both text and vision models for Qwen 2.5 VL."""
        # Initialize and load language model
        decoder_config = self.to_decoder_config(
            context_length=context_length,
            activation_precision=precision,
            accumulation_precision=accumulation_precision,
        )
        # Initialize decoder with random weights
        decoder = decoder_config.random_init(key=jax.random.PRNGKey(0))
        # Load weights from HF
        decoder = load_huggingface(decoder, weights_dict, False)
        
        # Initialize and load vision model
        vision_model = self.load_vision_model(precision, accumulation_precision, weights_dict)
        
        # Return both models
        return decoder, vision_model
        
    def merge_embeddings_with_vision_features(
        self,
        input_ids: jnp.ndarray,
        inputs_embeds: jnp.ndarray,
        vision_features: jnp.ndarray,
    ) -> jnp.ndarray:
        """Merge text embeddings with vision features at image token positions."""
        # Find positions of image and video tokens
        image_positions = input_ids == self.image_token_id
        video_positions = input_ids == self.video_token_id
        
        # Combine positions
        positions = jnp.logical_or(image_positions, video_positions)
        
        # Update embeddings at image/video token positions with vision features
        return jnp.where(
            positions[:, :, jnp.newaxis],
            vision_features,
            inputs_embeds
        )

    def create_joint_inference_fn(
        self,
        decoder: Decoder,
        vision_model: VisionTransformer,
    ):
        """Create a joint inference function for text and vision input.
        
        Returns a function that takes:
            input_ids: Int[Array, "batch_size seq_len"]
            images: Optional[Float[Array, "batch_size channels height width"]]
            mask: Bool[Array, "batch_size seq_len"]
            grid_thw: Optional[Int[Array, "batch_size 3"]]
            
        And returns:
            logits: Float[Array, "batch_size vocab_size"]
        """
        
        def process_inputs(
            input_ids: Int[Array, "batch_size seq_len"],
            images: Float[Array, "batch_size channels height width"] | None = None,
            mask: Bool[Array, "batch_size seq_len"] | None = None,
            grid_thw: Int[Array, "batch_size 3"] | None = None,
        ):
            # Get text embeddings using the embedding weights
            if input_ids is not None:
                # Process token IDs through the embedding
                token_positions = jnp.arange(input_ids.shape[-1])
                # Call decoder directly with the token IDs
                decoder_output = decoder(
                    token_ids=input_ids,
                    token_positions=token_positions,
                    mask=mask,
                    return_updated_kv_cache=False,
                )
                logits = decoder_output.output
                
                # If no images, return the text-only output
                if images is None:
                    return logits
                    
                # Otherwise, process the images and merge results
                vision_output = vision_model(images, grid_thw)
                vision_features = vision_output.output
                
                # For now, implement a simple placeholder for merging
                # In a real implementation, this would need to be more sophisticated
                print(f"Vision features shape: {vision_features.shape}")
                print(f"Text logits shape: {logits.shape}")
                
                # Return text-only output for now
                return logits
            else:
                raise ValueError("input_ids must be provided")
            
        return process_inputs
