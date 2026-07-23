# -*- coding: utf-8 -*-

from typing import Optional

from transformers.configuration_utils import PretrainedConfig


class MixerLoopConfig(PretrainedConfig):
    model_type = 'mixerloop'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        hidden_size: int = 288,
        num_hidden_layers: int = 6,
        num_heads: int = 4,
        head_dim: int = 72,
        expand_v: float = 1.0,
        intermediate_size: Optional[int] = 768,
        hidden_ratio: int = 4,
        conv_size: int = 4,
        attn_mode: str = 'chunk',
        loop_count: int = 4,
        use_residual: bool = True,
        max_position_embeddings: int = 1024,
        hidden_act: str = 'swish',
        initializer_range: float = 0.02,
        norm_eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id: Optional[int] = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = True,
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        fuse_linear_cross_entropy: bool = False,
        vocab_size: int = 32000,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.expand_v = expand_v
        self.intermediate_size = intermediate_size
        self.hidden_ratio = hidden_ratio
        self.conv_size = conv_size
        self.attn_mode = attn_mode
        self.loop_count = loop_count
        self.use_residual = use_residual
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.norm_eps = norm_eps
        self.use_cache = use_cache
        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.vocab_size = vocab_size

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
