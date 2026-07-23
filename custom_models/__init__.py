"""Register the three recurrent GDN architectures with Transformers."""

from . import ffnloop, fullloop, mixerloop
from .ffnloop import FFNLoopConfig, FFNLoopForCausalLM, FFNLoopModel
from .fullloop import FullLoopConfig, FullLoopForCausalLM, FullLoopModel
from .mixerloop import MixerLoopConfig, MixerLoopForCausalLM, MixerLoopModel

__all__ = [
    "FFNLoopConfig",
    "FFNLoopForCausalLM",
    "FFNLoopModel",
    "FullLoopConfig",
    "FullLoopForCausalLM",
    "FullLoopModel",
    "MixerLoopConfig",
    "MixerLoopForCausalLM",
    "MixerLoopModel",
    "ffnloop",
    "fullloop",
    "mixerloop",
]
