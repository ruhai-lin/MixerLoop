"""Register MixerLoop and FullLoop with Transformers."""

from . import mixerloop
from .mixerloop import MixerLoopConfig, MixerLoopForCausalLM, MixerLoopModel
from . import fullloop
from .fullloop import FullLoopConfig, FullLoopForCausalLM, FullLoopModel

__all__ = [
    "FullLoopConfig",
    "FullLoopForCausalLM",
    "FullLoopModel",
    "fullloop",
    "MixerLoopConfig",
    "MixerLoopForCausalLM",
    "MixerLoopModel",
    "mixerloop",
]
