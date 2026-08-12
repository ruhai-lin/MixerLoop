"""Register current MixerLoop and the historical FullLoop paper baseline."""

from . import fullloop, mixerloop
from .fullloop import FullLoopConfig, FullLoopForCausalLM, FullLoopModel
from .mixerloop import MixerLoopConfig, MixerLoopForCausalLM, MixerLoopModel

__all__ = [
    "FullLoopConfig",
    "FullLoopForCausalLM",
    "FullLoopModel",
    "MixerLoopConfig",
    "MixerLoopForCausalLM",
    "MixerLoopModel",
    "fullloop",
    "mixerloop",
]
