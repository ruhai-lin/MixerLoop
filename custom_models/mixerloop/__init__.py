from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_mixerloop import MixerLoopConfig
from .modeling_mixerloop import MixerLoopForCausalLM, MixerLoopModel

__all__ = ['MixerLoopConfig', 'MixerLoopForCausalLM', 'MixerLoopModel']

AutoConfig.register('mixerloop', MixerLoopConfig)
AutoModel.register(MixerLoopConfig, MixerLoopModel)
AutoModelForCausalLM.register(MixerLoopConfig, MixerLoopForCausalLM)
