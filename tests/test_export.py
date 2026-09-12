import pytest
import torch
import torch.distributed.checkpoint as dcp
from transformers import AutoModelForCausalLM, AutoTokenizer
from fla.models.gated_deltanet.configuration_gated_deltanet import GatedDeltaNetConfig

from custom_models.mixerloop import MixerLoopConfig
from flame.utils.convert_dcp_to_hf import save_pretrained


@pytest.mark.parametrize("config_class", [GatedDeltaNetConfig, MixerLoopConfig])
def test_hf_export_roundtrip(tmp_path, config_class):
    config = config_class(
        hidden_size=32, num_hidden_layers=1, num_heads=1, head_dim=32,
        expand_v=2, intermediate_size=64, vocab_size=32000,
        fuse_norm=False, fuse_swiglu=False, fuse_cross_entropy=False,
        tie_word_embeddings=True,
    )
    model = AutoModelForCausalLM.from_config(config)
    if hasattr(model.model, "residual_weight"):
        with torch.no_grad():
            model.model.residual_weight.fill_(0.125)
    config.save_pretrained(tmp_path)
    dcp.save({"model": model.state_dict()}, checkpoint_id=tmp_path / "checkpoint/step-1")
    save_pretrained(str(tmp_path), 1, str(tmp_path), "assets/tokenizer")
    restored = AutoModelForCausalLM.from_pretrained(tmp_path)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    assert restored.get_input_embeddings().weight is restored.get_output_embeddings().weight
    assert len(AutoTokenizer.from_pretrained(tmp_path)) == 32000
    assert (tmp_path / "model.safetensors").is_file()
    assert not (tmp_path / "tokenizer.bin").exists()
    assert not list(tmp_path.glob("*_q8.bin"))
