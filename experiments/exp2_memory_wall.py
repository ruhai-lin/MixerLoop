"""Architecture analysis: python3 experiments/exp2_memory_wall.py.

Offline, batch-one projection-MAC model; no model weights are downloaded.
Outputs vector PDF, 300-dpi PNG, and auditable CSV tables in outputs.
See exp2_memory_wall.md for whole-model accounting and the FFN-budget-adjusted memory wall.
"""

import csv
import hashlib
import json
import math
from itertools import zip_longest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea, VPacker


# Resolved dimensions from official revisions. Each entry becomes one whole-model
# point; adding a supported geometry needs no changes to the plotting functions.
MODELS = {
    "Qwen3-8B": {
        "main_figure": False,
        "plot_label": "Qwen3 8B",
        "family": "Qwen3", "arch": "dense_gqa", "mlp_type": "swiglu",
        "hidden_size": 4096, "num_attention_heads": 32,
        "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 12288,
        "num_hidden_layers": 36, "vocab_size": 151936,
        "source": "https://huggingface.co/Qwen/Qwen3-8B/raw/b968826d9c46dd6066d109eabc6255188de91218/config.json",
        "source_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "reference_ratio": 0.278,
    },
    "Qwen3-14B": {
        "main_figure": False,
        "family": "Qwen3", "arch": "dense_gqa", "mlp_type": "swiglu",
        "hidden_size": 5120, "num_attention_heads": 40,
        "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 17408,
        "num_hidden_layers": 40, "vocab_size": 151936,
        "source": "https://huggingface.co/Qwen/Qwen3-14B/raw/40c069824f4251a91eefaf281ebe4c544efd3e18/config.json",
        "source_revision": "40c069824f4251a91eefaf281ebe4c544efd3e18",
        "reference_ratio": 0.235,
    },
    "Qwen3-32B": {
        "main_figure": False,
        "family": "Qwen3", "arch": "dense_gqa", "mlp_type": "swiglu",
        "hidden_size": 5120, "num_attention_heads": 64,
        "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 25600,
        "num_hidden_layers": 64, "vocab_size": 151936,
        "source": "https://huggingface.co/Qwen/Qwen3-32B/raw/9216db5781bf21249d130ec9da846c4624c16137/config.json",
        "source_revision": "9216db5781bf21249d130ec9da846c4624c16137",
        "reference_ratio": 0.240,
    },
    "Llama-3.1-8B": {
        "main_figure": False,
        "plot_label": "Llama 3.1 8B",
        "family": "Llama 3.1", "arch": "dense_gqa", "mlp_type": "swiglu",
        "hidden_size": 4096, "num_attention_heads": 32,
        "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 14336,
        "num_hidden_layers": 32, "vocab_size": 128256,
        "ffn_dim_multiplier": 1.3, "multiple_of": 1024,
        "source": "https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/sku_list.py",
        "source_revision": "0e0b8c519242d5833d8c11bffc1232b77ad7f301",
        "implementation_source": "https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/llama3/model.py",
        "notes": "Official SKU args; resolved head/FFN widths from model.py.",
        "reference_ratio": 0.238,
    },
    "Llama-3.1-70B": {
        "main_figure": False,
        "plot_label": "Llama 3.1 70B",
        "family": "Llama 3.1", "arch": "dense_gqa", "mlp_type": "swiglu",
        "hidden_size": 8192, "num_attention_heads": 64,
        "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 28672,
        "num_hidden_layers": 80, "vocab_size": 128256,
        "ffn_dim_multiplier": 1.3, "multiple_of": 4096,
        "source": "https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/sku_list.py",
        "source_revision": "0e0b8c519242d5833d8c11bffc1232b77ad7f301",
        "implementation_source": "https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/llama3/model.py",
        "notes": "Official SKU args; resolved head/FFN widths from model.py.",
        "reference_ratio": 0.214,
    },
    "Llama-3.1-405B": {
        "main_figure": False,
        "family": "Llama 3.1", "arch": "dense_gqa", "mlp_type": "swiglu",
        "hidden_size": 16384, "num_attention_heads": 128,
        "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 53248,
        "num_hidden_layers": 126, "vocab_size": 128256,
        "ffn_dim_multiplier": 1.2, "multiple_of": 4096,
        "source": "https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/sku_list.py",
        "source_revision": "0e0b8c519242d5833d8c11bffc1232b77ad7f301",
        "implementation_source": "https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/llama3/model.py",
        "notes": "Canonical bf16-mp8 SKU, 8 KV heads; not the replicated-KV mp16 variant.",
        "reference_ratio": 0.218,
    },
    "Gemma-3-12B": {
        "main_figure": False,
        "plot_label": "Gemma 3 12B",
        "family": "Gemma 3", "arch": "dense_gqa", "mlp_type": "geglu",
        "hidden_size": 3840, "num_attention_heads": 16,
        "num_key_value_heads": 8, "head_dim": 256, "intermediate_size": 15360,
        "num_hidden_layers": 48, "vocab_size": 262144,
        "source": "https://github.com/google/gemma_pytorch/blob/014acb7ac4563a5f77c76d7ff98f31b568c16508/gemma/config.py",
        "source_revision": "014acb7ac4563a5f77c76d7ff98f31b568c16508",
        "implementation_source": "https://github.com/google/gemma_pytorch/blob/014acb7ac4563a5f77c76d7ff98f31b568c16508/gemma/model.py",
        "notes": "Text decoder; get_config_for_12b; gated GELU has three matrices.",
        "reference_ratio": 0.267,
    },
    "Gemma-3-27B": {
        "main_figure": False,
        "family": "Gemma 3", "arch": "dense_gqa", "mlp_type": "geglu",
        "hidden_size": 5376, "num_attention_heads": 32,
        "num_key_value_heads": 16, "head_dim": 128, "intermediate_size": 21504,
        "num_hidden_layers": 62, "vocab_size": 262144,
        "source": "https://github.com/google/gemma_pytorch/blob/014acb7ac4563a5f77c76d7ff98f31b568c16508/gemma/config.py",
        "source_revision": "014acb7ac4563a5f77c76d7ff98f31b568c16508",
        "implementation_source": "https://github.com/google/gemma_pytorch/blob/014acb7ac4563a5f77c76d7ff98f31b568c16508/gemma/model.py",
        "notes": "Text decoder; get_config_for_27b_v3, not the Gemma 2 27B config.",
        "reference_ratio": 0.190,
    },
    "Mistral-Small-3.1-24B": {
        "main_figure": False,
        "plot_label": "Mistral 24B",
        "family": "Mistral", "arch": "dense_gqa", "mlp_type": "swiglu",
        "hidden_size": 5120, "num_attention_heads": 32,
        "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 32768,
        "num_hidden_layers": 40, "vocab_size": 131072,
        "source": "https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/raw/68faf511d618ef198fef186659617cfd2eb8e33a/config.json",
        "source_revision": "68faf511d618ef198fef186659617cfd2eb8e33a",
        "notes": "Resolved text_config; vision encoder excluded.",
        "reference_ratio": 0.104,
    },
    "Phi-4": {
        "main_figure": False,
        "plot_label": "Phi-4 14B",
        "family": "Phi", "arch": "dense_gqa", "mlp_type": "swiglu",
        "hidden_size": 5120, "num_attention_heads": 40,
        "num_key_value_heads": 10, "head_dim": 128, "intermediate_size": 17920,
        "num_hidden_layers": 40, "vocab_size": 100352,
        "source": "https://huggingface.co/microsoft/phi-4/raw/2db69c1c3e91a05d2c64a3185acfbaf36f744e25/config.json",
        "source_revision": "2db69c1c3e91a05d2c64a3185acfbaf36f744e25",
        "implementation_source": "https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/phi3/modeling_phi3.py",
        "notes": "head_dim=d/heads; fused gate_up is two matrices, not one.",
        "reference_ratio": 0.238,
    },
    "OLMo-2-32B": {
        "main_figure": False,
        "family": "OLMo", "arch": "dense_gqa", "mlp_type": "swiglu",
        "hidden_size": 5120, "num_attention_heads": 40,
        "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 27648,
        "num_hidden_layers": 64, "vocab_size": 100352,
        "source": "https://huggingface.co/allenai/OLMo-2-0325-32B/raw/cc9d3cf9c7230b86ee6b84607b37db1c01e3f1ed/config.json",
        "source_revision": "cc9d3cf9c7230b86ee6b84607b37db1c01e3f1ed",
        "implementation_source": "https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/olmo2/modeling_olmo2.py",
        "notes": "head_dim resolved as hidden_size/num_attention_heads.",
        "reference_ratio": 0.148,
    },
    "Yi-1.5-34B": {
        "main_figure": False,
        "family": "Yi", "arch": "dense_gqa", "mlp_type": "swiglu",
        "hidden_size": 7168, "num_attention_heads": 56,
        "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 20480,
        "num_hidden_layers": 60, "vocab_size": 64000,
        "source": "https://huggingface.co/01-ai/Yi-1.5-34B/raw/58a29f6dca2a4eda38821c086c29c1906114aabb/config.json",
        "source_revision": "58a29f6dca2a4eda38821c086c29c1906114aabb",
        "implementation_source": "https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/llama/modeling_llama.py",
        "notes": "LlamaForCausalLM; head_dim=hidden_size/num_attention_heads.",
        "reference_ratio": 0.267,
    },
    "Falcon3-10B": {
        "main_figure": False,
        "family": "Falcon", "arch": "dense_gqa", "mlp_type": "swiglu",
        "hidden_size": 3072, "num_attention_heads": 12,
        "num_key_value_heads": 4, "head_dim": 256, "intermediate_size": 23040,
        "num_hidden_layers": 40, "vocab_size": 131072,
        "source": "https://huggingface.co/tiiuae/Falcon3-10B-Base/raw/34bb99a889fe0426412da3dd2b46e6f64c8fd003/config.json",
        "source_revision": "34bb99a889fe0426412da3dd2b46e6f64c8fd003",
        "reference_ratio": 0.119,
    },
    "Mixtral-8x7B": {
        "main_figure": False,
        "plot_label": "Mixtral 8x7B",
        "family": "Mixtral", "arch": "gqa_moe", "mlp_type": "swiglu",
        "hidden_size": 4096, "num_attention_heads": 32,
        "num_key_value_heads": 8, "head_dim": 128,
        "expert_intermediate_size": 14336, "num_experts_per_tok": 2,
        "num_shared_experts": 0, "shared_expert_intermediate_size": 0,
        "num_experts": 8,
        "num_hidden_layers": 32, "vocab_size": 32000,
        "source": "https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/raw/fc7ac94680e38d7348cfa806e51218e6273104b0/config.json",
        "source_revision": "fc7ac94680e38d7348cfa806e51218e6273104b0",
        "implementation_source": "https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/mixtral/modeling_mixtral.py",
        "notes": "2 of 8 experts; no shared expert; head_dim=d/heads.",
        "reference_ratio": 0.119,
    },
    "Qwen3-30B-A3B": {
        "main_figure": False,
        "plot_label": "Qwen3 30B-A3B",
        "family": "Qwen3", "arch": "gqa_moe", "mlp_type": "swiglu",
        "hidden_size": 2048, "num_attention_heads": 32,
        "num_key_value_heads": 4, "head_dim": 128,
        "expert_intermediate_size": 768, "num_experts_per_tok": 8,
        "num_shared_experts": 0, "shared_expert_intermediate_size": 0,
        "num_experts": 128,
        "num_hidden_layers": 48, "vocab_size": 151936,
        "source": "https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/config.json",
        "source_revision": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
        "implementation_source": "https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py",
        "reference_ratio": 0.500,
    },
    "Qwen3-235B-A22B": {
        "main_figure": False,
        "family": "Qwen3", "arch": "gqa_moe", "mlp_type": "swiglu",
        "hidden_size": 4096, "num_attention_heads": 64,
        "num_key_value_heads": 4, "head_dim": 128,
        "expert_intermediate_size": 1536, "num_experts_per_tok": 8,
        "num_shared_experts": 0, "shared_expert_intermediate_size": 0,
        "num_experts": 128,
        "num_hidden_layers": 94, "vocab_size": 151936,
        "source": "https://huggingface.co/Qwen/Qwen3-235B-A22B/raw/8efa61729e24bd65b1d152b5ab5409052aa80e65/config.json",
        "source_revision": "8efa61729e24bd65b1d152b5ab5409052aa80e65",
        "implementation_source": "https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py",
        "reference_ratio": 0.472,
    },
    "GLM-4.5": {
        "main_figure": False,
        "plot_label": "GLM-4.5",
        "family": "GLM-4.5", "arch": "gqa_moe", "mlp_type": "swiglu",
        "hidden_size": 5120, "num_attention_heads": 96,
        "num_key_value_heads": 8, "head_dim": 128,
        "expert_intermediate_size": 1536, "num_experts_per_tok": 8,
        "num_shared_experts": 1, "shared_expert_intermediate_size": 1536,
        "num_experts": 160, "num_hidden_layers": 92, "dense_prefix_layers": 3,
        "vocab_size": 151552, "intermediate_size": 12288,
        "source": "https://huggingface.co/zai-org/GLM-4.5/raw/cbb2c7cfb52fa128a9660cb1a7a78e017899e115/config.json",
        "source_revision": "cbb2c7cfb52fa128a9660cb1a7a78e017899e115",
        "implementation_source": "https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/glm4_moe/modeling_glm4_moe.py",
        "notes": "Representative MoE block, not the first three dense layers.",
        "reference_ratio": 0.642,
    },
    "DeepSeek-V3": {
        "main_figure": False,
        "plot_label": "DeepSeek-V3",
        "family": "DeepSeek-V3", "arch": "mla_moe", "mlp_type": "swiglu",
        "hidden_size": 7168, "num_attention_heads": 128,
        "q_lora_rank": 1536, "kv_lora_rank": 512,
        "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
        "expert_intermediate_size": 2048, "num_experts_per_tok": 8,
        "num_shared_experts": 1, "shared_expert_intermediate_size": 2048,
        "num_experts": 256, "num_hidden_layers": 61, "dense_prefix_layers": 3,
        "vocab_size": 129280, "intermediate_size": 18432,
        "source": "https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/e815299b0bcbac849fa540c768ef21845365c9eb/config.json",
        "source_revision": "e815299b0bcbac849fa540c768ef21845365c9eb",
        "implementation_source": "https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/e815299b0bcbac849fa540c768ef21845365c9eb/modeling_deepseek.py",
        "notes": "Unabsorbed MLA projections in a representative MoE block; excludes MTP.",
        "reference_ratio": 0.472,
    },
    "Kimi-K2": {
        "main_figure": False,
        "plot_label": "Kimi-K2",
        "family": "Kimi-K2", "arch": "mla_moe", "mlp_type": "swiglu",
        "hidden_size": 7168, "num_attention_heads": 64,
        "q_lora_rank": 1536, "kv_lora_rank": 512,
        "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
        "expert_intermediate_size": 2048, "num_experts_per_tok": 8,
        "num_shared_experts": 1, "shared_expert_intermediate_size": 2048,
        "num_experts": 384, "num_hidden_layers": 61, "dense_prefix_layers": 1,
        "vocab_size": 163840, "intermediate_size": 18432,
        "source": "https://huggingface.co/moonshotai/Kimi-K2-Instruct/raw/fd1984e2b7a3350dbf7305fe73a4ede25c14de50/config.json",
        "source_revision": "fd1984e2b7a3350dbf7305fe73a4ede25c14de50",
        "implementation_source": "https://huggingface.co/moonshotai/Kimi-K2-Instruct/raw/fd1984e2b7a3350dbf7305fe73a4ede25c14de50/modeling_deepseek.py",
        "notes": "Unabsorbed MLA projections in a representative MoE block.",
        "reference_ratio": 0.255,
    },
    "Ouro-1.4B": {
        "main_figure": False,
        "family": "Ouro", "arch": "dense_gqa", "mlp_type": "swiglu",
        "hidden_size": 2048, "num_attention_heads": 16,
        "num_key_value_heads": 16, "head_dim": 128, "intermediate_size": 5632,
        "num_hidden_layers": 24, "loop_count": 4, "recurrence": "full_block",
        "source": "https://huggingface.co/ByteDance/Ouro-1.4B/raw/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/config.json",
        "source_revision": "574fa66cb8bf5abdc979642d01cf2b79b16bfab1",
        "implementation_source": "https://huggingface.co/ByteDance/Ouro-1.4B/raw/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/modeling_ouro.py",
        "notes": "Full-block recurrence; FFN weights streamed again on every pass.",
        "reference_ratio": 0.484848,
    },
    "LT2-1.3B": {
        "main_figure": False,
        "family": "LT2", "arch": "hybrid", "mlp_type": "swiglu",
        "hidden_size": 2048, "num_attention_heads": 16,
        "intermediate_size": 5632, "num_hidden_layers": 21,
        "loop_count": 4, "recurrence": "full_block",
        "blocks": [
            {"arch": "dense_gqa", "count": 4, "num_attention_heads": 16,
             "num_key_value_heads": 16, "head_dim": 128},
            {"arch": "mixerloop", "count": 17, "num_heads": 16,
             "head_dim": 96, "expand_v": 2},
        ],
        "source": "https://github.com/chili-lab/LT2/blob/1ac9ed5c913d88623cca34da3658b57400079767/apps/LT2/configs/1B/looped_hybrid_gdn_4to1_1B.yaml",
        "source_revision": "1ac9ed5c913d88623cca34da3658b57400079767",
        "implementation_source": "https://github.com/chili-lab/LT2/blob/1ac9ed5c913d88623cca34da3658b57400079767/apps/LT2/transformer.py",
        "ffn_source": "https://github.com/chili-lab/LT2/blob/1ac9ed5c913d88623cca34da3658b57400079767/lingua/transformer.py",
        "dependency_source": "https://github.com/fla-org/flash-linear-attention/blob/954438d1fcb5e1bb05c22f9908de9c5c2df74ae5/fla/layers/gated_deltanet.py",
        "notes": "Executable recipe: 21 layers, interleaved 4 GDN:1 full => 17+4, T=4. Stale YAML comments say 16/T=3. FFN rounds 8d/3 to 256 => 5632. LT2 leaves FLA unpinned; audited default expand_v=2.",
        "reference_ratio": 0.681097,
    },
    "MiniMind2-104M": {
        "family": "MiniMind", "arch": "dense_gqa", "mlp_type": "swiglu",
        "plot_label": "MiniMind2 104M",
        "hidden_size": 768, "intermediate_size": 2048, "num_hidden_layers": 16,
        "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 96,
        "source": "https://huggingface.co/jingyaogong/MiniMind2/raw/d18103e50a735dfb8ab801183be734bc635dfc22/config.json",
        "source_revision": "d18103e50a735dfb8ab801183be734bc635dfc22",
        "reference_point": (0.0849, 0.02359),
    },
    "OPT-125M": {
        "family": "OPT", "arch": "dense_gqa", "mlp_type": "relu",
        "plot_label": "OPT 125M",
        "hidden_size": 768, "intermediate_size": 3072, "num_hidden_layers": 12,
        "num_attention_heads": 12, "num_key_value_heads": 12, "head_dim": 64,
        "source": "https://huggingface.co/facebook/opt-125m/raw/27dcfa74d334bc871f3234de431e71c6eeba5dd6/config.json",
        "source_revision": "27dcfa74d334bc871f3234de431e71c6eeba5dd6",
        "notes": "OPT ffn_dim=3072; ordinary two-matrix ReLU FFN; MHA.",
        "reference_point": (0.0637, 0.02831),
    },
    "SmolLM2-135M": {
        "family": "SmolLM2", "arch": "dense_gqa", "mlp_type": "swiglu",
        "plot_label": "SmolLM2 135M",
        "hidden_size": 576, "intermediate_size": 1536, "num_hidden_layers": 30,
        "num_attention_heads": 9, "num_key_value_heads": 3, "head_dim": 64,
        "source": "https://huggingface.co/HuggingFaceTB/SmolLM2-135M/raw/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/config.json",
        "source_revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
        "reference_point": (0.0896, 0.02654),
    },
    "SmolLM2-360M": {
        "family": "SmolLM2", "arch": "dense_gqa", "mlp_type": "swiglu",
        "plot_label": "SmolLM2 360M",
        "hidden_size": 960, "intermediate_size": 2560, "num_hidden_layers": 32,
        "num_attention_heads": 15, "num_key_value_heads": 5, "head_dim": 64,
        "source": "https://huggingface.co/HuggingFaceTB/SmolLM2-360M/raw/f8027fd0eaeea54caa13c31d31b9fdc459c38b49/config.json",
        "source_revision": "f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
        "reference_point": (0.2654, 0.07864),
    },
    "Qwen2.5-0.5B": {
        "family": "Qwen2.5", "arch": "dense_gqa", "mlp_type": "swiglu",
        "plot_label": "Qwen2.5 0.5B",
        "hidden_size": 896, "intermediate_size": 4864, "num_hidden_layers": 24,
        "num_attention_heads": 14, "num_key_value_heads": 2, "head_dim": 64,
        "source": "https://huggingface.co/Qwen/Qwen2.5-0.5B/raw/060db6499f32faf8b98477b0a26969ef7d8b9987/config.json",
        "source_revision": "060db6499f32faf8b98477b0a26969ef7d8b9987",
        "reference_point": (0.3530, 0.04404),
    },
    "FLAME-MoE-38M-100M": {
        "family": "FLAME-MoE", "arch": "gqa_moe", "mlp_type": "swiglu",
        "plot_label": "FLAME 38M/A",
        "hidden_size": 256, "num_hidden_layers": 9, "dense_prefix_layers": 1,
        "intermediate_size": 1368, "num_attention_heads": 16,
        "num_key_value_heads": 16, "head_dim": 16,
        "expert_intermediate_size": 176, "num_experts": 64, "num_experts_per_tok": 6,
        "num_shared_experts": 1, "shared_expert_intermediate_size": 352,
        "source": "https://github.com/cmu-flame/FLAME-MoE/blob/e9b2fe2df3f1abb8dbb9ec0eabde8cdfb65e5c78/scripts/release/flame-moe-38m.sh",
        "source_revision": "e9b2fe2df3f1abb8dbb9ec0eabde8cdfb65e5c78",
        "implementation_source": "https://github.com/cmu-flame/FLAME-MoE/blob/e9b2fe2df3f1abb8dbb9ec0eabde8cdfb65e5c78/configs/model/flame-moe.sh",
        "notes": "Released Megatron recipe: 1 dense + 8 MoE; routed top-6 plus shared width 2*176. HF model-card top-8 summary differs from executable configuration.",
        "reference_point": (0.0109, 0.002359),
    },
    "TinyMoE-100M-2x8": {
        "family": "TinyMoE", "arch": "gqa_moe", "mlp_type": "swiglu",
        "plot_label": "TinyMoE 100M",
        "hidden_size": 384, "num_hidden_layers": 10,
        "num_attention_heads": 8, "num_key_value_heads": 4, "head_dim": 48,
        "expert_intermediate_size": 768, "num_experts": 8, "num_experts_per_tok": 2,
        "num_shared_experts": 0, "shared_expert_intermediate_size": 0,
        "source": "https://huggingface.co/FlameF0X/TinyMoE-100m-2x8/raw/0819979158f9f1980dcaa4f300927a6ab308f88b/config.json",
        "source_revision": "0819979158f9f1980dcaa4f300927a6ab308f88b",
        "notes": "Mixtral head_dim=null resolves to hidden_size/num_attention_heads=48.",
        "reference_point": (0.0199, 0.004424),
    },
    "TinyMoE-200M-2x16": {
        "family": "TinyMoE", "arch": "gqa_moe", "mlp_type": "swiglu",
        "plot_label": "TinyMoE 200M",
        "hidden_size": 384, "num_hidden_layers": 12,
        "num_attention_heads": 8, "num_key_value_heads": 4, "head_dim": 48,
        "expert_intermediate_size": 768, "num_experts": 16, "num_experts_per_tok": 2,
        "num_shared_experts": 0, "shared_expert_intermediate_size": 0,
        "source": "https://huggingface.co/FlameF0X/TinyMoE-200m-2x16/raw/9b209f7439aadfac282e0bf44f2fda19771aba90/config.json",
        "source_revision": "9b209f7439aadfac282e0bf44f2fda19771aba90",
        "reference_point": (0.0239, 0.005308),
    },
    "Pluto-Nano-0.5": {
        "family": "Pluto", "arch": "gqa_moe", "mlp_type": "swiglu",
        "plot_label": "Pluto 1B/A47M",
        "hidden_size": 384, "num_hidden_layers": 16,
        "num_attention_heads": 6, "num_key_value_heads": 2, "head_dim": 64,
        "expert_intermediate_size": 1536, "num_experts": 35, "num_experts_per_tok": 1,
        "num_shared_experts": 0, "shared_expert_intermediate_size": 0,
        "source": "https://huggingface.co/ASTRAI-labs/pluto-nano-0.5/raw/8a6cf3cdbbe3e246a7a5127acd169b64d7e8011b/config.json",
        "source_revision": "8a6cf3cdbbe3e246a7a5127acd169b64d7e8011b",
        "implementation_source": "https://huggingface.co/ASTRAI-labs/pluto-nano-0.5/raw/8a6cf3cdbbe3e246a7a5127acd169b64d7e8011b/modeling_pluto.py",
        "notes": "Q/K/V/O plus top-1 SwiGLU, no shared expert; training-only MTP excluded.",
        "reference_point": (0.0319, 0.006291),
    },
    "MiniMind-3-MoE": {
        "family": "MiniMind", "arch": "gqa_moe", "mlp_type": "swiglu",
        "plot_label": "MiniMind3 MoE A64M",
        "hidden_size": 768, "num_hidden_layers": 8,
        "num_attention_heads": 8, "num_key_value_heads": 4, "head_dim": 96,
        "expert_intermediate_size": 2432, "num_experts": 4, "num_experts_per_tok": 1,
        "num_shared_experts": 0, "shared_expert_intermediate_size": 0,
        "source": "https://huggingface.co/jingyaogong/minimind-3-moe/raw/1dc1e5702b361d0fc024060d03efde6f9dfceaa7/config.json",
        "source_revision": "1dc1e5702b361d0fc024060d03efde6f9dfceaa7",
        "notes": "Qwen3-MoE export, all eight layers MoE, no shared expert.",
        "reference_point": (0.0504, 0.01416),
    },
}

HARDWARE = {
    "KV260 4HP": {
        "compute_tops": 2 * (64 * 8 / 9) * 150e6 / 1e12,
        "bandwidth_gbs": 9.377286, "precision": "int8", "kind": "measured",
        "source": "exp1.md#streaming-bandwidth-calibration", "label": "4HP",
    },
    "KV260 2HP": {
        "compute_tops": 2 * (64 * 8 / 9) * 150e6 / 1e12,
        "bandwidth_gbs": 4.799728, "precision": "int8", "kind": "measured",
        "source": "exp1.md#streaming-bandwidth-calibration", "label": "2HP",
    },
    "KV260 1HP": {
        "compute_tops": 2 * (64 * 8 / 9) * 150e6 / 1e12,
        "bandwidth_gbs": 2.399983, "precision": "int8", "kind": "measured",
        "source": "exp1.md#streaming-bandwidth-calibration", "label": "1HP",
    },
    "GTX 1080": {
        "compute_tops": 2560 * 1.733 * 8 / 1000, "bandwidth_gbs": 320,
        "precision": "int8", "kind": "nominal", "label": "1080",
        "source": "https://www.pny.com/file%20library/support/pny%20products/resource%20center/graphics%20cards/spanish/product%20brochure/pny-nvidia-geforce-gtx-1080-8gb-founders-edition-spa.pdf",
        "compute_source": "https://developer.nvidia.com/blog/mixed-precision-programming-cuda-8/",
    },
    "A100 80GB SXM": {
        "compute_tops": 624, "bandwidth_gbs": 2039,
        "precision": "int8", "kind": "nominal", "label": "A100",
        "source": "https://www.nvidia.com/en-us/data-center/a100/",
    },
    "RTX 3090 Ti": {
        "compute_tops": 84 * 1024 * 1.86 * 2 / 1000, "bandwidth_gbs": 1008,
        "precision": "int8", "kind": "nominal", "label": "3090 Ti",
        "source": "https://images.nvidia.cn/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf",
        "compute_source": "https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf",
    },
    "H100 SXM": {
        "compute_tops": 1979, "bandwidth_gbs": 3350,
        "precision": "int8", "kind": "nominal", "label": "H100",
        "source": "https://www.nvidia.com/en-us/data-center/h100/",
    },
}

MIXERLOOP_CONFIGS = {
    "MixerLoop 13M": "configs/mixerloop_13m.json",
    "MixerLoop 100M": "configs/mixerloop_100m.json",
    "MixerLoop ~450M": "configs/mixerloop_600m.json",
}
ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
WEIGHT_BYTES = 1 + 4 / 32
MLP_MATRICES = {"swiglu": 3, "geglu": 3, "gelu": 2, "relu": 2}
LOOP_COUNTS = (1, 2, 3, 4)
MIXER_COLOR = "#435E48"
AUX_COLOR = "#BCC5BA"
TEXT_COLOR = "#303330"
WALL_COLOR = "#C7A33A"


def calc_dense_attention_macs(cfg):
    d = cfg["hidden_size"]
    q = cfg["num_attention_heads"] * cfg["head_dim"]
    kv = cfg["num_key_value_heads"] * cfg["head_dim"]
    return 2 * d * (q + kv)


def calc_dense_ffn_macs(cfg):
    matrices = MLP_MATRICES[cfg["mlp_type"]]
    return matrices * cfg["hidden_size"] * cfg["intermediate_size"]


def calc_moe_ffn_macs(cfg):
    # Total expert count is metadata, never a single-token traffic multiplier.
    matrices = MLP_MATRICES[cfg["mlp_type"]]
    routed = cfg["expert_intermediate_size"] * cfg["num_experts_per_tok"]
    shared = cfg["shared_expert_intermediate_size"] * cfg["num_shared_experts"]
    return matrices * cfg["hidden_size"] * (routed + shared)


def calc_mla_attention_macs(cfg):
    """Declared MLA projection graph, before inference-time weight absorption."""
    d, h = cfg["hidden_size"], cfg["num_attention_heads"]
    rq, rkv = cfg["q_lora_rank"], cfg["kv_lora_rank"]
    nope, rope, v = cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"], cfg["v_head_dim"]
    q = d * h * (nope + rope) if rq is None else d * rq + rq * h * (nope + rope)
    return q + d * (rkv + rope) + rkv * h * (nope + v) + h * v * d


def calc_model_metrics(cfg):
    calculators = {
        "dense_gqa": (calc_dense_attention_macs, calc_dense_ffn_macs),
        "gqa_moe": (calc_dense_attention_macs, calc_moe_ffn_macs),
        "mla_moe": (calc_mla_attention_macs, calc_moe_ffn_macs),
    }
    if cfg["arch"] == "hybrid":
        # Sum the actual interleaved stack, not an approximate 4:1 average.
        a = f = 0
        for block in cfg["blocks"]:
            resolved = {**cfg, **block}
            if block["arch"] == "mixerloop":
                block_a, block_f = calc_mixerloop_macs(resolved)
            else:
                attention, ffn = calculators[block["arch"]]
                block_a, block_f = attention(resolved), ffn(resolved)
            a += block["count"] * block_a
            f += block["count"] * block_f
    else:
        attention, ffn = calculators[cfg["arch"]]
        a, f = attention(cfg), ffn(cfg)
    loops = cfg.get("loop_count", 1)
    assert loops == 1 or cfg.get("recurrence") == "full_block"
    # Full-block recurrence repeats both compute and streamed FFN weights.
    return {"attention_macs": a, "active_ffn_macs": f,
            "r_arch": a / f, "kappa": 1 + a / f, "loop_count": loops,
            "compute_macs": loops * (a + f),
            "ffn_weight_bytes": loops * f * WEIGHT_BYTES,
            "effective_intensity": (loops * (a + f)) / (loops * f * WEIGHT_BYTES)}


def calc_mixerloop_macs(cfg):
    """Q/K/V/G/O and W1/W3/W2: the model's large projection matrices."""
    d = cfg["hidden_size"]
    k = cfg["num_heads"] * cfg["head_dim"]
    head_v = cfg["head_dim"] * cfg["expand_v"]
    assert head_v == int(head_v) and head_v > 0
    v = cfg["num_heads"] * int(head_v)
    return d * (2 * k + 3 * v), 3 * d * cfg["intermediate_size"]


def calc_mixerloop_metrics(cfg, loop_count=1):
    m, f = calc_mixerloop_macs(cfg)
    assert isinstance(loop_count, int) and loop_count >= 1
    return {"attention_macs": m, "active_ffn_macs": f,
            "r_arch": m / f, "kappa": 1 + loop_count * m / f,
            "loop_count": loop_count, "compute_macs": f + loop_count * m,
            "ffn_weight_bytes": f * WEIGHT_BYTES,
            "effective_intensity": (f + loop_count * m) / (f * WEIGHT_BYTES)}


def calc_hardware_balance(cfg):
    assert cfg["precision"] == "int8"
    return (cfg["compute_tops"] / 2) * 1000 / cfg["bandwidth_gbs"]


def validate_registry(models=MODELS, hardware=HARDWARE):
    """Fail on missing dimensions, unsupported semantics, or regression drift."""
    for name, cfg in models.items():
        assert cfg["source"] and cfg["source_revision"] and cfg["family"], name
        assert cfg["hidden_size"] > 0 and cfg["num_attention_heads"] > 0, name
        if cfg["arch"] == "hybrid":
            assert sum(b["count"] for b in cfg["blocks"]) == cfg["num_hidden_layers"], name
            assert all(b["count"] > 0 for b in cfg["blocks"]), name
        elif cfg["arch"] != "mla_moe":
            assert cfg["head_dim"] > 0 and cfg["num_key_value_heads"] > 0, name
            assert cfg["num_attention_heads"] % cfg["num_key_value_heads"] == 0, name
        else:
            for key in ("kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"):
                assert cfg[key] > 0, (name, key)
            assert cfg["q_lora_rank"] is None or cfg["q_lora_rank"] > 0, name
        if cfg["arch"].endswith("_moe"):
            assert cfg["num_experts_per_tok"] > 0, name
            if "num_experts" in cfg:
                assert cfg["num_experts_per_tok"] <= cfg["num_experts"], name
            assert cfg["expert_intermediate_size"] > 0, name
            assert cfg["num_shared_experts"] >= 0, name
            assert cfg["shared_expert_intermediate_size"] >= 0, name
            assert not cfg["num_shared_experts"] or cfg["shared_expert_intermediate_size"] > 0, name
        if "ffn_dim_multiplier" in cfg:
            width = int(cfg["ffn_dim_multiplier"] * int(8 * cfg["hidden_size"] / 3))
            multiple = cfg["multiple_of"]
            assert cfg["intermediate_size"] == multiple * ((width + multiple - 1) // multiple), name
        result = calc_model_metrics(cfg)
        assert result["attention_macs"] > 0 and result["active_ffn_macs"] > 0, name
        assert math.isfinite(result["r_arch"]) and result["r_arch"] > 0, name
        assert result["kappa"] > 1, name
        if "reference_ratio" in cfg:
            assert abs(result["r_arch"] - cfg["reference_ratio"]) < 0.001, (name, result)
        elif not 0.1 <= result["r_arch"] <= 0.7:
            raise ValueError(f"{name}: A/F={result['r_arch']:.4g}; audit the geometry and "
                             "record a checked reference_ratio before plotting this outlier")
    # Regress the audited sizes, without forcing future family members into a cluster.
    for names in (("Llama-3.1-8B", "Llama-3.1-70B", "Llama-3.1-405B"),
                  ("Qwen3-8B", "Qwen3-14B", "Qwen3-32B")):
        ratios = [calc_model_metrics(models[name])["r_arch"] for name in names if name in models]
        if ratios:
            assert max(ratios) / min(ratios) < 1.5, (names, ratios)
    for name, cfg in hardware.items():
        assert cfg["kind"] in ("nominal", "measured") and cfg["source"], name
        assert cfg["compute_tops"] > 0 and cfg["bandwidth_gbs"] > 0, name
        assert math.isfinite(calc_hardware_balance(cfg)), name


def validate_calculators():
    # Independent small matrix counts catch family-specific normalization errors.
    cfg = {"hidden_size": 8, "num_attention_heads": 4, "num_key_value_heads": 2,
           "head_dim": 2, "intermediate_size": 16, "mlp_type": "swiglu"}
    assert calc_dense_attention_macs(cfg) == 8 * 8 + 2 * 8 * 4 + 8 * 8
    assert calc_dense_attention_macs({**cfg, "num_key_value_heads": 4}) == 4 * 8 * 8
    assert calc_dense_attention_macs({**cfg, "num_key_value_heads": 1}) == 2 * 8 * (8 + 2)
    assert calc_dense_ffn_macs(cfg) == 384
    assert calc_dense_ffn_macs({**cfg, "mlp_type": "geglu"}) == 384
    assert calc_dense_ffn_macs({**cfg, "mlp_type": "gelu"}) == 256
    assert calc_dense_ffn_macs({**cfg, "mlp_type": "relu"}) == 256
    moe = {**cfg, "expert_intermediate_size": 16, "num_experts_per_tok": 2,
           "num_shared_experts": 1, "shared_expert_intermediate_size": 4, "num_experts": 128}
    assert calc_moe_ffn_macs(moe) == 3 * 8 * (16 * 2 + 4)
    assert calc_moe_ffn_macs({**moe, "num_experts": 256}) == calc_moe_ffn_macs(moe)
    mla = {"hidden_size": 8, "num_attention_heads": 2, "q_lora_rank": 3,
           "kv_lora_rank": 2, "qk_nope_head_dim": 2, "qk_rope_head_dim": 1, "v_head_dim": 2}
    assert calc_mla_attention_macs(mla) == 24 + 18 + 24 + 16 + 32
    assert calc_mla_attention_macs({**mla, "q_lora_rank": None}) == 48 + 24 + 16 + 32
    a = calc_dense_attention_macs(cfg)
    f = calc_dense_ffn_macs(cfg)
    doubled = {**cfg, "hidden_size": 16, "head_dim": 4, "intermediate_size": 32}
    assert calc_dense_attention_macs(doubled) == 4 * a
    assert calc_dense_ffn_macs(doubled) == 4 * f


def load_mixerloop_configs():
    configs = {}
    for name, relative_path in MIXERLOOP_CONFIGS.items():
        data = (ROOT.parent / relative_path).read_bytes()
        cfg = json.loads(data)
        cfg.update(source=relative_path, source_revision="sha256:" + hashlib.sha256(data).hexdigest())
        configs[name] = cfg
    return configs


def calc_token_point(cfg, loop_count=None):
    """Whole decoder, one token: attention/mixer MACs and active FFN bytes."""
    loops = cfg.get("loop_count", 1) if loop_count is None else loop_count
    layers = cfg["num_hidden_layers"]
    if cfg.get("model_type") == "mixerloop":
        a, f = calc_mixerloop_macs(cfg)
        a, f = layers * loops * a, layers * f
    else:
        result = calc_model_metrics(cfg)
        a, f = result["attention_macs"], result["active_ffn_macs"]
        if cfg["arch"] != "hybrid":
            # MLA/MoE registries describe routed blocks; early dense blocks still
            # execute and stream their own FFNs on every token.
            dense = cfg.get("dense_prefix_layers", 0)
            a *= layers
            f = (layers - dense) * f
            if dense:
                f += dense * calc_dense_ffn_macs(cfg)
        a, f = loops * a, loops * f
    traffic = WEIGHT_BYTES * f
    return {"loop_count": loops, "attention_macs_per_token": a,
            "ffn_macs_per_token": f, "ffn_weight_bytes_per_token": traffic,
            "traffic_gb": traffic / 1e9, "mixer_gmac": a / 1e9}


def export_metrics_csv(model_metrics, configs, points, mixer_points, output=OUTPUT):
    output.mkdir(parents=True, exist_ok=True)
    path = output / "model_architecture_metrics.csv"
    fields = ["model", "family", "architecture", "loop_count", "num_hidden_layers",
              "attention_macs_per_token", "ffn_macs_per_token", "ffn_weight_bytes_per_token",
              "traffic_gb", "mixer_gmac", "whole_model_attention_ffn_ratio",
              "per_block_attention_ffn_ratio", "main_figure", "source", "source_revision",
              "implementation_source", "ffn_source", "dependency_source", "notes"]
    rows = [(n, p, MODELS[n], model_metrics[n]["r_arch"]) for n, p in points.items()]
    rows += [(f"{n} T={t}", p, {**configs[n], "family": "MixerLoop", "arch": "mixerloop"},
              calc_mixerloop_metrics(configs[n], t)["r_arch"])
             for n, by_loop in mixer_points.items() for t, p in by_loop.items()]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, p, cfg, ratio in sorted(rows, key=lambda row: row[1]["traffic_gb"]):
            writer.writerow({
                "model": name, "family": cfg["family"], "architecture": cfg["arch"],
                "num_hidden_layers": cfg["num_hidden_layers"], **p,
                "main_figure": cfg.get("main_figure", True)
                    and (cfg["arch"] != "mixerloop" or p["loop_count"] in (1, 4)),
                "whole_model_attention_ffn_ratio":
                    p["attention_macs_per_token"] / p["ffn_macs_per_token"],
                "per_block_attention_ffn_ratio": ratio,
                **{key: cfg.get(key, "") for key in fields[-6:]},
            })
    print(path)


def export_hardware_csv(hardware_metrics, output=OUTPUT):
    path = output / "hardware_balance_metrics.csv"
    fields = ["hardware", "compute_tops", "bandwidth_gbs", "precision", "kind",
              "required_ai_mac_per_byte", "mixer_wall_slope_mac_per_byte",
              "source", "compute_source"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, required_ai in sorted(hardware_metrics.items(), key=lambda item: item[1]):
            cfg = HARDWARE[name]
            writer.writerow({"hardware": name, **{k: cfg[k] for k in fields[1:5]},
                             "required_ai_mac_per_byte": required_ai,
                             "mixer_wall_slope_mac_per_byte": required_ai - 1 / WEIGHT_BYTES,
                             "source": cfg["source"], "compute_source": cfg.get("compute_source", "")})
    print(path)


FAMILY_COLORS = {"Dense": "#B87563", "MoE": "#7896A3"}
FAMILY_FILLS = {"Dense": "#F4EBE7", "MoE": "#EDF2F2"}
MARKERS = ("o", "s", "^", "v", "D", "p", "h", "8", "<", ">", "*",
           "d", "H", "P", "X", "+", "x", "1", "2", "3", "4")
WALL_HARDWARE = "KV260 1HP"


def point_xy(point):
    return point["traffic_gb"], point["mixer_gmac"]


def ellipse_region(points, padding=1.2):
    """Axis-aligned ellipse in log10 space, enclosing the padded point range."""
    z = np.log10(points)
    low, high = z.min(axis=0), z.max(axis=0)
    center = (low + high) / 2
    radii = (high - low) / 2 + np.log10(padding)
    # A padded bounding box's ellipse need not contain its corners. Expand
    # only when required to enclose an observed point; never move the points.
    radius_scale = max(1.0, np.linalg.norm((z - center) / radii, axis=1).max())
    radii = radii * radius_scale + np.log10(1.03)  # Keep marker edges off the outline.
    theta = np.linspace(0, 2 * np.pi, 361)
    path = 10 ** (center + np.column_stack((np.cos(theta), np.sin(theta))) * radii)
    return path, center, radii


def plot_clusters(ax, points):
    for family, color in FAMILY_COLORS.items():
        members = [p for n, p in points.items()
                   if ("MoE" if MODELS[n]["arch"].endswith("_moe") else "Dense") == family]
        if not members:
            continue
        path, center, radii = ellipse_region([point_xy(p) for p in members])
        ax.fill(path[:, 0], path[:, 1], color=FAMILY_FILLS[family], linewidth=0, zorder=1)
        ax.plot(path[:, 0], path[:, 1], color=color, lw=0.7, zorder=2)
        label = "Dense" if family == "Dense" else "MoE: sparse active FFN"
        # Labels follow the observed ellipse, without changing its data bounds.
        xy = (10 ** center[0], 10 ** (center[1] - radii[1]))
        ax.annotate(label, xy, xytext=(0, -5), textcoords="offset points",
                    ha="center", va="top", color=color, fontsize=5.6)
        print(f"{family} ellipse: x={path[:, 0].min():.5f}..{path[:, 0].max():.5f}, "
              f"y={path[:, 1].min():.6f}..{path[:, 1].max():.6f}")


def plot_models(ax, points):
    handles = {family: [] for family in FAMILY_COLORS}
    for index, (name, p) in enumerate(points.items()):
        cfg = MODELS[name]
        family = "MoE" if cfg["arch"].endswith("_moe") else "Dense"
        marker = MARKERS[index] if index < len(MARKERS) else f"${index + 1}$"
        line, = ax.plot(*point_xy(p), marker=marker, ms=3.8, mfc="white",
                        color=FAMILY_COLORS[family], mew=0.75, linestyle="none",
                        label=cfg.get("plot_label", name), zorder=5)
        handles[family].append(line)
    return handles


def mixerloop_sweep_curves(mixer_points, wall_slope, samples=300):
    """Schematic log-space fan; not a fixed-T model law or measured scaling."""
    profiles = sorted(mixer_points.values(), key=lambda p: p[1]["traffic_gb"])
    start, end = profiles[0][1], profiles[-1][1]
    x = np.geomspace(start["traffic_gb"], end["traffic_gb"], samples)
    u = np.log(x / x[0]) / np.log(x[-1] / x[0])
    # Keep the actual anchor, which lies slightly below the hardware wall.
    anchor_y = profiles[0][4]["mixer_gmac"]
    offset = np.log(anchor_y / (wall_slope * x[0]))
    center = wall_slope * x * np.exp(offset * (1 - u))
    bend = 1.2 * u**2
    return x, (center * np.exp(-bend), center * np.exp(-0.18 * u**2),
               center * np.exp(bend))


def plot_mixerloop(ax, mixer_points, wall_slope):
    profiles = sorted(mixer_points.items(), key=lambda item: item[1][1]["traffic_gb"])
    name, anchor = profiles[0]
    start, finish = anchor[1], anchor[4]
    ax.annotate("", xy=point_xy(finish), xytext=point_xy(start),
                arrowprops={"arrowstyle": "-|>", "color": MIXER_COLOR, "lw": 1.4,
                            "shrinkA": 3, "shrinkB": 3, "mutation_scale": 8},
                zorder=5)
    ax.plot(*point_xy(start), marker="o", ms=3.7, mfc="white", mec=MIXER_COLOR,
            mew=0.95, linestyle="none", zorder=6)
    ax.plot(*point_xy(finish), marker="D", ms=4.1, color=MIXER_COLOR,
            linestyle="none", zorder=6)
    ax.annotate(name.removeprefix("MixerLoop "), point_xy(start), xytext=(0, -7),
                textcoords="offset points", color=MIXER_COLOR, fontsize=5.0,
                va="top", ha="center")

    x, curves = mixerloop_sweep_curves(mixer_points, wall_slope)
    styles = [(MIXER_COLOR, 1.0, "-"), (MIXER_COLOR, 0.65, (0, (3, 1.5))),
              (AUX_COLOR, 1.0, (0, (3, 1.5)))]
    for y, (color, alpha, linestyle) in zip(curves, styles):
        visible = y <= ax.get_ylim()[1] * 0.93
        ax.plot(x[visible], y[visible], color=color, lw=1.2,
                alpha=alpha, linestyle=linestyle, zorder=4)

    # One cross-trajectory arrow: changing budget, not following any one curve.
    arrow_x = 0.09
    lower_y = np.exp(np.interp(np.log(arrow_x), np.log(x), np.log(curves[0])))
    upper_y = np.exp(np.interp(np.log(arrow_x), np.log(x), np.log(curves[-1])))
    ax.annotate("", xy=(arrow_x, upper_y), xytext=(arrow_x, lower_y),
                arrowprops={"arrowstyle": "-|>", "connectionstyle": "arc3,rad=0.28",
                            "color": MIXER_COLOR, "lw": 0.9, "mutation_scale": 7,
                            "shrinkA": 0, "shrinkB": 0}, zorder=6)

    y = curves[0]
    for name, by_loop in profiles[1:]:
        scale_x = by_loop[1]["traffic_gb"]
        scale_y = np.exp(np.interp(np.log(scale_x), np.log(x), np.log(y)))
        ax.plot(scale_x, scale_y, marker="o", ms=2.8, color=MIXER_COLOR,
                linestyle="none", zorder=6)
        ax.annotate(name.removeprefix("MixerLoop ").replace("~", ""),
                    (scale_x, scale_y),
                    xytext=(-2, -12) if name == profiles[1][0] else (3, 4),
                    textcoords="offset points", color=MIXER_COLOR, fontsize=5.0,
                    va="top" if name == profiles[1][0] else "bottom",
                    ha="right" if name == profiles[1][0] else "left")
    label_point = point_xy(profiles[1][1][4])
    ax.annotate("MixerLoop", label_point, xytext=(-40, 19),
                textcoords="offset points", color=MIXER_COLOR,
                fontsize=6.4, weight="semibold", ha="center")
    ax.annotate("T=1 → 4", label_point, xytext=(-40, 10),
                textcoords="offset points", color=MIXER_COLOR,
                fontsize=5.2, ha="center")


def plot_model_legend(fig, handles):
    # Two rows read horizontally: 2 MoE + 3 Dense, then 3 MoE + 2 Dense.
    moe, dense = handles["MoE"], handles["Dense"]
    first = len(moe) // 2
    split = (len(dense) + 1) // 2
    rows = [moe[:first] + dense[:split], moe[first:] + dense[split:]]
    columns = []
    for column in zip_longest(*rows):
        entries = []
        for handle in column:
            if handle is None:
                entries.append(DrawingArea(0, 6, 0, 0))
                continue
            symbol = DrawingArea(5, 6, 0, 0)
            symbol.add_artist(Line2D([2.5], [3], marker=handle.get_marker(),
                                    markersize=3.5, markerfacecolor="white",
                                    markeredgecolor=handle.get_color(),
                                    markeredgewidth=0.7, linestyle="none"))
            label = TextArea(handle.get_label(),
                             textprops={"size": 4.2, "color": "#555753"})
            entries.append(HPacker(children=[symbol, label], align="center",
                                   pad=0, sep=0.5))
        columns.append(VPacker(children=entries, align="left", pad=0, sep=5))
    content = HPacker(children=columns, align="top", pad=0, sep=2)
    legend = AnchoredOffsetbox(loc="lower center", child=content, pad=0.65,
                              borderpad=0, frameon=True,
                              bbox_to_anchor=(0.5, 0.085), bbox_transform=fig.transFigure)
    legend.patch.set(boxstyle="square,pad=0", facecolor="white",
                     edgecolor="#B9B7B0", linewidth=0.55)
    fig.add_artist(legend)


def plot_memory_wall(ax, hardware_metrics, x_limits):
    # Y excludes FFN MACs, so subtract F=D/b_w from the total-compute budget.
    slope = hardware_metrics[WALL_HARDWARE] - 1 / WEIGHT_BYTES
    assert slope > 0
    xs = np.geomspace(*x_limits, 300)
    ax.plot(xs, slope * xs, color=WALL_COLOR, lw=1.15, zorder=2)
    label_x = x_limits[0] * (x_limits[1] / x_limits[0]) ** 0.3
    # Log-log turns every physical C=slope*D line into a slope-one diagonal;
    # larger required AI moves this line upward rather than rotating it.
    angle = math.degrees(math.atan2(
        ax.bbox.height / np.log10(ax.get_ylim()[1] / ax.get_ylim()[0]),
        ax.bbox.width / np.log10(x_limits[1] / x_limits[0])))
    ax.annotate("Memory Wall", (label_x, slope * label_x), xytext=(0, 4),
                textcoords="offset points", rotation=angle, rotation_mode="anchor",
                color="#B28D24", fontsize=6.5, ha="center", va="bottom")


def make_figure(points, mixer_points, hardware_metrics):
    fig, ax = plt.subplots(figsize=(3.3, 3.3))
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.32, top=0.89)
    points = {n: p for n, p in points.items() if MODELS[n].get("main_figure", True)}
    x_limits = (0.002, 0.6)
    y_limits = (0.00085, 0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlim=x_limits, ylim=y_limits,
           xlabel="FFN Weight Traffic per Token (GB)",
           ylabel="Attention / Mixer Compute per Token (GMAC)")
    # ax.set_title("Climbing the Memory Wall", loc="center", fontsize=9,
    #              weight="semibold", color=MIXER_COLOR, pad=9)
    fig.suptitle("Climbing the Memory Wall", fontsize=9, weight="semibold",
             color=MIXER_COLOR, y=0.96)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(labelsize=6, pad=2)
    ax.xaxis.label.set_fontsize(6.4)
    ax.yaxis.label.set_fontsize(6.4)
    plot_memory_wall(ax, hardware_metrics, x_limits)
    plot_clusters(ax, points)
    handles = plot_models(ax, points)
    plot_mixerloop(ax, mixer_points, hardware_metrics[WALL_HARDWARE] - 1 / WEIGHT_BYTES)
    plot_model_legend(fig, handles)
    return fig, ax


def validate_token_points(points, mixer_points, configs):
    for name, p in points.items():
        cfg = MODELS[name]
        assert p["attention_macs_per_token"] > 0 and p["ffn_macs_per_token"] > 0
        assert math.isclose(p["ffn_weight_bytes_per_token"], WEIGHT_BYTES * p["ffn_macs_per_token"])
        if "reference_point" in cfg:
            assert np.allclose(point_xy(p), cfg["reference_point"], rtol=0.005, atol=0), (name, p)
        if cfg.get("recurrence") == "full_block":
            single = calc_token_point(cfg, 1)
            assert math.isclose(p["traffic_gb"], p["loop_count"] * single["traffic_gb"])
            assert math.isclose(p["mixer_gmac"], p["loop_count"] * single["mixer_gmac"])
    for name, by_loop in mixer_points.items():
        for t, p in by_loop.items():
            assert p["traffic_gb"] == by_loop[1]["traffic_gb"]
            assert math.isclose(p["mixer_gmac"], t * by_loop[1]["mixer_gmac"])
        m, f = calc_mixerloop_macs(configs[name])
        assert by_loop[1]["attention_macs_per_token"] == configs[name]["num_hidden_layers"] * m
        assert by_loop[1]["ffn_macs_per_token"] == configs[name]["num_hidden_layers"] * f
    # At the corrected wall, total compute time equals FFN transfer time.
    for cfg in HARDWARE.values():
        required_ai = calc_hardware_balance(cfg)
        d = 12_345_678
        a = (required_ai - 1 / WEIGHT_BYTES) * d
        assert math.isclose((a + d / WEIGHT_BYTES) / required_ai, d)


def validate_intensities(model_metrics, mixer_metrics, hardware_metrics):
    for name, cfg in MODELS.items():
        result = model_metrics[name]
        assert math.isclose(result["effective_intensity"], result["kappa"] / WEIGHT_BYTES)
        assert math.isclose(result["effective_intensity"],
                            result["compute_macs"] / result["ffn_weight_bytes"])
        if cfg.get("recurrence") == "full_block":
            single = calc_model_metrics({**cfg, "loop_count": 1})
            assert math.isclose(result["effective_intensity"], single["effective_intensity"])
            assert result["compute_macs"] == cfg["loop_count"] * single["compute_macs"]
            assert result["ffn_weight_bytes"] == cfg["loop_count"] * single["ffn_weight_bytes"]
    for by_loop in mixer_metrics.values():
        for t, result in by_loop.items():
            assert math.isclose(result["r_arch"], 8 / 11)
            assert math.isclose(result["effective_intensity"], (1 + t * 8 / 11) / WEIGHT_BYTES)
            assert result["ffn_weight_bytes"] == by_loop[1]["ffn_weight_bytes"]
    x_ref = hardware_metrics["KV260 1HP"]
    assert math.isclose(x_ref, 32 / 9, rel_tol=1e-4)
    assert hardware_metrics["KV260 4HP"] < hardware_metrics["KV260 2HP"] < x_ref


def main():
    validate_calculators()
    validate_registry()
    model_metrics = {n: calc_model_metrics(c) for n, c in MODELS.items()}
    configs = load_mixerloop_configs()
    mixer_metrics = {n: {t: calc_mixerloop_metrics(c, t) for t in LOOP_COUNTS}
                     for n, c in configs.items()}
    hardware_metrics = {n: calc_hardware_balance(c) for n, c in HARDWARE.items()}
    validate_intensities(model_metrics, mixer_metrics, hardware_metrics)
    points = {n: calc_token_point(c) for n, c in MODELS.items()}
    mixer_points = {n: {t: calc_token_point(c, t) for t in LOOP_COUNTS}
                    for n, c in configs.items()}
    validate_token_points(points, mixer_points, configs)
    rows = [(n, p) for n, p in points.items() if MODELS[n].get("main_figure", True)]
    rows += [(f"{n} T={t}", p) for n, group in mixer_points.items()
                                  for t, p in group.items()]
    print(f"{'Model':<30} {'FFN GB/token':>14} {'Mixer GMAC/token':>18}")
    for n, p in sorted(rows, key=lambda item: item[1]["traffic_gb"]):
        print(f"{n:<30} {p['traffic_gb']:14.6f} {p['mixer_gmac']:18.6f}")
    print(f"Memory Wall ({WALL_HARDWARE}): slope="
          f"{hardware_metrics[WALL_HARDWARE] - 1 / WEIGHT_BYTES:.6f} MAC/B (FFN budget removed)")
    export_metrics_csv(model_metrics, configs, points, mixer_points)
    export_hardware_csv(hardware_metrics)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.65, "pdf.fonttype": 42,
                         "text.color": TEXT_COLOR, "axes.labelcolor": TEXT_COLOR,
                         "axes.edgecolor": TEXT_COLOR, "xtick.color": TEXT_COLOR,
                         "ytick.color": TEXT_COLOR})
    fig, _ = make_figure(points, mixer_points, hardware_metrics)
    for extension in ("pdf", "png"):
        path = OUTPUT / f"exp2_memory_wall.{extension}"
        fig.savefig(path, dpi=300, facecolor="white")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
