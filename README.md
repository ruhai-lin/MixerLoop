<div align="center">

# MixerLoop

**Recurrent compute allocation for Gated Delta Networks, from pretraining to FPGA build.**

</div>

MixerLoop repeats each Gated DeltaNet token mixer while executing the FFN once.
The repeated calls share the physical layer's mixer parameters, so increasing
the loop count allocates more recurrent computation without adding mixer
weights. This repository is the end-to-end implementation: FLAME pretraining,
Hugging Face checkpoints, W8A8 deployment export, a C++ reference, Vitis HLS,
and the KV260 Vivado/Vitis build.

The current release intentionally validates one deployment profile. Other model
sizes remain configurable through Transformers JSON files, but are not shipped
as tuned presets.

## Architecture and deployment profile

For physical layer `i`, MixerLoop computes

```text
for loop_slot in range(T):
    h_input = h
    h = h + GDN_i(RMSNorm_i(h))
    h = h + residual_weight[loop_slot] * h_input
h = h + FFN_i(FFNNorm_i(h))
```

The mixer and its norm are shared across loop calls. During autoregressive
decode, each `(layer, loop_slot)` owns an independent recurrent state. T=1 and
T=4 therefore use the same weight layout, accelerator datapath, and xclbin;
only the runtime loop count and active state slots differ.

The validated 15M-class profile is:

| Field | Value |
|---|---:|
| hidden size | 256 |
| physical layers | 8 |
| heads / head dimension | 8 / 32 |
| FFN intermediate size | 768 |
| short convolution | 4 |
| tokenizer / vocabulary | Llama 2 / 32,000 |
| maximum hardware loop count | 4 |
| deployment arithmetic | W8A8, group size 32 |

## Environment

Run the project inside the WSL Linux filesystem, such as
`/home/<user>/Projects/MixerLoop`, rather than `/mnt/c`. Python uses one virtual
environment at the repository root.

The local validation environment is Ubuntu 22.04, Python 3.10.12, PyTorch
2.13.0+cu130, an RTX 4060 Laptop GPU, and AMD Vitis/Vivado 2025.2. Install the
PyTorch wheel that matches the target machine first; the remaining software
versions are pinned by `pyproject.toml`.

```bash
git clone https://github.com/ruhai-lin/MixerLoop.git
cd MixerLoop
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel ninja packaging
python -m pip install torch==2.13.0
python -m pip install -e '.[dev]'
python -m pip check
python -m pytest -q
```

Training machines only need the Python/CUDA stack. HLS and xclbin generation
additionally require matching Vitis/Vivado and a KV260 platform; see
[`hardware/README.md`](hardware/README.md).

## TinyStories smoke training

`train.sh` defaults to the pinned TinyStories revision and the included Llama 2
tokenizer. The launcher saves the resolved model config before training and,
after the final distributed checkpoint, automatically writes both software and
hardware artifacts.

Run short T=1 and T=4 smoke jobs:

```bash
source .venv/bin/activate

LOOP_COUNT=1 STEPS=1000 NGPU=1 MICRO_BATCH=1 GLOBAL_BATCH=8 \
  OUTPUT=outputs/tinystories15M_t1 bash train.sh

LOOP_COUNT=4 STEPS=1000 NGPU=1 MICRO_BATCH=1 GLOBAL_BATCH=8 \
  OUTPUT=outputs/tinystories15M_t4 bash train.sh
```

For a very short pipeline check, set `STEPS=1 GLOBAL_BATCH=1 SEQ_LEN=64`.
Common overrides include `DATASET`, `DATASET_REVISION`, `DATASET_SPLIT`,
`TOKENIZER`, `SEQ_LEN`, `MICRO_BATCH`, `GLOBAL_BATCH`, `LEARNING_RATE`,
`WARMUP_STEPS`, and `OUTPUT`. Set `WANDB=1` to enable Weights & Biases.

Each output directory is self-contained:

```text
outputs/tinystories15M_t4/
├── checkpoint/step-1000/     # resumable Flame/TorchTitan DCP
├── config.json               # resolved HF config (including T)
├── model.safetensors         # full-precision HF checkpoint
├── model.q8.bin              # version-3 hardware checkpoint
├── tokenizer.model           # SentencePiece source
├── tokenizer.bin             # hardware tokenizer table
├── logs/
└── source/                   # training source snapshot
```

Load a produced HF checkpoint after the package has registered MixerLoop:

```python
import custom_models  # registers the architecture with Transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("outputs/tinystories15M_t4")
tokenizer = AutoTokenizer.from_pretrained("outputs/tinystories15M_t4")
```

## Hardware validation without a board

The current completion boundary is an offline build; KV260 deployment is not
required. After training, use either T checkpoint with the same hardware tree:

```bash
cmake -S hardware -B hardware/outputs/cpu
cmake --build hardware/outputs/cpu -j

bash hardware/scripts/build_sim.sh
hardware/outputs/sim/kernel_sim outputs/tinystories15M_t1/model.q8.bin 8
hardware/outputs/sim/kernel_sim outputs/tinystories15M_t4/model.q8.bin 8

source /opt/xilinx/2025.2/Vitis/settings64.sh
bash hardware/scripts/build_hls.sh
bash hardware/scripts/build_link.sh
```

Successful HLS synthesis, Vivado implementation, and a non-empty xclbin close
the offline flow. Detailed prerequisites, reports, cross-compiling the optional
host, and packaging are documented in the hardware README.

## Other datasets and evaluation

The FLAME data path accepts ordinary Hugging Face datasets, streaming datasets,
multiple datasets, and the deterministic pretokenized ClimbMix reader. Prepare
the optional fixed ClimbMix corpus with:

```bash
python -m flame.datasets.climbmix --data_dir data/climbmix-10b
```

CORE and lm-eval entry points remain available:

```bash
python eval/core_eval.py --model_path outputs/tinystories15M_t4
python eval/harness_eval.py --model_path outputs/tinystories15M_t4
```

### AAAI paper milestone

`references/main.tex` records the earlier AAAI submission milestone. Its model
sizes were exploratory and are not the current hardware profile, but its finite
readout ITR and synchronized BF16 throughput experiments remain important
research evidence. The corresponding evaluators are retained so those claims
can be reproduced from the submitted NoLoop, MixerLoop, and FullLoop HF
checkpoints.

Finite ITR disables only cross-token computation in a selected mixer, restores
contextual passes in execution order, and measures the final vocabulary
distribution. It reports participation-ratio ITR, marginal ITR, squared
Hellinger effect, and ground-truth next-token log-probability gain:

```bash
python eval/itr_eval.py \
  --model_paths outputs/gdn-15m outputs/mixerloop-15m outputs/fullloop-15m \
  --data_dir data/climbmix-10b \
  --out_dir outputs/itr-readout-15m
```

The paper throughput protocol uses BF16 prefill at length 1,024, final-token
logits, six warmup forwards, 20 synchronized timed forwards, and one GPU:

```bash
python eval/throughput_eval.py \
  --model_paths outputs/mixerloop-15m outputs/fullloop-15m \
  --batch_size 32 \
  --out_file outputs/throughput-15m-batch32.json
```

Install these optional dependencies with `python -m pip install -e '.[eval]'`.
FullLoop is retained only to load and evaluate the paper checkpoint; it is not
a shipped training preset or part of the current deployment path.

## Repository layout

```text
assets/tokenizer/        Llama 2 tokenizer source
configs/                 shipped MixerLoop profile
custom_models/mixerloop/ Transformers model implementation
flame/                   FLAME/TorchTitan training and final export
eval/                    CORE and lm-eval entry points
hardware/                C++ reference, HLS kernel, build and simulation tools
tests/                   architecture, data-resume and export contracts
```

The local `references/` directory contains development history and is ignored by
Git. Release behavior is defined only by files in the main repository.

## Acknowledgments

The training runtime builds on FLAME, TorchTitan, Flash Linear Attention, and
Transformers. The hardware flow targets AMD Vitis/Vivado and the KV260 platform.
