# MixerLoop hardware: T=1 baseline

This directory is the clean starting milestone for MixerLoop hardware work. It
is the verified `gdn.hls` single-kernel Gated DeltaNet decoder, imported from
commit `45a7e14` without its nested Git history, generated outputs, or example
model binaries.

It intentionally implements **T=1 only**. Loop scheduling, loop-aware state,
and hiding recurrent mixer computation behind FFN weight transfer belong to the
next hardware milestone. Keeping this baseline unchanged makes future resource,
timing, and throughput comparisons meaningful.

## Fixed profile

```text
part=xck26-sfvc784-2LV-c  clock=150 MHz
dim=256  hidden_dim=768  layers=8  heads=8
head_k_dim=32  head_v_dim=32  conv_size=4
vocab_size=32000  maximum_seq_len=1024
W8A8 symmetric int8  group_size=32  checkpoint_version=2
```

The accelerator has one `decode` kernel, one HP0 parameter interface, one Q8
linear engine, and one persistent GDN recurrent state. The host uploads the
packed weights once and launches one complete autoregressive decode step per
token.

## Validated baseline

The imported M2 milestone was validated on KV260 with Vitis/Vivado 2025.2:

- kernel simulation matched the CPU Q8 reference for 16/16 steps;
- HLS estimated 5.231 ns at a 150 MHz target;
- HLS resources were 106 BRAM18, 415 DSP, 99,982 FF, 91,928 LUT, 16 URAM;
- routed WNS was +0.572 ns with TNS 0;
- measured steady-state decode was 108.849 token/s.

These numbers describe the T=1 baseline, not a looped MixerLoop accelerator.

## Source layout

```text
src/config.hpp       fixed profile and packed tensor offsets
src/weight.cpp       version-2 Q8 loader and 512-bit host packing
src/decode.cpp       HLS kernel plus matching CPU Q8 reference
src/main.cpp         CPU/XRT host
model/               canonical TinyStories T=1 and T=4 GDNe v2 weights
tools/kernel_sim.cpp synthesizable kernel versus CPU reference
scripts/             build, link, package, and optional deployment helpers
```

## Reproducible build

All scripts derive paths from their own location. Generated files are written
under `hardware/outputs/` and ignored by the root repository.

Build the CPU reference:

```bash
cmake -S hardware -B hardware/outputs/cpu
cmake --build hardware/outputs/cpu -j
hardware/outputs/cpu/gdn_host \
  --weight_path /path/to/gdn-v2-model.q8.bin \
  --vocab_path /path/to/tokenizer.bin \
  --max_seq 16 --temp 0 -i "Once"
```

Build and run native kernel simulation:

```bash
source /opt/xilinx/2025.2/Vitis/settings64.sh
bash hardware/scripts/build_sim.sh /path/to/gdn-v2-model.q8.bin 16
```

Run HLS and the KV260 platform link:

```bash
bash hardware/scripts/build_hls.sh
bash hardware/scripts/build_link.sh
```

Expected outputs are:

```text
hardware/outputs/hls/decode/work/decode.xo
hardware/outputs/link/binary_container_1.xclbin
```

The optional AArch64 host needs an extracted Xilinx common image:

```bash
COMMON=/path/to/xilinx-zynqmp-common-v2022.2 \
  bash hardware/scripts/build_host.sh
```

Package a future board bundle from an existing v2 weight directory:

```bash
bash hardware/scripts/package_bundle.sh /path/to/model-directory
```

`deploy_kv260.sh` and `restore_starter_kit.sh` require configured SSH access and
passwordless `sudo`; no board password or private artifact is stored here.

## Checkpoint boundary

The T=1 hardware loader and MixerLoop training use the same established
version-2 `GDNe` checkpoint. The header records the training context length;
the loader accepts values up to the hardware maximum of 1,024. Loop count is
not encoded in the weight file. The next hardware phase will add T=4 scheduling
from this known-good T=1 checkpoint.
