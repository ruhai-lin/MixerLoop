# MixerLoop hardware: T=1 / T=4 on one bitstream

One production `decode` kernel serves both TinyStories 15M checkpoints.
Runtime `loop_count` (AXI-lite, also stored in the GDNe v2 header pad) selects
T=1 or T=4. Compute engines stay ×1; T=4 uses extra loop-slot state and hides
the extra Mixer passes behind FFN weight prefetch into an on-chip ring.

## Fixed profile

```text
part=xck26-sfvc784-2LV-c  clock=150 MHz
dim=256  hidden_dim=768  layers=8  heads=8
head_k_dim=32  head_v_dim=32  conv_size=4
vocab_size=32000  maximum_seq_len=1024
W8A8 symmetric int8  group_size=32  checkpoint_version=2
loop_count=1..4   T=4 enables Mixer/FFN overlap
```

## Process graph

Nine persistent DATAFLOW processes, each synthesized once:

```text
controller  memory  scratch  weight_router
q8          beta    conv     rec            post
```

Rules: one Q8 call site; physical HP0 only in `memory`; Q8 consumes a unified
`weight_stream`; AXI vs SRAM mux stays in `weight_router`; recurrent S stays
inside `rec`. Command FIFOs are SRL (depth 16 on hardware, 256 in C-sim).

T=1 (`loop_count=1`) is native GDN: loop0 Mixer from HP0, then FFN from HP0.
T=4 (`loop_count=4`) per physical layer:

```text
loop0: HP0 Mixer → Q8 + Mixer scratch[0,5759]
loop1–3: scratch replay Mixer while HP0 prefetches FFN into scratch[5760,12287]
then FFN: ring drain + 3840-word HP0 tail
```

Semantics match the CPU oracle:

```text
for layer:
  for loop in T:
    h = h + GDN(RMSNorm(h))   # independent S/conv per loop slot
  h = h + FFN(FFNNorm(h))
```

## Kernel ABI

```c
void decode(int token, int reset_state, int loop_count,
            const ap_uint<128>* packed_params, const float* side,
            uint32_t* next_token, uint32_t* stats);
```

`stats[0]` is the FFN ring occupancy high-water mark (0 for T=1, 6528 for T=4).

## Header `loop_count`

GDNe v2 header pad (file offset 48, little-endian int32) stores T. Legacy 0
means unspecified: the host then requires `--loop_count`. A CLI value that
disagrees with a non-zero header is an error. Tensor bytes after the 256-byte
header are unchanged.

## Validated results (Vitis/Vivado 2025.2, KV260)

Kernel-sim vs CPU Q8 oracle:

| Check | Result |
|---|---|
| T=1 16/16 | pass, `ring_max=0` |
| T=4 16/16 | pass, `ring_max=6528` |
| T=1/T=4 reset and T cross-switch | pass |

HLS csynth (`xck26-sfvc784-2LV-c`, 150 MHz):

| Resource | Used | Available | % |
|---|---:|---:|---:|
| LUT | 91542 | 117120 | 78 |
| FF | 78240 | 234240 | 33 |
| DSP | 292 | 1248 | 23 |
| BRAM18 | 274 | 288 | 95 |
| URAM | 56 | 64 | 87 |

Each of `q8` / `beta` / `conv` / `rec` / `post` / `scratch` / `memory` /
`router` / `controller` is a single instance. Recurrence is 32 URAM / 121 DSP;
Mixer+FFN scratch is 24 URAM.

Routed implementation:

| Item | Value |
|---|---|
| Clock | 150 MHz (`clk_out1` period 6.667 ns) |
| WNS / TNS | **+0.340 ns / 0** |
| Kernel LUT / REG | 57859 / 70688 |
| Kernel BRAM tiles / URAM / DSP | 103 / 56 / 309 |

KV260, prompt `Once`, temp 0, same bitstream:

| Run | Text vs CPU | tok/s | cycles/token | `ring_max` |
|---|---|---:|---:|---:|
| T=1 n=16 | match | 116.01 | — | 0 |
| T=4 n=16 | match | 88.17 | — | 6528 |
| T=1 n=128 | — | **116.40** | 1.289e6 | 0 |
| T=4 n=128 | — | **88.45** | 1.696e6 | 6528 |

T=4 is 76% of T=1 throughput: the extra three Mixer passes are partly hidden
behind FFN prefetch, but not fully. One bitstream, one HP0.

## Source layout

```text
src/config.hpp       fixed profile and packed tensor offsets
src/weight.cpp       version-2 Q8 loader (header pad = loop_count)
src/decode.cpp       HLS kernel plus matching CPU Q8 reference
src/main.cpp         CPU/XRT host (`--loop_count`)
model/               TinyStories T=1 and T=4 GDNe v2 weights
tools/kernel_sim.cpp synthesizable kernel versus CPU reference
scripts/             build, link, package, and KV260 deploy helpers
```

## Reproducible build

All scripts derive paths from their own location. Generated files are written
under `hardware/outputs/` and ignored by the root repository.

Build the CPU reference:

```bash
cmake -S hardware -B hardware/outputs/cpu
cmake --build hardware/outputs/cpu -j
hardware/outputs/cpu/gdn_host \
  --weight_path hardware/model/tinystories15m_t1_q8.bin \
  --vocab_path hardware/model/tokenizer.bin \
  --max_seq 16 --temp 0 -i "Once"
```

Build and run native kernel simulation:

```bash
source /opt/xilinx/2025.2/Vitis/settings64.sh
bash hardware/scripts/build_sim.sh hardware/model/tinystories15m_t1_q8.bin 16
bash hardware/scripts/build_sim.sh hardware/model/tinystories15m_t4_q8.bin 16
bash hardware/scripts/build_sim.sh hardware/model/tinystories15m_t4_q8.bin 4 4 gates
```

Run HLS and the KV260 platform link (all kernel memory ports → physical HP0):

```bash
bash hardware/scripts/build_hls.sh
bash hardware/scripts/build_link.sh
```

Expected outputs:

```text
hardware/outputs/hls/decode/work/decode.xo
hardware/outputs/link/binary_container_1.xclbin
```

AArch64 host (extracted Xilinx common image):

```bash
COMMON=/path/to/xilinx-zynqmp-common-v2022.2 \
  bash hardware/scripts/build_host.sh
```

Package and deploy. `SUDO_PASSWORD` is only needed when the board account is
not passwordless sudo; it is not stored in the repo.

```bash
bash hardware/scripts/package_bundle.sh hardware/model
RESTORE_STARTER=0 SUDO_PASSWORD=ubuntu bash hardware/scripts/deploy_kv260.sh
```

Board smoke:

```bash
cd ~/Projects/gdn_bundle
./gdn_host --weight_path model/tinystories15m_t1_q8.bin \
  --vocab_path model/tokenizer.bin \
  --xclbin binary_container_1.bin -i "Once" -n 16 -t 0
./gdn_host --weight_path model/tinystories15m_t4_q8.bin \
  --vocab_path model/tokenizer.bin \
  --xclbin binary_container_1.bin -i "Once" -n 128 -t 0
```

Restore the quiet starter app when measurements are done:

```bash
SUDO_PASSWORD=ubuntu bash hardware/scripts/restore_starter_kit.sh
```
