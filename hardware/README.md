# MixerLoop hardware: T=1 / T=4 on one bitstream

One `decode` kernel serves both TinyStories 15M checkpoints. Runtime
`loop_count` selects T=1 through T=4; the checkpoint header supplies the normal
default. Every setting uses the same arithmetic hardware. T>1 replays cached
Mixer weights while HP0 fetches the one non-recurrent FFN.

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

Nine fixed DATAFLOW processes, each synthesized once:

```text
schedule  memory  scratch  weight_router
q8        beta    conv     rec            post
```

`schedule` emits the fixed command sequence for this one model profile. It has
no queue, arbitration, cache policy, or runtime work discovery. `memory` is the
only HP0 owner. `weight_router` selects AXI or SRAM before the one shared Q8
engine. Recurrent and convolution state remain inside their owning processes.

T=1 (`loop_count=1`) is native GDN: loop0 Mixer from HP0, then FFN from HP0.
T=4 (`loop_count=4`) per physical layer:

```text
loop0: HP0 Mixer → Q8 and pinned scratch[0,5759]
loop1–3: scratch replay while HP0 fills the FFN ring scratch[5760,12287]
FFN: consume cached words and immediately reuse each released ring entry
```

The Mixer region remains pinned until the last loop. The 6,528-word FFN region
is a fixed ring with read pointer, write pointer, occupancy, and producer
credit. There is no separate “cached drain” and “DDR tail” phase: after the
last Mixer pass, consumption and the remaining HP0 transfer continue together.
No same-address URAM read/write behavior is assumed.

Q/K/V/G are produced one head at a time. Recurrence and post-processing consume
those heads as a producer-consumer pipeline. Since `head_dim == group_size ==
32`, each completed head is one full quantization group for O. The same Q8
engine therefore accumulates O in 144-word head slices; there is no second O
engine. Internal vector streams carry two FP32 values per 64-bit word.

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
            uint32_t* next_token);
```

The production ABI contains no trace or performance-counter buffer. The
instrumented development build used to close the schedule equations was
removed after validation.

## Header `loop_count`

GDNe v2 header pad (file offset 48, little-endian int32) stores T. Legacy 0
means unspecified: the host then requires `--loop_count`. A CLI value that
disagrees with a non-zero header is an error. Tensor bytes after the 256-byte
header are unchanged.

## Validated results (Vitis/Vivado 2025.2, KV260)

Kernel-sim vs CPU Q8 oracle:

| Check | Result |
|---|---|
| T=1 16/16 | pass, exact argmax vs CPU Q8 oracle |
| T=4 16/16 | pass, exact argmax vs CPU Q8 oracle |
| T=1/T=4 reset and T cross-switch | pass |

HLS csynth (`xck26-sfvc784-2LV-c`, 150 MHz):

| Resource | Used | Available | % |
|---|---:|---:|---:|
| LUT | 113277 | 117120 | 96 |
| FF | 111059 | 234240 | 47 |
| DSP | 477 | 1248 | 38 |
| BRAM18 | 283 | 288 | 98 |
| URAM | 56 | 64 | 87 |

Each of `q8` / `beta` / `conv` / `rec` / `post` / `scratch` / `memory` /
`router` / `schedule` is a single instance. Recurrence is 32 URAM; Mixer+FFN
scratch is 24 URAM. All scratch fill, replay, and consume loops achieve II=1.

Routed implementation:

| Item | Value |
|---|---|
| Clock | 150 MHz (`clk_out1` period 6.667 ns) |
| WNS / TNS | **+0.082 ns / 0** |
| Kernel LUT / REG | 74489 / 98416 |
| Kernel BRAM tiles / URAM / DSP | 104 / 56 / 493 |

KV260, prompt `Once`, temp 0, same bitstream:

| Run | Text vs CPU | tok/s |
|---|---|---:|
| T=1 n=16 | match | 116.323 |
| T=4 n=16 | match | 117.124 |
| T=1 n=128, three runs | match | 117.350, 119.549, 119.841 |
| T=4 n=128, three runs | match | 118.467, 118.484, 118.846 |

The steady-state medians are 119.549 tok/s for T=1 and 118.484 tok/s for
T=4. T=4 therefore retains **99.1%** of T=1 throughput: three extra Mixer
passes are hidden by the single-HP memory window to within host timing noise.
Both runs use the same bitstream and one physical HP0 port.

## Why the extra loops are hidden

HP0 supplies one 512-bit packed word every four cycles. Streaming the 10,368
FFN words therefore exposes a 41,472-cycle memory window. An instrumented build
measured one cached Mixer pass at 8,738 cycles, so three replays require 26,214
cycles and satisfy the compute-side contract `3*C_M <= 4*C_F`.

A 6,528-word ring has the stricter pre-consumer capacity bound `C_M <= 8,704`.
The measured Mixer misses that bound by 34 cycles, so HP0 briefly sees a full
ring before FFN starts. Consume-and-replace prevents this from becoming a
serialized tail: once FFN starts, every consumed entry releases producer
credit. The board result above is the final end-to-end check—the residual
boundary effect is below 1% of token throughput. The release kernel removes the
instrumentation used for these measurements.

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
RESTORE_STARTER=0 SUDO_PASSWORD='<board-password>' \
  bash hardware/scripts/deploy_kv260.sh
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
SUDO_PASSWORD='<board-password>' bash hardware/scripts/restore_starter_kit.sh
```

Compact release reports are archived under
`hardware/baselines/mixerloop_release/`. Generated build trees and deployment
bundles remain under `hardware/outputs/` and are never release inputs.
